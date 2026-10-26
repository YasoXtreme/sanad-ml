"""Person-grouped cross-validation of all five approaches on the same folds.

    uv run python -I -m sanad.train.run_cv                      # datasets from configs/train.yaml
    uv run python -I -m sanad.train.run_cv --datasets weda_fall # quick run on what's available

For every fold: trigger threshold (>= 99 % of training falls), windows, normalization stats and
decision thresholds come from the training subjects only. Results -> runs/<name>/results.{json,md}.
Approaches: threshold rule, Random Forest, teacher (1D ResNet), Sanad-Tiny alone, Sanad-Tiny distilled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from sanad.data.augment import Augment
from sanad.data.prepare import load_interim
from sanad.data.windowing import Windows, choose_threshold, extract_windows
from sanad.eval.metrics import event_metrics, threshold_for_sensitivity, window_metrics
from sanad.models.common import Normalize, channel_stats, count_macs, count_params
from sanad.models.student import SanadTiny
from sanad.models.teacher import SanadTeacher
from sanad.paths import CONFIGS, INTERIM, RUNS
from sanad.train.baselines import features, threshold_scores, train_forest
from sanad.train.loop import TrainConfig, pick_device, predict, train

APPROACHES = ["threshold", "forest", "teacher", "student", "student_kd"]


def subject_of(recs):
    return np.array([r.subject_id for r in recs])


def split_recordings(recs, n_folds, seed):
    """Fold id per recording, grouped by subject, balanced on 'subject has falls'."""
    subj = subject_of(recs)
    has_fall = pd.Series([r.is_fall for r in recs]).groupby(subj).transform("max").to_numpy()
    fold = np.full(len(recs), -1)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(recs)), has_fall.astype(int), subj)):
        fold[te] = k
    return fold


def tensors(w: Windows, dev):
    keep = w.y >= 0
    return (torch.as_tensor(w.x[keep], device=dev), torch.as_tensor(w.y[keep], device=dev))


def evaluate(name, w: Windows, prob, thr, out: dict, prefix="", target=0.95):
    keep = w.y >= 0
    m = window_metrics(w.y[keep], prob[keep], thr) if keep.any() else {}
    m.update(event_metrics(w.meta[keep].reset_index(drop=True), w.recs, prob[keep], thr))
    if (w.y == 1).any():  # threshold-free comparison: specificity when this set's own threshold
        oracle = threshold_for_sensitivity(w.y[keep], prob[keep], target)  # gives `target` sensitivity
        m["spec_at_target"] = window_metrics(w.y[keep], prob[keep], oracle)["specificity"]
    out[f"{prefix}{name}"] = m


@dataclass
class Fold:
    """Everything one fold needs, built only from that fold's training people."""

    k: int
    seed: int
    trigger_g: float
    w_fit: Windows
    w_val: Windows
    sets: dict  # "test" (+ "elderly") -> Windows
    norm: Normalize
    mean: np.ndarray
    std: np.ndarray
    xf: torch.Tensor
    yf: torch.Tensor
    xv: torch.Tensor
    yv: torch.Tensor
    aug: Augment
    target: float
    dev: torch.device

    def score_all(self, fn) -> dict:
        return {name: fn(w) for name, w in [("val", self.w_val), *self.sets.items()]}

    def torch_scores(self, model) -> dict:
        return self.score_all(lambda w: predict(model, self.norm, torch.as_tensor(w.x, device=self.dev)))

    def finish(self, name: str, scores: dict, res: dict) -> None:
        """Decision threshold from validation people, then metrics on every held-out set."""
        kv = self.w_val.y >= 0
        thr = threshold_for_sensitivity(self.w_val.y[kv], scores["val"][kv], self.target)
        res[f"{name}_decision"] = float(thr)
        for set_name, w in self.sets.items():
            evaluate(name, w, scores[set_name], thr, res, prefix=f"{set_name}/", target=self.target)


def prepare_fold(k, train_recs, test_recs, elderly_recs, cfg, dev, log) -> Fold:
    seed = cfg["seed"] + k
    # validation subjects (early stopping, decision thresholds) carved out of the training side
    subj = subject_of(train_recs)
    has_fall = pd.Series([r.is_fall for r in train_recs]).groupby(subj).transform("max").to_numpy()
    n_split = max(2, round(1 / cfg["val_fraction"]))
    sgkf = StratifiedGroupKFold(n_splits=n_split, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(sgkf.split(np.zeros(len(train_recs)), has_fall.astype(int), subj))
    fit_recs = [train_recs[i] for i in tr_idx]
    val_recs = [train_recs[i] for i in va_idx]

    trig = choose_threshold(fit_recs)
    w_fit, w_val, w_test = (extract_windows(r, trig) for r in (fit_recs, val_recs, test_recs))
    w_eld = extract_windows(elderly_recs, trig) if elderly_recs else None
    log(f"fold {k}: trigger {trig:.2f} g | windows fit {len(w_fit)} (falls {int((w_fit.y == 1).sum())}) "
        f"val {len(w_val)} test {len(w_test)}" + (f" elderly {len(w_eld)}" if w_eld else ""))

    mean, std = channel_stats(w_fit.x[w_fit.y >= 0])
    xf, yf = tensors(w_fit, dev)
    xv, yv = tensors(w_val, dev)
    return Fold(
        k=k, seed=seed, trigger_g=trig, w_fit=w_fit, w_val=w_val,
        sets={"test": w_test, **({"elderly": w_eld} if w_eld else {})},
        norm=Normalize(mean, std), mean=mean, std=std, xf=xf, yf=yf, xv=xv, yv=yv,
        aug=Augment(**{**cfg["augment"], "scale": tuple(cfg["augment"]["scale"])}),
        target=cfg["target_sensitivity"], dev=dev,
    )


def run_fold(k, train_recs, test_recs, elderly_recs, cfg, dev, log):
    fd = prepare_fold(k, train_recs, test_recs, elderly_recs, cfg, dev, log)
    seed, norm, aug = fd.seed, fd.norm, fd.aug
    xf, yf, xv, yv, w_fit = fd.xf, fd.yf, fd.xv, fd.yv, fd.w_fit
    res: dict = {"trigger_g": fd.trigger_g}

    def finish(name, scores):
        fd.finish(name, scores, res)

    score_all, torch_scores = fd.score_all, fd.torch_scores

    # 1. threshold rule on peak |a|
    finish("threshold", score_all(lambda w: threshold_scores(w.x)))
    # 2. random forest
    rf = train_forest(w_fit.x, w_fit.y, seed=seed)
    finish("forest", score_all(lambda w: rf.predict_proba(features(w.x))[:, 1]))

    tc, sc, dc = cfg["teacher"], cfg["student"], cfg["distill"]
    # 3. teacher
    teacher = SanadTeacher(widths=tuple(tc["widths"]), blocks=tc["blocks"])
    t0 = time.time()
    h = train(teacher, norm, xf, yf, xv, yv, TrainConfig(
        epochs=tc["epochs"], lr=tc["lr"], batch_size=tc["batch_size"], patience=tc["patience"],
        seed=seed), aug, log=lambda s: log(f"  teacher {s}"))
    res["teacher_train_s"] = time.time() - t0
    res["teacher_best_epoch"] = h["best_epoch"]
    finish("teacher", torch_scores(teacher))
    # 4. student alone, 5. student distilled from this fold's teacher
    for name, t in [("student", None), ("student_kd", teacher)]:
        student = SanadTiny(widths=tuple(sc["widths"]), kernels=tuple(sc["kernels"]), dense=sc["dense"])
        h = train(student, norm, xf, yf, xv, yv, TrainConfig(
            epochs=sc["epochs"], lr=sc["lr"], batch_size=sc["batch_size"], patience=sc["patience"],
            seed=seed, alpha=dc["alpha"], temperature=dc["temperature"],
            extra_per_batch=dc["extra_per_batch"]), aug, teacher=t, log=lambda s, n=name: log(f"  {n} {s}"))
        res[f"{name}_best_epoch"] = h["best_epoch"]
        finish(name, torch_scores(student))

    res["sizes"] = {
        "teacher": {"params": count_params(teacher), "macs": count_macs(teacher, (200, 7))},
        "student": {"params": count_params(student), "macs": count_macs(student, (200, 7))},
    }
    return res


def summarize(folds: list[dict]) -> pd.DataFrame:
    rows = []
    for set_name in ("test", "elderly"):
        for a in APPROACHES:
            key = f"{set_name}/{a}"
            ms = [f[key] for f in folds if key in f]
            if not ms:
                continue
            row = {"set": set_name, "approach": a}
            for metric in ("event_sensitivity", "sensitivity", "specificity", "f1", "auc",
                           "false_alarms_per_hour"):
                vals = np.array([m.get(metric, np.nan) for m in ms], dtype=float)
                row[metric] = f"{np.nanmean(vals):.3f} ± {np.nanstd(vals):.3f}"
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIGS / "train.yaml"))
    ap.add_argument("--datasets", nargs="*", help="override datasets.cv")
    ap.add_argument("--folds", type=int)
    ap.add_argument("--name", default=datetime.now().strftime("cv_%Y%m%d_%H%M"))
    ap.add_argument("--allow-cpu", action="store_true", help="train even if no GPU is visible")
    args = ap.parse_args(argv)
    dev = pick_device(args.allow_cpu)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.folds:
        cfg["folds"] = args.folds
    ds_ids = args.datasets or cfg["datasets"]["cv"]
    missing = [d for d in ds_ids if not (INTERIM / f"{d}.parquet").exists()]
    if missing:
        print(f"skipping datasets not prepared yet: {missing}")
        ds_ids = [d for d in ds_ids if d not in missing]

    out_dir = RUNS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"
    log_path.write_text("", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(s + "\n")

    log(f"datasets {ds_ids}, device {dev}")
    recs = [r for d in ds_ids for r in load_interim(d)]
    hold = cfg["datasets"]["holdout_elderly"]
    elderly = [r for r in recs if hold and r.age_group == "elderly"]
    recs = [r for r in recs if not (hold and r.age_group == "elderly")]
    fold = split_recordings(recs, cfg["folds"], cfg["seed"])
    log(f"{len(recs)} recordings from {len(set(subject_of(recs)))} subjects in CV; "
        f"{len(elderly)} elderly recordings held out")

    results = []
    for k in range(cfg["folds"]):
        train_recs = [r for r, f in zip(recs, fold, strict=True) if f != k]
        test_recs = [r for r, f in zip(recs, fold, strict=True) if f == k]
        assert not set(subject_of(train_recs)) & set(subject_of(test_recs))
        results.append(run_fold(k, train_recs, test_recs, elderly, cfg, dev, log))
        (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float))

    table = summarize(results)
    md = table.to_markdown(index=False) if hasattr(table, "to_markdown") else table.to_string()
    (out_dir / "results.md").write_text(
        f"# CV results ({', '.join(ds_ids)}, {cfg['folds']} person-grouped folds)\n\n"
        f"Decision thresholds tuned on validation subjects for {cfg['target_sensitivity']:.0%} sensitivity.\n"
        f"Sizes: {json.dumps(results[0]['sizes'])}\n\n{md}\n", encoding="utf-8")
    log("\n" + table.to_string(index=False))
    log(f"\nsaved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
