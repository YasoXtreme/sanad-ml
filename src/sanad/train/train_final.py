"""Train the band's model (configs/deploy.yaml), calibrate its threshold, export for the Pico 2 W.

    uv run python -I -m sanad.train.train_final

1. Out-of-fold pass: the same 5 person-grouped folds as the tournaments; in each, the teacher and
   the student are trained without the fold's people, who are then scored. Pooling those scores
   gives every person an unbiased prediction (WEDA's elderly are scored by every fold model).
2. Decision threshold = the highest one that still catches `target_event_sensitivity` of all
   pooled falls end to end (falls the trigger missed count as misses). The pooled metrics at that
   threshold are the honest estimate of what the band will do.
3. Final model: teacher + student trained on everyone (elderly daily activities included), with a
   few validation people kept only for early stopping; exported with the calibrated threshold.

Writes models/: teacher.pt, student.pt, sanad_tiny_int8.tflite, firmware/*, final_report.json,
and runs/deploy/oof_windows.parquet (per-window out-of-fold scores, for analysis).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from sanad.data.augment import Augment
from sanad.data.prepare import load_interim
from sanad.data.windowing import choose_threshold, extract_windows
from sanad.eval.metrics import event_metrics, threshold_for_event_sensitivity, window_metrics
from sanad.models.common import Normalize, channel_stats, count_macs, count_params
from sanad.models.factory import build_teacher
from sanad.models.student import SanadTiny
from sanad.paths import CONFIGS, INTERIM, MODELS, RUNS
from sanad.train.loop import TrainConfig, pick_device, predict, train
from sanad.train.run_cv import prepare_fold, split_recordings, subject_of


def fit_pair(dcfg, cfg, norm, mean, std, xf, yf, xv, yv, aug, seed, log):
    """Teacher, then the student distilled from it."""
    tc, sc = dcfg["teacher"], cfg["student"]
    teacher = build_teacher(tc, mean, std)
    train(teacher, norm, xf, yf, xv, yv, TrainConfig(
        epochs=tc["epochs"], lr=tc["lr"], weight_decay=tc.get("weight_decay", 1e-4),
        batch_size=cfg["teacher"]["batch_size"], patience=tc["patience"], seed=seed), aug,
        log=lambda s: log(f"    teacher {s}"))
    student = SanadTiny(**{k: tuple(v) if isinstance(v, list) else v for k, v in dcfg["student"].items()})
    train(student, norm, xf, yf, xv, yv, TrainConfig(
        epochs=sc["epochs"], lr=sc["lr"], batch_size=sc["batch_size"], patience=sc["patience"],
        seed=seed, alpha=dcfg["distill"]["alpha"], temperature=dcfg["distill"]["temperature"]),
        aug, teacher=teacher, log=lambda s: log(f"    student {s}"))
    return teacher.eval(), student.eval()


def window_rows(w, prob, fold, set_name):
    df = w.meta[["dataset", "subject_id", "recording_id", "activity", "age_group", "peak_t",
                 "peak_g", "label"]].copy()
    df["prob"], df["fold"], df["set"] = prob, fold, set_name
    return df


def out_of_fold(recs, elderly, cfg, dcfg, dev, log):
    fold = split_recordings(recs, cfg["folds"], cfg["seed"])  # same folds as run_cv / tournaments
    rows, rec_tables, eld_tables = [], [], []
    for k in range(cfg["folds"]):
        tr = [r for r, f in zip(recs, fold, strict=True) if f != k]
        te = [r for r, f in zip(recs, fold, strict=True) if f == k]
        assert not set(subject_of(tr)) & set(subject_of(te))
        fd = prepare_fold(k, tr, te, elderly, cfg, dev, lambda s: log(f"  {s}"))
        _, student = fit_pair(dcfg, cfg, fd.norm, fd.mean, fd.std, fd.xf, fd.yf, fd.xv, fd.yv,
                              fd.aug, fd.seed, lambda s: None)
        for set_name, w in fd.sets.items():
            p = predict(student, fd.norm, torch.as_tensor(w.x, device=dev))
            rows.append(window_rows(w, p, k, set_name))
            (rec_tables if set_name == "test" else eld_tables).append(w.recs.assign(fold=k))
        log(f"  fold {k} scored")
    return pd.concat(rows, ignore_index=True), pd.concat(rec_tables, ignore_index=True), eld_tables


def calibrate(oof: pd.DataFrame, test_recs: pd.DataFrame, eld_recs: list, target: float) -> dict:
    t = oof[(oof.set == "test") & (oof.label >= 0)].reset_index(drop=True)
    thr = threshold_for_event_sensitivity(t, test_recs, t.prob.to_numpy(), target)
    m = window_metrics(t.label.to_numpy(), t.prob.to_numpy(), thr)
    m.update(event_metrics(t, test_recs, t.prob.to_numpy(), thr))
    per_ds = {}
    for ds, g in t.groupby("dataset"):
        r = test_recs[test_recs.dataset == ds]
        e = event_metrics(g.reset_index(drop=True), r, g.prob.to_numpy(), thr)
        adl_h = r.loc[~r.is_fall, "duration_s"].sum() / 3600
        per_ds[ds] = {"falls_caught": e["event_sensitivity"], "false_alarms_per_hour": e["false_alarms_per_hour"],
                      "triggers_per_hour_adl": int((g.label == 0).sum()) / adl_h if adl_h else float("nan"),
                      "adl_hours": adl_h}
    eld = []
    for k, r in enumerate(eld_recs):
        g = oof[(oof.set == "elderly") & (oof.fold == k)].reset_index(drop=True)
        eld.append(event_metrics(g, r, g.prob.to_numpy(), thr)["false_alarms_per_hour"])
    e0 = eld_recs[0]
    eld_h = e0["duration_s"].sum() / 3600
    eld_trig = len(oof[(oof.set == "elderly") & (oof.fold == 0)]) / eld_h
    return {"decision_prob": thr, "target_event_sensitivity": target, "pooled": m, "per_dataset": per_ds,
            "elderly": {"false_alarms_per_hour": float(np.mean(eld)), "triggers_per_hour": eld_trig,
                        "hours": eld_h}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIGS / "deploy.yaml"))
    ap.add_argument("--allow-cpu", action="store_true", help="train even if no GPU is visible")
    args = ap.parse_args(argv)
    dev = pick_device(args.allow_cpu)
    dcfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load((CONFIGS / dcfg["base_config"]).read_text(encoding="utf-8"))
    ds_ids = [d for d in cfg["datasets"]["cv"] if (INTERIM / f"{d}.parquet").exists()]
    all_recs = [r for d in ds_ids for r in load_interim(d)]
    elderly = [r for r in all_recs if r.age_group == "elderly"]
    recs = [r for r in all_recs if r.age_group != "elderly"]
    print(f"datasets {ds_ids}: {len(all_recs)} recordings ({len(elderly)} elderly), device {dev}")

    # 1-2. out-of-fold predictions -> calibrated threshold
    print("out-of-fold pass (5 folds):")
    oof, test_recs, eld_recs = out_of_fold(recs, elderly, cfg, dcfg, dev, print)
    out_dir = RUNS / "deploy"
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(out_dir / "oof_windows.parquet", index=False)
    test_recs.to_parquet(out_dir / "oof_recordings.parquet", index=False)
    cal = calibrate(oof, test_recs, eld_recs, dcfg["calibration"]["target_event_sensitivity"])
    p = cal["pooled"]
    print(f"calibrated threshold {cal['decision_prob']:.4f}: falls caught {p['event_sensitivity']:.1%}, "
          f"specificity {p['specificity']:.1%}, false alarms/h {p['false_alarms_per_hour']:.1f}, "
          f"elderly {cal['elderly']['false_alarms_per_hour']:.1f}/h")

    # 3. final model on everyone
    subj = subject_of(all_recs)
    has_fall = pd.Series([r.is_fall for r in all_recs]).groupby(subj).transform("max").to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=max(2, round(1 / cfg["val_fraction"])), shuffle=True,
                                random_state=cfg["seed"])
    tr_idx, va_idx = next(sgkf.split(np.zeros(len(all_recs)), has_fall.astype(int), subj))
    fit, val = [all_recs[i] for i in tr_idx], [all_recs[i] for i in va_idx]
    trig = choose_threshold(fit)
    w_fit, w_val = extract_windows(fit, trig), extract_windows(val, trig)
    mean, std = channel_stats(w_fit.x[w_fit.y >= 0])
    norm = Normalize(mean, std).to(dev)
    kf, kv = w_fit.y >= 0, w_val.y >= 0
    xf, yf = torch.as_tensor(w_fit.x[kf], device=dev), torch.as_tensor(w_fit.y[kf], device=dev)
    xv, yv = torch.as_tensor(w_val.x[kv], device=dev), torch.as_tensor(w_val.y[kv], device=dev)
    aug = Augment(**{**cfg["augment"], "scale": tuple(cfg["augment"]["scale"])})
    print(f"final model: trigger {trig:.3f} g, {len(w_fit)} training windows ({int((w_fit.y == 1).sum())} falls)")
    teacher, student = fit_pair(dcfg, cfg, norm, mean, std, xf, yf, xv, yv, aug, cfg["seed"], print)
    thr = cal["decision_prob"]
    yv_np = w_val.y[kv]
    p_student = predict(student, norm, xv)

    # ---- export ----
    from sanad.export.to_c_array import write_config_header, write_model_array
    from sanad.export.to_keras import max_abs_diff, to_keras
    from sanad.export.to_tflite import op_names, to_int8_tflite

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state": teacher.state_dict(), "config": teacher.config}, MODELS / "teacher.pt")
    torch.save({"state": student.state_dict(), "config": student.config, "mean": mean, "std": std},
               MODELS / "student.pt")
    with torch.no_grad():
        zf, zv = norm(xf).cpu().numpy(), norm(xv).cpu().numpy()
    kmodel = to_keras(student.cpu())
    parity = max_abs_diff(student, kmodel, zv[:256])
    assert parity < 1e-4, f"Keras copy differs from PyTorch by {parity}"
    rng = np.random.default_rng(cfg["seed"])
    yf_np = w_fit.y[kf]
    rep_idx = np.r_[rng.choice(np.flatnonzero(yf_np == 1), 150, replace=False),
                    rng.choice(np.flatnonzero(yf_np == 0), 350, replace=False)]
    q = to_int8_tflite(kmodel, zf[rep_idx])
    p_int8 = q.predict(zv)
    (MODELS / "sanad_tiny_int8.tflite").write_bytes(q.blob)
    fw = MODELS / "firmware"
    write_model_array(q.blob, fw)
    write_config_header(fw, q, mean, std, trigger_g=trig, decision_prob=thr)
    report = {
        "datasets": ds_ids, "trigger_g": trig, "decision_prob": thr, "deploy_config": dcfg,
        "out_of_fold": cal,
        "teacher": {"params": count_params(teacher), "kind": dcfg["teacher"]["kind"]},
        "student": {"params": count_params(student), "macs": count_macs(student, (200, 7)),
                    "config": student.config},
        "val_student_float": window_metrics(yv_np, p_student, thr),
        "val_student_int8": window_metrics(yv_np, p_int8, thr),
        "keras_parity_max_abs": parity, "tflite_bytes": len(q.blob), "tflite_ops": op_names(q.blob),
        "int8_vs_float_max_abs": float(np.max(np.abs(p_int8 - p_student))),
    }
    (MODELS / "final_report.json").write_text(json.dumps(report, indent=1, default=float))
    s, i8 = report["val_student_float"], report["val_student_int8"]
    print(f"\nSanad-Tiny: {report['student']['params']} params, {report['student']['macs']} MACs, "
          f"{len(q.blob) / 1024:.1f} KB int8; threshold {thr:.4f}")
    print(f"validation  float: sens {s['sensitivity']:.3f} spec {s['specificity']:.3f} AUC {s['auc']:.3f}")
    print(f"validation   int8: sens {i8['sensitivity']:.3f} spec {i8['specificity']:.3f} AUC {i8['auc']:.3f}")
    print(f"saved to {MODELS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
