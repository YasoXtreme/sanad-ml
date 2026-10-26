"""Teacher tournament + distillation tournament on the same person-grouped folds as run_cv.

    uv run python -I -m sanad.train.tournament --name tournament_v1

Per fold: every teacher in configs/tournament.yaml is trained and scored ("T:<name>"), plus an
ensemble of them; then Sanad-Tiny is trained with no teacher ("S:none") and distilled from every
teacher with every recipe ("S:<teacher>/<recipe>"). Contenders in each tournament play a
round robin: for each pair, whoever has the better primary metric on more folds wins the match.
Writes runs/<name>/results.json and tournament.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sanad.data.prepare import load_interim
from sanad.models.common import count_params
from sanad.models.oxford import OxfordTeacher
from sanad.models.student import SanadTiny
from sanad.models.teacher import SanadTeacher
from sanad.models.transformer import EnsembleTeacher, SanadTransformer
from sanad.paths import CONFIGS, INTERIM, RAW, RUNS
from sanad.train.loop import TrainConfig, pick_device, train
from sanad.train.run_cv import prepare_fold, split_recordings, subject_of

OXFORD_WEIGHTS = RAW / "oxford_ssl" / "mtl_5_best.mdl"
METRICS = [  # (key in results, set, label, higher is better)
    ("spec_at_target", "test", "Specificity @ 95% sens.", True),
    ("auc", "test", "AUC", True),
    ("event_sensitivity", "test", "Falls caught", True),
    ("specificity", "test", "Specificity", True),
    ("false_alarms_per_hour", "test", "False alarms/h", False),
    ("false_alarms_per_hour", "elderly", "Elderly false alarms/h", False),
]


def build_teacher(spec: dict, fd):
    kind = spec["kind"]
    if kind == "resnet":
        return SanadTeacher()
    if kind == "transformer":
        return SanadTransformer(d_model=spec["d_model"], heads=spec["heads"], layers=spec["layers"],
                                ff=spec["ff"])
    if kind == "oxford":
        return OxfordTeacher(fd.mean, fd.std, pretrained=OXFORD_WEIGHTS if spec["pretrained"] else None)
    raise ValueError(f"unknown teacher kind {kind!r}")


def run_fold(k, train_recs, test_recs, elderly, cfg, tcfg, dev, log) -> dict:
    fd = prepare_fold(k, train_recs, test_recs, elderly, cfg, dev, log)
    res: dict = {"trigger_g": fd.trigger_g, "params": {}, "train_s": {}, "best_epoch": {}}

    teachers = {}
    for name, spec in tcfg["teachers"].items():
        model = build_teacher(spec, fd)
        t0 = time.time()
        h = train(model, fd.norm, fd.xf, fd.yf, fd.xv, fd.yv, TrainConfig(
            epochs=spec["epochs"], lr=spec["lr"], weight_decay=spec.get("weight_decay", 1e-4),
            batch_size=cfg["teacher"]["batch_size"], patience=spec["patience"], seed=fd.seed),
            fd.aug, log=lambda s, n=name: log(f"  T:{n} {s}"))
        res["train_s"][f"T:{name}"] = time.time() - t0
        res["best_epoch"][f"T:{name}"] = h["best_epoch"]
        res["params"][f"T:{name}"] = count_params(model)
        fd.finish(f"T:{name}", fd.torch_scores(model), res)
        teachers[name] = model.eval()
        log(f"  T:{name}: val AUC {h['best_val_auc']:.4f}, "
            f"test spec@95 {res[f'test/T:{name}']['spec_at_target']:.3f}, {time.time() - t0:.0f} s")
    teachers["ensemble"] = EnsembleTeacher([teachers[n] for n in tcfg["ensemble"]]).eval()
    fd.finish("T:ensemble", fd.torch_scores(teachers["ensemble"]), res)
    res["params"]["T:ensemble"] = count_params(teachers["ensemble"])
    log(f"  T:ensemble: test spec@95 {res['test/T:ensemble']['spec_at_target']:.3f}")

    sc = cfg["student"]
    # one student size (cfg["student"]) unless the tournament config lists several: "<name>@<size>"
    sizes = tcfg.get("students") or {"": {k: sc[k] for k in ("widths", "kernels", "dense")}}
    runs = []
    for size, ss in sizes.items():
        sfx = f"@{size}" if size else ""
        runs.append((f"none{sfx}", None, None, ss))
        runs += [(f"{t}/{r}{sfx}", teachers[t], rc, ss)
                 for t in teachers for r, rc in tcfg["recipes"].items()]
    for name, teacher, rc, ss in runs:
        student = SanadTiny(widths=tuple(ss["widths"]), kernels=tuple(ss["kernels"]), dense=ss["dense"])
        res["params"][f"S:{name}"] = count_params(student)
        t0 = time.time()
        h = train(student, fd.norm, fd.xf, fd.yf, fd.xv, fd.yv, TrainConfig(
            epochs=sc["epochs"], lr=sc["lr"], batch_size=sc["batch_size"], patience=sc["patience"],
            seed=fd.seed, alpha=(rc or {}).get("alpha", 1.0), temperature=(rc or {}).get("temperature", 1.0)),
            fd.aug, teacher=teacher, log=lambda s: None)
        res["train_s"][f"S:{name}"] = time.time() - t0
        res["best_epoch"][f"S:{name}"] = h["best_epoch"]
        fd.finish(f"S:{name}", fd.torch_scores(student), res)
        log(f"  S:{name}: test spec@95 {res[f'test/S:{name}']['spec_at_target']:.3f}, "
            f"falls caught {res[f'test/S:{name}']['event_sensitivity']:.3f}, {time.time() - t0:.0f} s")
    return res


def contenders(folds: list[dict], prefix: str) -> list[str]:
    return sorted({k.split("/", 1)[1] for k in folds[0] if k.startswith(f"test/{prefix}")})


def per_fold(folds, name, metric, set_name) -> np.ndarray:
    return np.array([f.get(f"{set_name}/{name}", {}).get(metric, np.nan) for f in folds], dtype=float)


def standings(folds: list[dict], prefix: str, primary: str) -> pd.DataFrame:
    """Round robin: each pair plays one match; a match is won by being better on more folds."""
    names = contenders(folds, prefix)
    pts = dict.fromkeys(names, 0.0)
    wins = dict.fromkeys(names, 0)
    for a, b in itertools.combinations(names, 2):
        va, vb = per_fold(folds, a, primary, "test"), per_fold(folds, b, primary, "test")
        fa, fb = int(np.sum(va > vb)), int(np.sum(vb > va))
        if fa > fb:
            pts[a] += 1
            wins[a] += 1
        elif fb > fa:
            pts[b] += 1
            wins[b] += 1
        else:
            pts[a] += 0.5
            pts[b] += 0.5
    rows = []
    for n in names:
        row = {"contender": n, "points": pts[n], "match_wins": wins[n]}
        for metric, set_name, label, _ in METRICS:
            v = per_fold(folds, n, metric, set_name)
            row[label] = np.nanmean(v)
            row[f"{label} sd"] = np.nanstd(v)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["points", METRICS[0][2]], ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def to_markdown(df: pd.DataFrame, params: dict) -> str:
    cols = ["Rank", "Contender", "Points", "Params"] + [m[2] for m in METRICS]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        p = params.get(r.contender, 11825)
        cells = [str(r["rank"]), r.contender.split(":", 1)[1], f"{r.points:g}",
                 f"{p / 1e6:.2f} M" if p >= 1e6 else f"{p / 1e3:.1f} k"]
        for _, _, label, _ in METRICS:
            v, s = r[label], r[f"{label} sd"]
            pct = label not in ("AUC",) and "alarms" not in label
            cells.append("–" if np.isnan(v) else (f"{v:.1%} ± {s:.1%}" if pct else f"{v:.2f} ± {s:.2f}"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, folds: list[dict], tcfg: dict) -> None:
    primary = tcfg["primary_metric"]
    params = folds[0]["params"]
    t, s = standings(folds, "T:", primary), standings(folds, "S:", primary)
    t.to_csv(out_dir / "teachers.csv", index=False)
    s.to_csv(out_dir / "students.csv", index=False)
    md = f"""# Teacher and distillation tournament (`{out_dir.name}`, {len(folds)} folds)

Same people, folds and windows as `cv_v1`. Ranking = round robin on **specificity at 95%
sensitivity** on test people: in each match the contender that is better on more folds wins
(1 point, draw 0.5). Columns show mean ± SD over folds.

## Teacher tournament

{to_markdown(t, params)}

## Distillation tournament (all students are Sanad-Tiny, 11.8 k parameters)

`none` = no teacher. `<teacher>/<recipe>`: `standard` = half labels, half teacher (T = 2);
`teacher_only` = only the teacher's soft answers; `hot` = half/half with softer answers (T = 4).

{to_markdown(s, params)}

Oxford harnet5 weights: Copyright © 2022, University of Oxford (Academic Use Licence; Yuan et al. 2024).
"""
    (out_dir / "tournament.md").write_text(md, encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIGS / "tournament.yaml"))
    ap.add_argument("--name", default=datetime.now().strftime("tournament_%Y%m%d_%H%M"))
    ap.add_argument("--folds", type=int, help="run only the first N folds (smoke test)")
    ap.add_argument("--epochs-scale", type=float, default=1.0, help="multiply all epochs (smoke test)")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="rebuild tournament.md from results.json")
    args = ap.parse_args(argv)
    tcfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load((CONFIGS / tcfg["base_config"]).read_text(encoding="utf-8"))
    out_dir = RUNS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        write_report(out_dir, json.loads((out_dir / "results.json").read_text()), tcfg)
        return 0
    dev = pick_device(args.allow_cpu)
    if args.epochs_scale != 1.0:
        for spec in [*tcfg["teachers"].values(), cfg["student"]]:
            spec["epochs"] = max(2, round(spec["epochs"] * args.epochs_scale))
    if any(s["kind"] == "oxford" and s["pretrained"] for s in tcfg["teachers"].values()):
        assert OXFORD_WEIGHTS.exists(), "run: python -I -m sanad.data.download oxford_ssl"

    log_path = out_dir / "log.txt"
    log_path.write_text("", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(s + "\n")

    ds_ids = [d for d in cfg["datasets"]["cv"] if (INTERIM / f"{d}.parquet").exists()]
    recs = [r for d in ds_ids for r in load_interim(d)]
    hold = cfg["datasets"]["holdout_elderly"]
    elderly = [r for r in recs if hold and r.age_group == "elderly"]
    recs = [r for r in recs if not (hold and r.age_group == "elderly")]
    fold = split_recordings(recs, cfg["folds"], cfg["seed"])  # identical to run_cv
    log(f"datasets {ds_ids}, device {dev}, {len(recs)} recordings, {len(elderly)} elderly held out")

    results = []
    for k in range(args.folds or cfg["folds"]):
        tr = [r for r, f in zip(recs, fold, strict=True) if f != k]
        te = [r for r, f in zip(recs, fold, strict=True) if f == k]
        assert not set(subject_of(tr)) & set(subject_of(te))
        results.append(run_fold(k, tr, te, elderly, cfg, tcfg, dev, log))
        (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float))
        write_report(out_dir, results, tcfg)
    log(f"saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
