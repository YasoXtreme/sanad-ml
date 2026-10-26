"""Train the final teacher + distilled Sanad-Tiny on all data, then export for the Pico 2 W.

    uv run python -I -m sanad.train.train_final

Uses every dataset in configs/train.yaml (elderly daily activities included: more realistic
negatives). A stratified set of subjects is still held out for early stopping and for picking the
decision threshold. Writes to models/:
  teacher.pt, student.pt               PyTorch weights + config
  sanad_tiny_int8.tflite               full-int8 model for TFLite Micro
  firmware/sanad_model_data.{cc,h}     the model as a C array
  firmware/sanad_model_config.h        trigger + decision thresholds, normalization, quantization
  final_report.json                    sizes, thresholds, float vs int8 validation metrics
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
from sanad.eval.metrics import threshold_for_sensitivity, window_metrics
from sanad.models.common import Normalize, channel_stats, count_macs, count_params
from sanad.models.student import SanadTiny
from sanad.models.teacher import SanadTeacher
from sanad.paths import CONFIGS, INTERIM, MODELS
from sanad.train.loop import TrainConfig, pick_device, predict, train


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIGS / "train.yaml"))
    ap.add_argument("--allow-cpu", action="store_true", help="train even if no GPU is visible")
    args = ap.parse_args(argv)
    dev = pick_device(args.allow_cpu)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ds_ids = [d for d in cfg["datasets"]["cv"] if (INTERIM / f"{d}.parquet").exists()]
    recs = [r for d in ds_ids for r in load_interim(d)]
    print(f"datasets {ds_ids}: {len(recs)} recordings, device {dev}")

    subj = np.array([r.subject_id for r in recs])
    has_fall = pd.Series([r.is_fall for r in recs]).groupby(subj).transform("max").to_numpy()
    n_split = max(2, round(1 / cfg["val_fraction"]))
    sgkf = StratifiedGroupKFold(n_splits=n_split, shuffle=True, random_state=cfg["seed"])
    tr_idx, va_idx = next(sgkf.split(np.zeros(len(recs)), has_fall.astype(int), subj))
    fit = [recs[i] for i in tr_idx]
    val = [recs[i] for i in va_idx]

    trig = choose_threshold(fit)
    w_fit, w_val = extract_windows(fit, trig), extract_windows(val, trig)
    print(f"trigger {trig:.3f} g | fit windows {len(w_fit)} ({int((w_fit.y == 1).sum())} falls), "
          f"val windows {len(w_val)} ({int((w_val.y == 1).sum())} falls)")
    mean, std = channel_stats(w_fit.x[w_fit.y >= 0])
    norm = Normalize(mean, std)
    kf, kv = w_fit.y >= 0, w_val.y >= 0
    xf = torch.as_tensor(w_fit.x[kf], device=dev)
    yf = torch.as_tensor(w_fit.y[kf], device=dev)
    xv = torch.as_tensor(w_val.x[kv], device=dev)
    yv = torch.as_tensor(w_val.y[kv], device=dev)
    aug = Augment(**{**cfg["augment"], "scale": tuple(cfg["augment"]["scale"])})
    tc, sc, dc = cfg["teacher"], cfg["student"], cfg["distill"]

    teacher = SanadTeacher(widths=tuple(tc["widths"]), blocks=tc["blocks"])
    train(teacher, norm, xf, yf, xv, yv, TrainConfig(
        epochs=tc["epochs"], lr=tc["lr"], batch_size=tc["batch_size"], patience=tc["patience"],
        seed=cfg["seed"]), aug, log=lambda s: print(f"  teacher {s}"))
    student = SanadTiny(widths=tuple(sc["widths"]), kernels=tuple(sc["kernels"]), dense=sc["dense"])
    train(student, norm, xf, yf, xv, yv, TrainConfig(
        epochs=sc["epochs"], lr=sc["lr"], batch_size=sc["batch_size"], patience=sc["patience"],
        seed=cfg["seed"], alpha=dc["alpha"], temperature=dc["temperature"],
        extra_per_batch=dc["extra_per_batch"]), aug, teacher=teacher,
        log=lambda s: print(f"  student_kd {s}"))

    yv_np = w_val.y[kv]
    p_teacher = predict(teacher, norm, xv)
    p_student = predict(student, norm, xv)
    thr = threshold_for_sensitivity(yv_np, p_student, cfg["target_sensitivity"])
    report = {
        "datasets": ds_ids, "trigger_g": trig, "decision_prob": thr,
        "val_teacher": window_metrics(yv_np, p_teacher,
                                      threshold_for_sensitivity(yv_np, p_teacher, cfg["target_sensitivity"])),
        "val_student_float": window_metrics(yv_np, p_student, thr),
        "teacher": {"params": count_params(teacher), "macs": count_macs(teacher, (200, 7))},
        "student": {"params": count_params(student), "macs": count_macs(student, (200, 7))},
    }

    # ---- export ----
    from sanad.export.to_c_array import write_config_header, write_model_array
    from sanad.export.to_keras import max_abs_diff, to_keras
    from sanad.export.to_tflite import op_names, to_int8_tflite

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state": teacher.state_dict(), "config": teacher.config}, MODELS / "teacher.pt")
    torch.save({"state": student.state_dict(), "config": student.config, "mean": mean, "std": std},
               MODELS / "student.pt")
    with torch.no_grad():
        zf = norm(xf).cpu().numpy()
        zv = norm(xv).cpu().numpy()
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
    report.update({
        "keras_parity_max_abs": parity,
        "tflite_bytes": len(q.blob),
        "tflite_ops": op_names(q.blob),
        "val_student_int8": window_metrics(yv_np, p_int8, thr),
        "int8_vs_float_max_abs": float(np.max(np.abs(p_int8 - p_student))),
    })
    (MODELS / "final_report.json").write_text(json.dumps(report, indent=1, default=float))
    s, i8 = report["val_student_float"], report["val_student_int8"]
    print(f"\nSanad-Tiny: {report['student']['params']} params, {report['student']['macs']} MACs, "
          f"{len(q.blob) / 1024:.1f} KB int8")
    print(f"validation  float: sens {s['sensitivity']:.3f} spec {s['specificity']:.3f} AUC {s['auc']:.3f}")
    print(f"validation   int8: sens {i8['sensitivity']:.3f} spec {i8['specificity']:.3f} AUC {i8['auc']:.3f}")
    print(f"saved to {MODELS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
