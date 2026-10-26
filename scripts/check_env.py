"""Check that this machine is ready for Sanad ML work.

Run:  uv run python scripts/check_env.py
Every line must say PASS before fetching data or training. Exit code is 1 if anything fails.
"""

from __future__ import annotations

import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's C++ info logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # keep TF float results deterministic

ROOT = Path(__file__).resolve().parents[1]
DVC_STORE = Path(r"D:\dvc-store\sanad-ml")
WINDOW, CHANNELS = 200, 7  # Sanad model input: 4 s at 50 Hz x (ax, ay, az, gx, gy, gz, |a|)

results: list[tuple[str, bool, str]] = []


def check(name: str):
    def wrap(fn):
        try:
            detail = fn() or ""
            results.append((name, True, detail))
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            msg = f"{type(exc).__name__}: {exc}"
            if os.environ.get("SANAD_DEBUG"):
                msg += "\n" + traceback.format_exc()
            results.append((name, False, msg))
        return fn

    return wrap


@check("Python 3.12")
def _python():
    v = sys.version_info
    assert (v.major, v.minor) == (3, 12), f"running {sys.version.split()[0]}"
    return f"{sys.version.split()[0]} at {sys.executable}"


@check("Core data libraries")
def _core():
    import matplotlib
    import numpy
    import pandas
    import pyarrow
    import scipy
    import sklearn

    return ", ".join(
        f"{m.__name__} {m.__version__}" for m in (numpy, scipy, pandas, pyarrow, sklearn, matplotlib)
    )


@check("PyTorch + CUDA build")
def _torch():
    import torch

    assert torch.version.cuda, "PyTorch is a CPU-only build"
    return f"torch {torch.__version__}, built for CUDA {torch.version.cuda}"


@check("GPU visible to PyTorch")
def _gpu():
    import torch

    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False. Is the RTX 5060 enabled (Lenovo GPU mode / restart)?"
    )
    p = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    sm = f"sm_{cap[0]}{cap[1]}"
    assert sm in arch_list, f"this PyTorch build has no kernels for {sm} (has {arch_list})"
    return f"{p.name}, {p.total_memory / 2**30:.1f} GB, capability {cap}"


@check("GPU training step (Conv1d + AMP)")
def _gpu_train():
    import torch
    from torch import nn

    assert torch.cuda.is_available(), "no GPU"
    dev = "cuda"
    model = nn.Sequential(
        nn.Conv1d(CHANNELS, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
        nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 1),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(512, CHANNELS, WINDOW, device=dev)
    y = (torch.rand(512, 1, device=dev) > 0.5).float()
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    assert all(math.isfinite(v) for v in losses), "loss became NaN/inf"
    assert losses[-1] < losses[0], "loss did not go down on a fixed batch"
    return f"50 steps of batch 512 in {dt:.2f} s, loss {losses[0]:.3f} -> {losses[-1]:.3f}"


@check("TensorFlow import")
def _tf():
    import tensorflow as tf

    return f"tensorflow {tf.__version__} (CPU is fine; only used for TFLite conversion)"


@check("Full-int8 TFLite export + run")
def _tflite():
    import numpy as np
    import tensorflow as tf

    keras = tf.keras
    rng = np.random.default_rng(0)
    inp = keras.Input((WINDOW, CHANNELS))
    h = keras.layers.Conv1D(16, 7, padding="same", activation="relu")(inp)
    h = keras.layers.MaxPooling1D(2)(h)
    h = keras.layers.Conv1D(32, 5, padding="same", activation="relu")(h)
    h = keras.layers.GlobalAveragePooling1D()(h)
    out = keras.layers.Dense(1, activation="sigmoid")(h)
    model = keras.Model(inp, out)

    calib = rng.standard_normal((64, WINDOW, CHANNELS)).astype(np.float32)

    def representative():
        for i in range(len(calib)):
            yield [calib[i : i + 1]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()

    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter
    interp = Interpreter(model_content=blob)
    interp.allocate_tensors()
    i_det, o_det = interp.get_input_details()[0], interp.get_output_details()[0]
    assert i_det["dtype"] == np.int8 and o_det["dtype"] == np.int8, "model is not full-int8"

    x = calib[:8]
    ref = model(x, training=False).numpy().ravel()
    i_scale, i_zero = i_det["quantization"]
    o_scale, o_zero = o_det["quantization"]
    got = []
    for row in x:
        q = np.clip(np.round(row / i_scale + i_zero), -128, 127).astype(np.int8)[None]
        interp.set_tensor(i_det["index"], q)
        interp.invoke()
        got.append((interp.get_tensor(o_det["index"]).astype(np.float32) - o_zero) * o_scale)
    err = float(np.max(np.abs(np.ravel(got) - ref)))
    assert err < 0.05, f"int8 output differs from float by {err:.3f}"
    return f"{len(blob) / 1024:.1f} KB int8 model, max |int8 - float| = {err:.4f}"


@check("emlearn (RF baseline -> C)")
def _emlearn():
    import emlearn

    return f"emlearn {getattr(emlearn, '__version__', 'installed')}"


@check("DVC backup store round-trip")
def _dvc():
    dvc = shutil.which("dvc", path=str(Path(sys.executable).parent)) or shutil.which("dvc")
    assert dvc, "dvc not found in the environment"
    assert DVC_STORE.exists(), f"{DVC_STORE} does not exist (run the DVC setup step)"
    out = subprocess.run([dvc, "remote", "list"], cwd=ROOT, capture_output=True, text=True,
                         check=False)
    assert str(DVC_STORE).lower() in out.stdout.lower().replace("/", "\\"), (
        f"repo's DVC remote is not {DVC_STORE}: {out.stdout.strip() or out.stderr.strip()}"
    )
    payload = "sanad dvc round-trip check\n"
    env = {**os.environ, "DVC_NO_ANALYTICS": "1"}  # no background analytics process locking tmp
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        run = lambda *a: subprocess.run(
            [dvc, *a], cwd=tmp, env=env, capture_output=True, text=True, check=True
        )
        run("init", "--no-scm", "-q")
        run("remote", "add", "-d", "store", str(DVC_STORE))
        (Path(tmp) / "probe.txt").write_text(payload)
        run("add", "probe.txt", "-q")
        run("push", "-q")
        (Path(tmp) / "probe.txt").unlink()

        def _force(func, path, _exc):  # DVC makes cache files read-only
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(Path(tmp) / ".dvc" / "cache", onexc=_force)
        run("pull", "-q")
        assert (Path(tmp) / "probe.txt").read_text() == payload, "pulled file differs"
    return f"push + pull through {DVC_STORE} OK"


@check("Free disk space on D:")
def _disk():
    free = shutil.disk_usage("D:\\").free / 2**30
    assert free > 50, f"only {free:.0f} GB free"
    return f"{free:.0f} GB free"


def main() -> int:
    width = max(len(n) for n, _, _ in results)
    print()
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = [n for n, ok, _ in results if not ok]
    print()
    print("All checks passed." if not failed else f"{len(failed)} check(s) failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
