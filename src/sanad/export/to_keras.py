"""Copy a trained PyTorch SanadTiny into an identical Keras model (BatchNorm folded into the convs).

Keras is only the bridge to the TFLite converter; training stays in PyTorch.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn

from sanad.models.student import SanadTiny

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _fold_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> tuple[np.ndarray, np.ndarray]:
    """Conv(no bias) + BN (eval) -> one conv with bias. Returns Keras-layout kernel (k, in, out)."""
    w = conv.weight.detach().double().cpu().numpy()  # (out, in, k)
    b = np.zeros(w.shape[0]) if conv.bias is None else conv.bias.detach().double().cpu().numpy()
    scale = (bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)).cpu().numpy()
    shift = bn.bias.detach().double().cpu().numpy() - bn.running_mean.double().cpu().numpy() * scale
    w = w * scale[:, None, None]
    b = b * scale + shift
    return w.transpose(2, 1, 0).astype(np.float32), b.astype(np.float32)


def to_keras(model: SanadTiny, window: int = 200):
    import tensorflow as tf

    keras = tf.keras
    model = model.eval().cpu()
    cfg = model.config
    convs = [m for m in model.features if isinstance(m, nn.Conv1d)]
    bns = [m for m in model.features if isinstance(m, nn.BatchNorm1d)]
    lins = [m for m in model.head if isinstance(m, nn.Linear)]

    inp = keras.Input((window, cfg["in_ch"]), name="window")
    h = inp
    for i, (conv, bn) in enumerate(zip(convs, bns)):
        k, b = _fold_bn(conv, bn)
        layer = keras.layers.Conv1D(k.shape[2], k.shape[0], padding="same", activation="relu",
                                    name=f"conv{i + 1}")
        h = layer(h)
        layer.set_weights([k, b])
        if i < len(convs) - 1:
            h = keras.layers.MaxPooling1D(2, name=f"pool{i + 1}")(h)
    h = keras.layers.GlobalAveragePooling1D(name="gap")(h)
    for i, lin in enumerate(lins):
        last = i == len(lins) - 1
        layer = keras.layers.Dense(lin.out_features, activation=None if last else "relu",
                                   name="logit" if last else f"dense{i + 1}")
        h = layer(h)
        layer.set_weights([lin.weight.detach().cpu().numpy().T, lin.bias.detach().cpu().numpy()])
    out = keras.layers.Activation("sigmoid", name="fall_prob")(h)
    return keras.Model(inp, out, name="sanad_tiny")


def max_abs_diff(model: SanadTiny, kmodel, x: np.ndarray) -> float:
    """Largest |p_torch - p_keras| over a batch of normalized (B, T, 7) windows."""
    with torch.no_grad():
        p_t = torch.sigmoid(model.eval().cpu()(torch.as_tensor(x, dtype=torch.float32))).numpy()
    p_k = kmodel(x, training=False).numpy().ravel()
    return float(np.max(np.abs(p_t - p_k)))
