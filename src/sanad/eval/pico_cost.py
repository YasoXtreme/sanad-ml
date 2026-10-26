"""What a Sanad-Tiny size costs on the Pico 2 W (RP2350: Cortex-M33 @ 150 MHz, 520 KB SRAM, 4 MB flash).

Converts an untrained student of the given shape to full-int8 TFLite (the file size and operator
list don't depend on the weights) and estimates RAM for activations and the inference time.
The time is only a range from typical CMSIS-NN int8 throughput on Cortex-M33 (0.25-0.6 MAC per
cycle): it must be measured on the board.
"""

from __future__ import annotations

import numpy as np
import torch

from sanad.models.common import count_macs, count_params
from sanad.models.student import SanadTiny

CLOCK_HZ = 150e6
MAC_PER_CYCLE = (0.25, 0.6)  # slow and fast end for CMSIS-NN int8 conv on Cortex-M33
WINDOW, CHANNELS = 200, 7


def activation_bytes(widths: tuple[int, ...]) -> int:
    """Largest input+output pair of int8 activations across the conv stack (TFLM reuses the rest)."""
    sizes, t, c = [], WINDOW, CHANNELS
    for i, w in enumerate(widths):
        sizes.append(t * c + t * w)  # conv in + out
        if i < len(widths) - 1:
            sizes.append(t * w + (t // 2) * w)  # pool in + out
            t //= 2
        c = w
    return max(sizes)


def pico_cost(widths, kernels, dense) -> dict:
    from sanad.export.to_keras import to_keras
    from sanad.export.to_tflite import op_names, to_int8_tflite

    torch.manual_seed(0)
    m = SanadTiny(widths=tuple(widths), kernels=tuple(kernels), dense=dense).eval()
    rep = np.random.default_rng(0).normal(size=(64, WINDOW, CHANNELS)).astype(np.float32)
    q = to_int8_tflite(to_keras(m), rep)
    macs = count_macs(m, (WINDOW, CHANNELS))
    act = activation_bytes(tuple(widths))
    return {
        "params": count_params(m),
        "macs": macs,
        "tflite_kb": len(q.blob) / 1024,
        "ops": op_names(q.blob),
        "activation_kb": act / 1024,
        "arena_kb_est": (act + 4096) / 1024,  # + scratch/bookkeeping, rough
        "ms_est": tuple(macs / (r * CLOCK_HZ) * 1000 for r in MAC_PER_CYCLE[::-1]),
    }
