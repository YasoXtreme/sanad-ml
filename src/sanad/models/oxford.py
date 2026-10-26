"""Oxford SSL-wearables "harnet5" as a Sanad teacher (accelerometer only).

Copyright © 2022, University of Oxford. The network below re-implements the `Resnet` (version 1,
epoch_len 5) from https://github.com/OxWearables/ssl-wearables (sslearning/models/accNet.py) so its
pretrained weights load by name. That software and its weights are under the Oxford Academic Use
Licence: internal academic, non-commercial use for education only; not part of any product for
sale; keep this notice and credit "Copyright © 2022, University of Oxford" wherever results built
with it are shown. Yuan et al., "Self-supervised learning for human activity recognition using
700,000 person-days of wearable data", npj Digital Medicine (2024).

The pretrained model expects 3 x 150 (5 s at 30 Hz, acceleration in g). Its five downsampling
stages (2, 2, 3, 3, 3) also reduce our 4 s window at 30 Hz (120 samples) to exactly 1 step, so the
adapter only resamples 50 -> 30 Hz. Weights are read with torch.load(weights_only=True): tensors only,
no code from the download runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HARNET5 = [  # (out_channels, conv_kernel, n_resblocks, resblock_kernel, down_factor, down_order)
    (64, 5, 2, 5, 2, 2),
    (128, 5, 2, 5, 2, 2),
    (256, 5, 2, 5, 3, 1),
    (256, 5, 2, 5, 3, 1),
    (512, 5, 0, 5, 3, 1),
]


class Downsample(nn.Module):
    """Anti-aliased downsampling (box filter convolved `order` times), as in the original."""

    def __init__(self, channels: int, factor: int, order: int):
        super().__init__()
        self.stride, self.padding = factor, order * (factor - 1) // 2
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, np.ones(factor))
        kernel = torch.tensor(kernel / kernel.sum(), dtype=torch.float32)
        self.register_buffer("kernel", kernel[None, None, :].repeat(channels, 1, 1))

    def forward(self, x):
        return F.conv1d(x, self.kernel, stride=self.stride, padding=self.padding, groups=x.shape[1])


class ResBlock(nn.Module):
    """bn-relu-conv-bn-relu-conv with identity skip (pre-activation)."""

    def __init__(self, channels: int, k: int):
        super().__init__()
        self.bn1, self.bn2 = nn.BatchNorm1d(channels), nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, k, 1, k // 2, bias=False, padding_mode="circular")
        self.conv2 = nn.Conv1d(channels, channels, k, 1, k // 2, bias=False, padding_mode="circular")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return out + x


def harnet5_features() -> nn.Sequential:
    fe, c_in = nn.Sequential(), 3
    for i, (c, k, n_res, rk, down, order) in enumerate(HARNET5):
        mods = [nn.Conv1d(c_in, c, k, 1, k // 2, bias=False, padding_mode="circular")]
        mods += [ResBlock(c, rk) for _ in range(n_res)]
        mods += [nn.BatchNorm1d(c), nn.ReLU(True), Downsample(c, down, order)]
        fe.add_module(f"layer{i + 1}", nn.Sequential(*mods))
        c_in = c
    for m in fe.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    return fe


def load_pretrained(features: nn.Sequential, path: Path) -> int:
    """Load the feature-extractor weights from an Oxford checkpoint. Returns the tensor count."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("module.").removeprefix("feature_extractor."): v for k, v in sd.items()
          if ".feature_extractor." in f".{k}" or k.startswith("feature_extractor.")}
    own = features.state_dict()
    missing = set(own) - set(sd)
    shape_bad = [k for k in own if k in sd and sd[k].shape != own[k].shape]
    if missing or shape_bad:
        raise ValueError(f"checkpoint doesn't match harnet5: missing {sorted(missing)[:5]}, "
                         f"shape mismatch {shape_bad[:5]}")
    features.load_state_dict({k: sd[k] for k in own})
    return len(own)


class OxfordTeacher(nn.Module):
    """Takes the same normalized (B, 200, 7) input as the other teachers.

    Undoes the normalization, keeps acceleration (g), resamples 50 -> 30 Hz (200 -> 120 samples),
    runs harnet5's feature extractor and a fresh two-layer head (like the original EvaClassifier).
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, pretrained: Path | None = None,
                 head_dropout: float = 0.3):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean[:3], dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std[:3], dtype=torch.float32))
        self.feature_extractor = harnet5_features()
        self.pretrained = pretrained is not None
        if pretrained is not None:
            load_pretrained(self.feature_extractor, pretrained)
        self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(head_dropout),
                                  nn.Linear(512, 1))
        self.config = {"pretrained": self.pretrained, "head_dropout": head_dropout}

    def forward(self, x):  # x: normalized (B, T, 7)
        acc = x[..., :3] * self.std + self.mean  # back to g
        acc = F.interpolate(acc.transpose(1, 2), size=round(x.shape[1] * 30 / 50), mode="linear",
                            align_corners=False)
        h = self.feature_extractor(acc)
        return self.head(h.flatten(1)).squeeze(-1)
