"""Model input preparation shared by teacher, student, export and (in C) the firmware."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz", "amag")


def add_magnitude(x6: torch.Tensor) -> torch.Tensor:
    """(B, T, 6) raw window -> (B, T, 7) with |a| appended (acc already clipped to +-8 g)."""
    return torch.cat([x6, x6[..., :3].norm(dim=-1, keepdim=True)], dim=-1)


def channel_stats(x6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std of the 7-channel input, from training windows only."""
    x7 = add_magnitude(torch.as_tensor(x6, dtype=torch.float32)).reshape(-1, len(CHANNELS))
    mean, std = x7.mean(0), x7.std(0).clamp_min(1e-6)
    return mean.numpy(), std.numpy()


class Normalize(nn.Module):
    """Raw (B, T, 6) -> normalized (B, T, 7). Not part of the exported model: the band does this in C."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x6: torch.Tensor) -> torch.Tensor:
        return (add_magnitude(x6) - self.mean) / self.std


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_macs(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Multiply-accumulates of Conv1d and Linear layers for one input (batch of 1)."""
    total = 0

    def conv_hook(m: nn.Conv1d, _inp, out):
        nonlocal total
        total += out.shape[-1] * m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0]

    def lin_hook(m: nn.Linear, _inp, _out):
        nonlocal total
        total += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    was_training = model.training
    model.eval()
    p = next(model.parameters())
    model(torch.zeros(1, *input_shape, device=p.device, dtype=p.dtype))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return total
