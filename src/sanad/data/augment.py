"""Batch augmentation on the GPU, applied to raw windows (B, T, 6) in g and deg/s.

Teacher and student see the *same* augmented batch during distillation ("consistent views").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from sanad.data.harmonize import ACC_LIMIT_G, MIRROR_ACC, MIRROR_GYRO


def random_rotations(n: int, max_deg: float, device=None, generator=None) -> torch.Tensor:
    """(n, 3, 3) rotations about random axes by angles uniform in [-max_deg, max_deg] (Rodrigues)."""
    axis = torch.randn(n, 3, device=device, generator=generator)
    axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-8)
    ang = (torch.rand(n, device=device, generator=generator) * 2 - 1) * math.radians(max_deg)
    k = torch.zeros(n, 3, 3, device=device)
    k[:, 0, 1], k[:, 0, 2] = -axis[:, 2], axis[:, 1]
    k[:, 1, 0], k[:, 1, 2] = axis[:, 2], -axis[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -axis[:, 1], axis[:, 0]
    s, c = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
    eye = torch.eye(3, device=device).expand(n, 3, 3)
    return eye + s * k + (1 - c) * (k @ k)


def roll_rotations(n: int, max_deg: float, device=None, generator=None) -> torch.Tensor:
    """(n, 3, 3) rotations about the forearm (x) axis by angles uniform in [-max_deg, max_deg]."""
    ang = (torch.rand(n, device=device, generator=generator) * 2 - 1) * math.radians(max_deg)
    c, s = torch.cos(ang), torch.sin(ang)
    r = torch.zeros(n, 3, 3, device=device)
    r[:, 0, 0] = 1
    r[:, 1, 1], r[:, 1, 2] = c, -s
    r[:, 2, 1], r[:, 2, 2] = s, c
    return r


@dataclass
class Augment:
    rot_deg: float = 25.0  # band mounted slightly differently / strap rotated
    roll_deg: float = 180.0  # band slid around the wrist; FallAllD's roll is unknown per subject
    mirror_p: float = 0.5  # mirror-image motion (e.g. falling left instead of right)
    scale: tuple[float, float] = (0.9, 1.1)  # sensor gain differences
    shift: int = 10  # +-0.2 s jitter of the detected peak
    noise_acc_g: float = 0.01
    noise_gyro_dps: float = 1.0

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        dev = x.device
        acc, gyro = x[..., :3], x[..., 3:]

        rot = random_rotations(b, self.rot_deg, dev, generator)  # proper rotation: same for both
        if self.roll_deg:
            rot = rot @ roll_rotations(b, self.roll_deg, dev, generator)
        acc = acc @ rot.transpose(1, 2)
        gyro = gyro @ rot.transpose(1, 2)

        m = (torch.rand(b, 1, 1, device=dev, generator=generator) < self.mirror_p).float()
        ma = torch.as_tensor(MIRROR_ACC, dtype=x.dtype, device=dev)
        mg = torch.as_tensor(MIRROR_GYRO, dtype=x.dtype, device=dev)
        acc = acc * (1 - m + m * ma)
        gyro = gyro * (1 - m + m * mg)

        lo, hi = self.scale
        acc = acc * (lo + (hi - lo) * torch.rand(b, 1, 1, device=dev, generator=generator))
        gyro = gyro * (lo + (hi - lo) * torch.rand(b, 1, 1, device=dev, generator=generator))

        if self.shift:
            off = torch.randint(-self.shift, self.shift + 1, (b, 1), device=dev, generator=generator)
            idx = (torch.arange(t, device=dev)[None] - off).clamp(0, t - 1)
            acc = torch.gather(acc, 1, idx[..., None].expand(-1, -1, 3))
            gyro = torch.gather(gyro, 1, idx[..., None].expand(-1, -1, 3))

        acc = acc + self.noise_acc_g * torch.randn(acc.shape, device=dev, generator=generator)
        gyro = gyro + self.noise_gyro_dps * torch.randn(gyro.shape, device=dev, generator=generator)
        return torch.cat([acc.clamp(-ACC_LIMIT_G, ACC_LIMIT_G), gyro], dim=-1)
