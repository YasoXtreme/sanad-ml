"""Teacher: a 1D ResNet (~0.9 M params) trained on the GPU, then distilled into Sanad-Tiny.

Same input as the student, (B, T=200, C=7), so its predictions can supervise the student
window-for-window on identical augmented batches.
"""

from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, stride: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm1d(c_out), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, k, padding=k // 2, bias=False), nn.BatchNorm1d(c_out),
        )
        self.skip = (
            nn.Identity() if stride == 1 and c_in == c_out
            else nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                               nn.BatchNorm1d(c_out))
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class SanadTeacher(nn.Module):
    def __init__(
        self,
        in_ch: int = 7,
        widths: tuple[int, ...] = (32, 64, 128, 160),
        blocks: int = 2,
        k: int = 5,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, widths[0], 7, padding=3, bias=False), nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages, c = [], widths[0]
        for i, w in enumerate(widths):
            for j in range(blocks):
                stages.append(ResBlock(c, w, k, stride=2 if (i > 0 and j == 0) else 1, dropout=dropout))
                c = w
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.Dropout(head_dropout), nn.Linear(2 * c, 128), nn.ReLU(inplace=True), nn.Linear(128, 1)
        )
        self.config = {"in_ch": in_ch, "widths": widths, "blocks": blocks, "k": k,
                       "dropout": dropout, "head_dropout": head_dropout}

    def forward(self, x):  # x: (B, T, C)
        h = self.stages(self.stem(x.transpose(1, 2)))
        h = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.head(h).squeeze(-1)
