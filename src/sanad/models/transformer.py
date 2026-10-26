"""Transformer teachers trained from scratch on Sanad's windows.

Patch tokens: the 4 s window (200 samples x 7 channels) is cut into 25 patches of 8 samples
(160 ms); each becomes one token. A [CLS] token plus learned positions go through a pre-norm
Transformer encoder; the [CLS] output is the fall logit.
"""

from __future__ import annotations

import torch
from torch import nn


class SanadTransformer(nn.Module):
    def __init__(
        self,
        in_ch: int = 7,
        window: int = 200,
        patch: int = 8,
        d_model: int = 128,
        heads: int = 4,
        layers: int = 4,
        ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert window % patch == 0
        self.embed = nn.Conv1d(in_ch, d_model, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, window // patch + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(64, 1))
        self.config = {"patch": patch, "d_model": d_model, "heads": heads, "layers": layers,
                       "ff": ff, "dropout": dropout}

    def forward(self, x):  # x: (B, T, C)
        tok = self.embed(x.transpose(1, 2)).transpose(1, 2)  # (B, T/patch, d)
        tok = torch.cat([self.cls.expand(len(x), -1, -1), tok], dim=1) + self.pos
        h = self.norm(self.encoder(tok)[:, 0])
        return self.head(h).squeeze(-1)


class EnsembleTeacher(nn.Module):
    """Average of several trained teachers' logits (all in eval mode)."""

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.config = {"members": len(members)}

    def forward(self, x):
        return torch.stack([m(x) for m in self.members]).mean(0)
