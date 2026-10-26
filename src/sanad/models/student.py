"""Sanad-Tiny: the model that runs on the Pico 2 W (approval-form design).

Conv(16,k7)-BN-ReLU-MaxPool2 -> Conv(32,k5)-BN-ReLU-MaxPool2 -> Conv(64,k3)-BN-ReLU-GAP
-> Dense 32-ReLU-Dropout 0.3 -> Dense 1 (logit; sigmoid is added at export).
Every layer maps 1:1 to a TFLite Micro int8 kernel. Input is (B, T=200, C=7), channels-last like TFLite.
"""

from __future__ import annotations

from torch import nn


class SanadTiny(nn.Module):
    def __init__(
        self,
        in_ch: int = 7,
        widths: tuple[int, ...] = (16, 32, 64),
        kernels: tuple[int, ...] = (7, 5, 3),
        dense: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert len(widths) == len(kernels) and all(k % 2 for k in kernels), "odd kernels only"
        layers: list[nn.Module] = []
        c = in_ch
        for i, (w, k) in enumerate(zip(widths, kernels)):
            layers += [nn.Conv1d(c, w, k, padding=k // 2, bias=False), nn.BatchNorm1d(w), nn.ReLU()]
            if i < len(widths) - 1:
                layers.append(nn.MaxPool1d(2))
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(c, dense), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dense, 1)
        )
        self.config = {"in_ch": in_ch, "widths": widths, "kernels": kernels, "dense": dense,
                       "dropout": dropout}

    def forward(self, x):  # x: (B, T, C)
        h = self.features(x.transpose(1, 2)).mean(dim=-1)
        return self.head(h).squeeze(-1)
