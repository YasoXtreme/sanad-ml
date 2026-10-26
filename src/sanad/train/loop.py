"""One training loop for the teacher, the plain student and the distilled student.

- Raw windows (B, 200, 6) live on the GPU; each batch is augmented, then |a| is added and normalized.
- Plain training: class-weighted BCE on labels.
- Distillation: the teacher scores the *same* augmented batch, and the student matches its softened
  probabilities (temperature T) as well as the labels. Extra daily-activity windows with no labels
  (e.g. PAMAP2) are scored by the teacher only.
- Early stopping on validation AUC; the best weights are restored at the end.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn

from sanad.data.augment import Augment
from sanad.models.common import Normalize


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    patience: int = 15
    amp: bool = True
    seed: int = 0
    # distillation (ignored without a teacher)
    alpha: float = 0.5  # weight of the label loss; (1 - alpha) goes to matching the teacher
    temperature: float = 2.0
    extra_per_batch: int = 64  # unlabeled teacher-scored windows mixed into each batch


def soft_bce(student_logit: torch.Tensor, teacher_logit: torch.Tensor, t: float) -> torch.Tensor:
    """Binary distillation loss: BCE between softened probabilities, scaled by T^2."""
    target = torch.sigmoid(teacher_logit / t)
    return F.binary_cross_entropy_with_logits(student_logit / t, target) * t * t


@torch.no_grad()
def predict(model: nn.Module, norm: Normalize, x: torch.Tensor, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(norm(x[i : i + batch]))).float() for i in range(0, len(x), batch)]
    return torch.cat(out).cpu().numpy() if out else np.zeros(0, np.float32)


def train(
    model: nn.Module,
    norm: Normalize,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: TrainConfig,
    augment: Augment | None = None,
    teacher: nn.Module | None = None,
    x_extra: torch.Tensor | None = None,
    log=print,
) -> dict:
    """Train in place and return the history. Tensors must already be on the training device."""
    dev = x_train.device
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    augment = augment or Augment()
    model.to(dev)
    norm.to(dev)
    if teacher is not None:
        teacher.to(dev).eval()

    pos = float(y_train.sum())
    pos_weight = torch.tensor((len(y_train) - pos) / max(pos, 1.0), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(x_train) / cfg.batch_size)
    total = cfg.epochs * steps_per_epoch
    warm = cfg.warmup_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1)))
    )
    use_amp = cfg.amp and dev.type == "cuda"
    y_val_np = y_val.cpu().numpy()

    best = {"auc": -1.0, "epoch": -1, "state": None}
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(x_train), device=dev, generator=gen)
        tot_loss = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            xb, yb = x_train[idx], y_train[idx].float()
            n_lab = len(xb)
            if teacher is not None and x_extra is not None and len(x_extra) and cfg.extra_per_batch:
                e = torch.randint(0, len(x_extra), (cfg.extra_per_batch,), device=dev, generator=gen)
                xb = torch.cat([xb, x_extra[e]])
            xb = norm(augment(xb, generator=gen))
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                logit = model(xb).float()
                loss = F.binary_cross_entropy_with_logits(logit[:n_lab], yb, pos_weight=pos_weight)
                if teacher is not None:
                    with torch.no_grad():
                        t_logit = teacher(xb).float()
                    loss = cfg.alpha * loss + (1 - cfg.alpha) * soft_bce(logit, t_logit, cfg.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot_loss += loss.item() * n_lab

        p_val = predict(model, norm, x_val)
        auc = float(roc_auc_score(y_val_np, p_val)) if len(np.unique(y_val_np)) == 2 else float("nan")
        history.append({"epoch": epoch, "loss": tot_loss / len(x_train), "val_auc": auc})
        if auc > best["auc"]:
            best = {"auc": auc, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            log(f"epoch {epoch:3d}  loss {tot_loss / len(x_train):.4f}  val AUC {auc:.4f}")
        if epoch - best["epoch"] >= cfg.patience:
            log(f"early stop at epoch {epoch}; best val AUC {best['auc']:.4f} at epoch {best['epoch']}")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"history": history, "best_epoch": best["epoch"], "best_val_auc": best["auc"]}
