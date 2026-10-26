"""The training loop, distillation and baselines learn an easy synthetic task (CPU, a few seconds)."""

import numpy as np
import torch

from sanad.data.augment import Augment
from sanad.data.windowing import PRE
from sanad.eval.metrics import threshold_for_sensitivity, window_metrics
from sanad.models.common import Normalize, channel_stats
from sanad.models.student import SanadTiny
from sanad.models.teacher import SanadTeacher
from sanad.train.baselines import FEATURES, features, train_forest
from sanad.train.loop import TrainConfig, predict, soft_bce, train


def toy(n, seed):
    """Falls: impact spike then lying still on the side. Non-falls: a spike, then upright again."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.05, size=(n, 200, 6)).astype(np.float32)
    y = (rng.random(n) < 0.3).astype(np.int8)
    x[:, :, 2] += 1.0  # upright: gravity on z
    x[:, PRE, 2] += rng.uniform(2, 5, n)
    x[y == 1, PRE + 10 :, 2] -= 1.0  # after a fall, gravity moves to x (lying down)
    x[y == 1, PRE + 10 :, 0] += 1.0
    return x, y


def tensors(x, y):
    return torch.as_tensor(x), torch.as_tensor(y)


def test_soft_bce_is_zero_gradient_when_matching():
    s = torch.tensor([0.3, -1.0], requires_grad=True)
    soft_bce(s, s.detach(), 2.0).backward()
    assert torch.allclose(s.grad, torch.zeros(2), atol=1e-6)


def test_plain_and_distilled_training_learn_the_toy_task():
    xtr, ytr = toy(600, 0)
    xva, yva = toy(200, 1)
    norm = Normalize(*channel_stats(xtr))
    cfg = TrainConfig(epochs=12, batch_size=64, warmup_epochs=1, patience=12, amp=False)
    aug = Augment(rot_deg=10)

    teacher = SanadTeacher(widths=(16, 32), blocks=1)
    out = train(teacher, norm, *tensors(xtr, ytr), *tensors(xva, yva), cfg, aug, log=lambda *_: None)
    assert out["best_val_auc"] > 0.95

    student = SanadTiny()
    extra = torch.as_tensor(toy(100, 2)[0])
    out = train(student, norm, *tensors(xtr, ytr), *tensors(xva, yva), cfg, aug, teacher=teacher,
                x_extra=extra, log=lambda *_: None)
    assert out["best_val_auc"] > 0.95

    p = predict(student, norm, torch.as_tensor(xva))
    thr = threshold_for_sensitivity(yva, p, 0.95)
    m = window_metrics(yva, p, thr)
    assert m["sensitivity"] >= 0.95 and m["specificity"] > 0.8


def test_forest_baseline():
    xtr, ytr = toy(600, 0)
    xva, yva = toy(200, 1)
    assert features(xtr).shape == (600, len(FEATURES))
    rf = train_forest(xtr, ytr)
    p = rf.predict_proba(features(xva))[:, 1]
    assert window_metrics(yva, p, 0.5)["auc"] > 0.95
