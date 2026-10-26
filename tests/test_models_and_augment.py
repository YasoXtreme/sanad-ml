import numpy as np
import pandas as pd
import torch

from sanad.data.augment import Augment, random_rotations
from sanad.data.splits import assert_no_leakage, subject_folds
from sanad.models.common import Normalize, channel_stats, count_macs, count_params
from sanad.models.student import SanadTiny
from sanad.models.teacher import SanadTeacher


def test_student_fits_the_pico_budget():
    m = SanadTiny()
    assert m(torch.randn(4, 200, 7)).shape == (4,)
    assert 11_000 < count_params(m) < 15_000
    assert 600_000 < count_macs(m, (200, 7)) < 800_000


def test_teacher_is_much_bigger():
    m = SanadTeacher()
    assert m(torch.randn(4, 200, 7)).shape == (4,)
    assert 500_000 < count_params(m) < 1_500_000


def test_normalize_adds_magnitude_channel():
    x6 = np.random.default_rng(0).normal(size=(32, 200, 6)).astype(np.float32)
    mean, std = channel_stats(x6)
    z = Normalize(mean, std)(torch.as_tensor(x6))
    assert z.shape == (32, 200, 7)
    assert torch.allclose(z.reshape(-1, 7).mean(0), torch.zeros(7), atol=1e-4)


def test_rotations_are_proper_and_bounded():
    r = random_rotations(256, 25.0)
    assert torch.allclose(r @ r.transpose(1, 2), torch.eye(3).expand(256, 3, 3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(r), torch.ones(256), atol=1e-5)
    angle = torch.rad2deg(torch.arccos(((r.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1, 1)))
    assert angle.max() <= 25.0 + 1e-3


def test_augment_identity_and_mirror():
    x = torch.randn(8, 200, 6)
    off = Augment(rot_deg=0, mirror_p=0, scale=(1, 1), shift=0, noise_acc_g=0, noise_gyro_dps=0)
    assert torch.allclose(off(x), x.clamp(-8, 8))
    flip = Augment(rot_deg=0, mirror_p=1, scale=(1, 1), shift=0, noise_acc_g=0, noise_gyro_dps=0)
    sign = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])  # acc: flip y; gyro: flip x and z
    expected = torch.cat([(x[..., :3] * sign[:3]).clamp(-8, 8), x[..., 3:] * sign[3:]], dim=-1)
    assert torch.allclose(flip(x), expected)


def test_folds_never_share_subjects():
    rng = np.random.default_rng(0)
    subj = rng.integers(0, 30, size=2000)
    meta = pd.DataFrame({"subject_id": [f"demo:{s}" for s in subj]})
    y = (rng.random(2000) < 0.2).astype(np.int8)
    fold = subject_folds(meta, y, n_splits=5)
    for k in range(5):
        assert_no_leakage(meta, fold != k, fold == k)
    assert set(np.unique(fold)) == set(range(5))
