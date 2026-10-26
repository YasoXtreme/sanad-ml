import numpy as np
import pytest
import torch

from sanad.models.common import count_params
from sanad.models.oxford import OxfordTeacher, harnet5_features, load_pretrained
from sanad.models.transformer import EnsembleTeacher, SanadTransformer
from sanad.paths import RAW

OXFORD = RAW / "oxford_ssl" / "mtl_5_best.mdl"
MEAN, STD = np.zeros(7, np.float32), np.ones(7, np.float32)


def test_transformer_sizes_and_shapes():
    small = SanadTransformer()
    large = SanadTransformer(d_model=256, heads=8, layers=6, ff=512)
    x = torch.randn(4, 200, 7)
    assert small(x).shape == (4,) and large(x).shape == (4,)
    assert 400_000 < count_params(small) < 900_000
    assert 2_500_000 < count_params(large) < 6_000_000


def test_harnet5_reduces_120_and_150_samples_to_one_step():
    fe = harnet5_features().eval()
    for n in (120, 150):
        assert fe(torch.randn(2, 3, n)).shape == (2, 512, 1)


def test_oxford_teacher_scratch_runs():
    t = OxfordTeacher(MEAN, STD)
    assert t(torch.randn(3, 200, 7)).shape == (3,)


@pytest.mark.skipif(not OXFORD.exists(), reason="Oxford weights not downloaded")
def test_oxford_weights_load_by_name():
    fe = harnet5_features()
    before = fe.layer1[0].weight.clone()
    n = load_pretrained(fe, OXFORD)
    assert n == len(fe.state_dict())
    assert not torch.allclose(before, fe.layer1[0].weight)
    t = OxfordTeacher(MEAN, STD, pretrained=OXFORD).eval()
    assert torch.isfinite(t(torch.randn(3, 200, 7))).all()


def test_ensemble_averages_logits():
    a, b = SanadTransformer(layers=1), SanadTransformer(layers=1)
    e = EnsembleTeacher([a, b]).eval()
    x = torch.randn(2, 200, 7)
    with torch.no_grad():
        assert torch.allclose(e(x), (a.eval()(x) + b.eval()(x)) / 2, atol=1e-6)
