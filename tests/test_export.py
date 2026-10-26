"""PyTorch -> Keras -> int8 TFLite on a random (untrained) student. Slow-ish: imports TensorFlow."""

import numpy as np
import pytest
import torch

from sanad.models.student import SanadTiny

tf = pytest.importorskip("tensorflow")

from sanad.export.to_c_array import write_config_header, write_model_array
from sanad.export.to_keras import max_abs_diff, to_keras
from sanad.export.to_tflite import op_names, to_int8_tflite

TFLM_OK = {"CONV_2D", "MAX_POOL_2D", "MEAN", "FULLY_CONNECTED", "LOGISTIC", "RESHAPE",
           "EXPAND_DIMS", "SQUEEZE", "QUANTIZE", "DEQUANTIZE"}


@pytest.fixture(scope="module")
def student():
    torch.manual_seed(0)
    m = SanadTiny()
    for bn in (x for x in m.modules() if isinstance(x, torch.nn.BatchNorm1d)):
        bn.running_mean.uniform_(-0.5, 0.5)
        bn.running_var.uniform_(0.5, 2.0)
        bn.weight.data.uniform_(0.5, 1.5)
        bn.bias.data.uniform_(-0.2, 0.2)
    return m.eval()


@pytest.fixture(scope="module")
def windows():
    return np.random.default_rng(0).normal(size=(256, 200, 7)).astype(np.float32)


def test_keras_copy_matches_pytorch(student, windows):
    k = to_keras(student)
    assert max_abs_diff(student, k, windows[:64]) < 1e-4


def test_int8_model_is_small_close_and_tflm_compatible(student, windows, tmp_path):
    q = to_int8_tflite(to_keras(student), windows[:200])
    assert len(q.blob) < 25 * 1024
    with torch.no_grad():
        ref = torch.sigmoid(student(torch.as_tensor(windows[200:]))).numpy()
    assert np.max(np.abs(q.predict(windows[200:]) - ref)) < 0.05
    assert set(op_names(q.blob)) <= TFLM_OK, op_names(q.blob)

    files = write_model_array(q.blob, tmp_path)
    hdr = write_config_header(tmp_path, q, np.zeros(7), np.ones(7), trigger_g=2.0, decision_prob=0.5)
    assert all(f.exists() for f in files) and "SANAD_TRIGGER_G" in hdr.read_text()
