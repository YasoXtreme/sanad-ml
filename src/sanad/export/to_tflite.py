"""Keras SanadTiny -> full-integer int8 TFLite (what TFLite Micro runs on the Pico 2 W)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@dataclass
class Int8Model:
    blob: bytes
    in_scale: float
    in_zero: int
    out_scale: float
    out_zero: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Fall probabilities for normalized float windows (B, T, 7), run through the int8 model."""
        interp = _interpreter(self.blob)
        i_det, o_det = interp.get_input_details()[0], interp.get_output_details()[0]
        q = np.clip(np.round(x / self.in_scale + self.in_zero), -128, 127).astype(np.int8)
        out = np.empty(len(x), dtype=np.float32)
        for j in range(len(x)):
            interp.set_tensor(i_det["index"], q[j : j + 1])
            interp.invoke()
            out[j] = (interp.get_tensor(o_det["index"]).astype(np.float32).ravel()[0]
                      - self.out_zero) * self.out_scale
        return out


def _interpreter(blob: bytes):
    """TFLite interpreter with *reference* kernels and no XNNPACK delegate.

    Reference int8 kernels are what TFLite Micro (and CMSIS-NN, bit-exact for int8) runs on the
    Pico, so accuracy measured here is what the band gets.
    """
    import warnings

    import tensorflow as tf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tf.lite.Interpreter deprecation notice
        interp = tf.lite.Interpreter(
            model_content=blob,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
        )
    interp.allocate_tensors()
    return interp


def to_int8_tflite(kmodel, representative: np.ndarray) -> Int8Model:
    """`representative`: a few hundred normalized training windows (B, T, 7), both classes."""
    import tensorflow as tf

    rep = representative.astype(np.float32)

    def gen():
        for i in range(len(rep)):
            yield [rep[i : i + 1]]

    conv = tf.lite.TFLiteConverter.from_keras_model(kmodel)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()

    interp = _interpreter(blob)
    i_det, o_det = interp.get_input_details()[0], interp.get_output_details()[0]
    assert i_det["dtype"] == np.int8 and o_det["dtype"] == np.int8, "conversion is not full-int8"
    (i_s, i_z), (o_s, o_z) = i_det["quantization"], o_det["quantization"]
    return Int8Model(blob, float(i_s), int(i_z), float(o_s), int(o_z))


def op_names(blob: bytes) -> list[str]:
    """Operators used by the model (to check TFLite Micro supports all of them)."""
    interp = _interpreter(blob)
    return sorted({d["op_name"] for d in interp._get_ops_details()})
