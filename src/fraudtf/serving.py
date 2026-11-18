"""Export and serving checks: SavedModel, TFLite (float and dynamic-range quantised), latency.

Fraud scoring sits in the authorisation path of a card payment, so latency per single
transaction matters as much as PR-AUC. This module measures it for each runtime and checks
that exporting and quantising did not change the decisions.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from .net import FraudNet, raw_inputs  # noqa: E402


def dir_size_kb(path: str) -> float:
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)
    return round(total / 1024, 1)


def save(net: FraudNet, path: str) -> float:
    tf.saved_model.save(net, path, signatures={"serving_default": net.serve})
    return dir_size_kb(path)


def load_signature(path: str):
    return tf.saved_model.load(path).signatures["serving_default"]


def signature_scores(sig, X, batch: int = 8192) -> np.ndarray:
    num, ch, cat = raw_inputs(X)
    out = [
        sig(num=tf.constant(num[i : i + batch]), channel=tf.constant(ch[i : i + batch]),
            category=tf.constant(cat[i : i + batch]))["p_fraud"].numpy()
        for i in range(0, len(num), batch)
    ]  # fmt: skip
    return np.concatenate(out)


def to_tflite(saved_model_dir: str, quantize: bool = False) -> bytes:
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if quantize:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range: int8 weights
    return conv.convert()


class TFLiteScorer:
    def __init__(self, model: bytes):
        self.interpreter = tf.lite.Interpreter(model_content=model)
        self.run = self.interpreter.get_signature_runner("serving_default")
        self.size_kb = round(len(model) / 1024, 1)

    def score_one(self, num, ch, cat) -> float:
        return float(self.run(num=num, channel=ch, category=cat)["p_fraud"][0])

    def score(self, X) -> np.ndarray:
        num, ch, cat = raw_inputs(X)
        return np.array([self.score_one(num[i : i + 1], ch[i : i + 1], cat[i : i + 1]) for i in range(len(num))])


def latency_ms(fn, n: int = 1500, warmup: int = 50) -> dict:
    for _ in range(warmup):
        fn(0)
    t = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        t.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": round(float(np.percentile(t, 50)), 3), "p99_ms": round(float(np.percentile(t, 99)), 3)}


def one_row_inputs(X, n: int = 1500):
    num, ch, cat = raw_inputs(X.iloc[:n])
    return num, ch, cat
