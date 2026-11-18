"""A fraud-scoring network written directly in TensorFlow (no Keras layers).

* ``FraudNet`` is a ``tf.Module`` holding plain ``tf.Variable`` weights.
* Preprocessing (log1p on skewed columns, standardisation, clipping) lives **inside** the
  graph, so the exported model takes raw features and cannot suffer train/serve skew.
* ``train`` is a hand-written loop: ``tf.data`` pipeline, ``tf.GradientTape``, a
  ``tf.function`` training step with a hand-written Adam update, weighted cross-entropy and
  early stopping on the validation PR-AUC.

Only operations that TFLite supports natively are used (``log`` instead of ``log1p``,
arithmetic masks instead of ``tf.where``), so the same graph converts to TFLite unchanged.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from . import metrics  # noqa: E402
from .data import CATEGORIES, NUMERIC  # noqa: E402

SKEWED = [
    "dist_home_km", "dist_prev_km", "speed_kmh", "cnt_10m", "cnt_1h", "cnt_24h",
    "sum_1h_ratio", "amount_ratio", "n_prev",
]  # fmt: skip
N_NUM, N_CHANNELS, N_CATEGORIES = len(NUMERIC), 3, len(CATEGORIES)


def raw_inputs(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three raw tensors the serving signature expects (nothing is pre-scaled)."""
    return (
        X[NUMERIC].to_numpy("float32"),
        X["channel"].to_numpy("int32"),
        X["category"].to_numpy("int32"),
    )


def fit_stats(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of the log-transformed numeric block, from the training rows only."""
    a = X[NUMERIC].to_numpy("float64").copy()
    idx = [NUMERIC.index(c) for c in SKEWED]
    a[:, idx] = np.log(1.0 + np.maximum(a[:, idx], 0.0))
    return a.mean(axis=0), a.std(axis=0) + 1e-6


class FraudNet(tf.Module):
    def __init__(self, mean, std, hidden=(64, 32), seed: int = 0):
        super().__init__(name="fraud_net")
        self.mean = tf.constant(mean, tf.float32)
        self.std = tf.constant(std, tf.float32)
        mask = np.zeros(N_NUM, "float32")
        mask[[NUMERIC.index(c) for c in SKEWED]] = 1.0
        self.skew = tf.constant(mask)
        g = tf.random.Generator.from_seed(seed)
        self.emb_channel = tf.Variable(g.normal([N_CHANNELS, 2], stddev=0.1), name="emb_channel")
        self.emb_category = tf.Variable(g.normal([N_CATEGORIES, 4], stddev=0.1), name="emb_category")
        dims = [N_NUM + 2 + 4, *hidden, 1]
        self.w, self.b = [], []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            self.w.append(tf.Variable(g.normal([a, b], stddev=float(np.sqrt(2.0 / a))), name=f"w{i}"))
            self.b.append(tf.Variable(tf.zeros([b]), name=f"b{i}"))

    def features(self, num, channel, category):
        logged = tf.math.log(1.0 + tf.maximum(num, 0.0))
        x = self.skew * logged + (1.0 - self.skew) * num
        x = tf.clip_by_value((x - self.mean) / self.std, -6.0, 6.0)
        return tf.concat([x, tf.gather(self.emb_channel, channel), tf.gather(self.emb_category, category)], axis=1)

    def logits(self, num, channel, category, training: bool = False, dropout: float = 0.2):
        h = self.features(num, channel, category)
        for w, b in zip(self.w[:-1], self.b[:-1]):
            h = tf.nn.relu(h @ w + b)
            if training:
                h = tf.nn.dropout(h, dropout)
        return (h @ self.w[-1] + self.b[-1])[:, 0]

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, N_NUM], tf.float32, name="num"),
            tf.TensorSpec([None], tf.int32, name="channel"),
            tf.TensorSpec([None], tf.int32, name="category"),
        ]
    )
    def serve(self, num, channel, category):
        return {"p_fraud": tf.sigmoid(self.logits(num, channel, category))}

    @property
    def params(self):
        return [self.emb_channel, self.emb_category, *self.w, *self.b]

    def predict(self, X: pd.DataFrame, batch: int = 8192) -> np.ndarray:
        num, ch, cat = raw_inputs(X)
        return np.concatenate(
            [self.serve(num[i : i + batch], ch[i : i + batch], cat[i : i + batch])["p_fraud"].numpy()
             for i in range(0, len(num), batch)]
        )  # fmt: skip


class Adam:
    """Adam written with tf ops, applied inside the compiled training step."""

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-7):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [tf.Variable(tf.zeros_like(p), trainable=False) for p in params]
        self.v = [tf.Variable(tf.zeros_like(p), trainable=False) for p in params]
        self.t = tf.Variable(0.0, trainable=False)

    def apply(self, grads):
        self.t.assign_add(1.0)
        c1, c2 = 1.0 - self.b1**self.t, 1.0 - self.b2**self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = tf.convert_to_tensor(g)  # embedding gradients arrive as sparse IndexedSlices
            m.assign(self.b1 * m + (1.0 - self.b1) * g)
            v.assign(self.b2 * v + (1.0 - self.b2) * tf.square(g))
            p.assign_sub(self.lr * (m / c1) / (tf.sqrt(v / c2) + self.eps))


def make_train_step(net: FraudNet, opt: Adam, pos_weight: float, l2: float = 1e-5, dropout: float = 0.2):
    @tf.function(reduce_retracing=True)
    def step(num, channel, category, y):
        with tf.GradientTape() as tape:
            z = net.logits(num, channel, category, training=dropout > 0, dropout=dropout)
            ce = tf.nn.weighted_cross_entropy_with_logits(y, z, pos_weight)
            loss = tf.reduce_mean(ce) + l2 * tf.add_n([tf.reduce_sum(tf.square(w)) for w in net.w])
        opt.apply(tape.gradient(loss, net.params))
        return loss

    return step


def train(net: FraudNet, X_tr, y_tr, X_va, y_va, epochs: int = 40, patience: int = 6,
          batch: int = 1024, seed: int = 0, eager: bool = False) -> dict:  # fmt: skip
    """Training loop with early stopping on validation PR-AUC. Returns the history."""
    y_tr = np.asarray(y_tr, "float32")
    pos_weight = float(np.sqrt((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)))
    opt = Adam(net.params)
    step = make_train_step(net, opt, pos_weight)
    if eager:  # same code without graph compilation, for the timing comparison
        step = step.python_function
    num, ch, cat = raw_inputs(X_tr)
    ds = tf.data.Dataset.from_tensor_slices((num, ch, cat, y_tr))
    ds = ds.shuffle(50_000, seed=seed, reshuffle_each_iteration=True).batch(batch).prefetch(2)
    best, best_w, bad = -1.0, None, 0
    hist = {"loss": [], "val_pr_auc": [], "epoch_s": []}
    import time

    for _ in range(epochs):
        t0, losses = time.time(), []
        for xb in ds:
            losses.append(float(step(*xb)))
        val = metrics.average_precision(np.asarray(y_va), net.predict(X_va))
        hist["loss"].append(round(float(np.mean(losses)), 5))
        hist["val_pr_auc"].append(round(float(val), 4))
        hist["epoch_s"].append(round(time.time() - t0, 2))
        if val > best + 1e-4:
            best, bad, best_w = val, 0, [p.numpy().copy() for p in net.params]
        else:
            bad += 1
            if bad >= patience:
                break
    for p, w in zip(net.params, best_w):
        p.assign(w)
    hist["best_val_pr_auc"], hist["epochs_run"] = round(float(best), 4), len(hist["loss"])
    return hist
