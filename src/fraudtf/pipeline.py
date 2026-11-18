"""End-to-end study with the same temporal protocol as the sklearn and Keras repositories,
plus what is specific to TensorFlow: tf.function speed-up, export parity, TFLite quantisation
and per-transaction latency.

Days 0-59 train, 60-74 validation (early stopping + threshold), 75-89 test; the
``mule_cashout`` attack appears only in the test period.
"""

from __future__ import annotations

import os
import shutil
import time

import numpy as np
import tensorflow as tf

from . import metrics, net, serving
from .data import build_features, simulate, split_by_day


def bench_step(X_tr, y_tr, eager: bool, n_batches: int = 40, batch: int = 1024) -> float:
    """Mean milliseconds per training step, graph-compiled vs eager."""
    mean, std = net.fit_stats(X_tr)
    m = net.FraudNet(mean, std)
    step = net.make_train_step(m, net.Adam(m.params), pos_weight=15.0)
    if eager:
        step = step.python_function
    num, ch, cat = net.raw_inputs(X_tr)
    y = np.asarray(y_tr, "float32")
    xs = [(num[i : i + batch], ch[i : i + batch], cat[i : i + batch], y[i : i + batch]) for i in range(0, n_batches * batch, batch)]
    step(*xs[0])  # trace / warm up
    t0 = time.perf_counter()
    for xb in xs[1:]:
        step(*xb)
    return round((time.perf_counter() - t0) / (len(xs) - 1) * 1000, 2)


def run_eval(n_cards: int = 2500, days: int = 90, seed: int = 0, n_boot: int = 200, workdir: str = "artifacts") -> dict:
    t0 = time.time()
    tf.keras.utils.set_random_seed(seed)
    df = simulate(n_cards, days, seed)
    X = build_features(df)
    tr, va, te = split_by_day(df)
    y = df["is_fraud"].to_numpy()
    dte = df[te].reset_index(drop=True)
    out: dict = {
        "data": {
            "n_cards": n_cards, "days": days, "seed": seed, "n_transactions": int(len(df)),
            "fraud_rate": round(float(y.mean()), 4),
            "n_fraud": {"train": int(y[tr].sum()), "validation": int(y[va].sum()), "test": int(y[te].sum())},
        }
    }  # fmt: skip

    mean, std = net.fit_stats(X[tr])
    model = net.FraudNet(mean, std, seed=seed)
    hist = net.train(model, X[tr], y[tr], X[va], y[va], seed=seed)
    out["training"] = hist
    sv, st = model.predict(X[va]), model.predict(X[te])
    thr = metrics.threshold_for_budget(sv)
    res = metrics.evaluate(dte, st, thr)
    res.update(metrics.bootstrap_ci(dte, st, thr, n_boot=n_boot, seed=seed))
    out["tf_mlp"] = res
    out["tf_function_vs_eager_ms_per_step"] = {
        "graph": bench_step(X[tr], y[tr], eager=False),
        "eager": bench_step(X[tr], y[tr], eager=True),
    }

    # export, reload, and check that the decisions are unchanged
    shutil.rmtree(workdir, ignore_errors=True)
    sm_dir = os.path.join(workdir, "saved_model")
    sm_kb = serving.save(model, sm_dir)
    sig = serving.load_signature(sm_dir)
    p_sm = serving.signature_scores(sig, X[te])
    exp = {
        "saved_model": {
            "size_kb": sm_kb,
            "max_abs_diff_vs_in_memory": round(float(np.abs(p_sm - st).max()), 8),
            "same_alert_decisions": bool(((p_sm >= thr) == (st >= thr)).all()),
        }
    }
    num, ch, cat = serving.one_row_inputs(X[te])
    lat = {
        "saved_model_signature": serving.latency_ms(
            lambda i: sig(num=tf.constant(num[i : i + 1]), channel=tf.constant(ch[i : i + 1]), category=tf.constant(cat[i : i + 1]))
        )
    }  # fmt: skip
    for name, quant in (("tflite_float32", False), ("tflite_dynamic_int8", True)):
        try:
            sc = serving.TFLiteScorer(serving.to_tflite(sm_dir, quantize=quant))
            p = sc.score(X[te])
            r = metrics.evaluate(dte, p, thr)
            exp[name] = {
                "size_kb": sc.size_kb,
                "max_abs_diff_vs_in_memory": round(float(np.abs(p - st).max()), 6),
                "pr_auc": r["pr_auc"], "recall": r["recall"],
                "same_alert_decisions_share": round(float(((p >= thr) == (st >= thr)).mean()), 5),
            }  # fmt: skip
            lat[name] = serving.latency_ms(lambda i: sc.score_one(num[i : i + 1], ch[i : i + 1], cat[i : i + 1]))
        except Exception as e:  # keep the study alive and record why a conversion failed
            exp[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    out["export"], out["latency_batch1"] = exp, lat
    out["runtime_s"] = round(time.time() - t0, 1)
    return out
