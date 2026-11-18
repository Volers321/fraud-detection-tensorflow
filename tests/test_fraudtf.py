import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from fraudtf import metrics, net, serving
from fraudtf.data import NUMERIC, build_features, simulate, split_by_day


@pytest.fixture(scope="module")
def small():
    df = simulate(n_cards=500, days=90, seed=1)
    X = build_features(df)
    tr, va, te = split_by_day(df)
    return df, X, tr, va, te


@pytest.fixture(scope="module")
def trained(small):
    df, X, tr, va, te = small
    y = df["is_fraud"].to_numpy()
    mean, std = net.fit_stats(X[tr])
    model = net.FraudNet(mean, std, seed=0)
    hist = net.train(model, X[tr], y[tr], X[va], y[va], epochs=12)
    return model, hist


def test_features_do_not_look_into_the_future(small):
    df, X, *_ = small
    prefix = df[df["time"] < df["time"].quantile(0.6)]
    Xp = build_features(prefix)
    pd.testing.assert_frame_equal(X.loc[Xp.index], Xp, check_exact=False, atol=1e-9)


def test_raw_inputs_shapes_and_dtypes(small):
    _, X, *_ = small
    num, ch, cat = net.raw_inputs(X.head(10))
    assert num.shape == (10, len(NUMERIC)) and num.dtype == np.float32
    assert ch.dtype == np.int32 and cat.dtype == np.int32


def test_in_graph_preprocessing_matches_numpy(small):
    _, X, tr, *_ = small
    mean, std = net.fit_stats(X[tr])
    model = net.FraudNet(mean, std)
    num, ch, cat = net.raw_inputs(X[tr].head(200))
    got = model.features(tf.constant(num), tf.constant(ch), tf.constant(cat)).numpy()[:, : len(NUMERIC)]
    a = X[tr].head(200)[NUMERIC].to_numpy("float64").copy()
    idx = [NUMERIC.index(c) for c in net.SKEWED]
    a[:, idx] = np.log(1.0 + np.maximum(a[:, idx], 0.0))
    want = np.clip((a - mean) / std, -6, 6)
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_serving_outputs_are_probabilities(small):
    _, X, tr, *_ = small
    mean, std = net.fit_stats(X[tr])
    p = net.FraudNet(mean, std).predict(X[tr].head(500))
    assert p.shape == (500,) and ((p >= 0) & (p <= 1)).all()


def test_training_improves_validation_pr_auc(trained):
    _, hist = trained
    assert hist["epochs_run"] >= 2
    assert hist["best_val_pr_auc"] >= hist["val_pr_auc"][0]
    assert hist["loss"][-1] < hist["loss"][0]


def test_trained_network_beats_the_base_rate_on_known_patterns(small, trained):
    df, X, tr, va, te = small
    model, _ = trained
    y = df["is_fraud"].to_numpy()
    known = te & (df["pattern"] != "mule_cashout").to_numpy()
    ap = metrics.average_precision(y[known], model.predict(X[known]))
    assert ap > 5 * y[known].mean()


def test_eager_and_graph_steps_agree(small):
    """With dropout off, the compiled training step must equal the eager one."""
    df, X, tr, *_ = small
    y = df["is_fraud"].to_numpy().astype("float32")
    mean, std = net.fit_stats(X[tr])
    num, ch, cat = net.raw_inputs(X[tr].head(256))
    yb = y[tr][:256]
    outs = []
    for eager in (False, True):
        m = net.FraudNet(mean, std, seed=5)
        step = net.make_train_step(m, net.Adam(m.params), pos_weight=10.0, dropout=0.0)
        step = step.python_function if eager else step
        for _ in range(3):
            loss = step(num, ch, cat, yb)
        outs.append((float(loss), m.w[0].numpy().copy()))
    assert outs[0][0] == pytest.approx(outs[1][0], rel=1e-4)
    np.testing.assert_allclose(outs[0][1], outs[1][1], atol=1e-5)


def test_saved_model_round_trip_preserves_scores(small, trained, tmp_path):
    _, X, _, _, te = small
    model, _ = trained
    path = str(tmp_path / "sm")
    assert serving.save(model, path) > 0
    p = serving.signature_scores(serving.load_signature(path), X[te].head(500))
    np.testing.assert_allclose(p, model.predict(X[te].head(500)), atol=1e-5)


def test_tflite_matches_saved_model(small, trained, tmp_path):
    _, X, _, _, te = small
    model, _ = trained
    path = str(tmp_path / "sm")
    serving.save(model, path)
    sc = serving.TFLiteScorer(serving.to_tflite(path))
    rows = X[te].head(300)
    np.testing.assert_allclose(sc.score(rows), model.predict(rows), atol=1e-4)


def test_latency_helper_reports_percentiles():
    r = serving.latency_ms(lambda i: sum(range(50)), n=100, warmup=5)
    assert 0 <= r["p50_ms"] <= r["p99_ms"]
