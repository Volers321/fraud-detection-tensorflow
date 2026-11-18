"""Evaluation metrics for fraud detection (identical in all three repositories).

With ~0.5% fraud, accuracy and ROC-AUC say little. We report:
* ``average_precision`` (PR-AUC): ranking quality,
* recall and precision at an **alert budget**: analysts can review only X% of transactions;
  the threshold is chosen on validation and applied unchanged to test, as in production,
* net savings in PLN: value of stopped fraud minus the cost of reviewing alerts,
* recall per attack pattern, including the new one that was absent from training,
* confidence intervals from a card-level bootstrap (one card's transactions are not independent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REVIEW_COST_PLN = 5.0  # cost of manually reviewing one alert
BUDGET = 0.005  # review about 0.5% of transactions


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC as the step-wise area under the precision-recall curve (same definition as
    scikit-learn's ``average_precision_score``, implemented in numpy so that the Keras and
    TensorFlow repositories do not depend on scikit-learn)."""
    y = np.asarray(y).astype(float)
    order = np.argsort(-np.asarray(score), kind="stable")
    y, s = y[order], np.asarray(score)[order]
    tp, fp = np.cumsum(y), np.cumsum(1.0 - y)
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # last index of each distinct score
    tp, fp = tp[last], fp[last]
    if tp[-1] == 0:
        return 0.0
    precision, recall = tp / (tp + fp), tp / tp[-1]
    return float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))


def threshold_for_budget(val_scores: np.ndarray, budget: float = BUDGET) -> float:
    """Threshold at which ``budget`` of validation transactions raise an alert."""
    return float(np.quantile(val_scores, 1.0 - budget))


def evaluate(df: pd.DataFrame, score: np.ndarray, thr: float) -> dict:
    """``df``: test frame (is_fraud, amount, pattern); ``score``: higher = more suspicious."""
    y = df["is_fraud"].to_numpy()
    alert = score >= thr
    caught = alert & (y == 1)
    amount = df["amount"].to_numpy()
    out = {
        "n": int(len(df)),
        "n_fraud": int(y.sum()),
        "pr_auc": round(float(average_precision(y, score)), 4),
        "alert_rate": round(float(alert.mean()), 4),
        "recall": round(float(caught.sum() / max(y.sum(), 1)), 4),
        "precision": round(float(caught.sum() / max(alert.sum(), 1)), 4),
        "fraud_amount_caught_share": round(float(amount[caught].sum() / max(amount[y == 1].sum(), 1)), 4),
        "net_savings_pln": round(float(amount[caught].sum() - REVIEW_COST_PLN * alert.sum()), 0),
    }
    by = {}
    for p, g in df.loc[df["is_fraud"] == 1].groupby("pattern"):
        by[p] = {"n": int(len(g)), "recall": round(float(alert[g.index.map(df.index.get_loc)].mean()), 3)}
    out["recall_by_pattern"] = by
    known = (df["pattern"] != "mule_cashout").to_numpy()  # patterns seen in training
    out["pr_auc_known_patterns"] = round(float(average_precision(y[known], score[known])), 4)
    return out


def bootstrap_ci(df: pd.DataFrame, score: np.ndarray, thr: float, n_boot: int = 300, seed: int = 0) -> dict:
    """95% CI for PR-AUC and recall; cards are resampled with replacement."""
    rng = np.random.default_rng(seed)
    cards = df["card_id"].to_numpy()
    uniq, inv = np.unique(cards, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(inv))]
    y = df["is_fraud"].to_numpy()
    ap, rec = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[starts[k] : starts[k + 1]] for k in pick])
        if y[idx].sum() == 0:
            continue
        ap.append(average_precision(y[idx], score[idx]))
        rec.append(((score[idx] >= thr) & (y[idx] == 1)).sum() / y[idx].sum())
    q = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]  # noqa: E731
    return {"pr_auc_ci95": q(ap), "recall_ci95": q(rec)}
