"""Command line: ``fraudtf eval`` trains, evaluates, exports and benchmarks the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(res: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    h = res["training"]
    ep = np.arange(1, len(h["loss"]) + 1)
    ax[0].plot(ep, h["val_pr_auc"], color="#d95f02", label="validation PR-AUC")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("validation PR-AUC")
    ax2 = ax[0].twinx()
    ax2.plot(ep, h["loss"], color="#9aa0a6", label="train loss")
    ax2.set_ylabel("train loss")
    ax[0].set_title("Custom GradientTape loop with early stopping")
    lat = {k: v for k, v in res["latency_batch1"].items() if "p50_ms" in v}
    names = list(lat)
    x = np.arange(len(names))
    ax[1].bar(x - 0.2, [lat[n]["p50_ms"] for n in names], 0.4, label="p50", color="#2b6cb0")
    ax[1].bar(x + 0.2, [lat[n]["p99_ms"] for n in names], 0.4, label="p99", color="#c05621")
    ax[1].set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=8)
    ax[1].set_ylabel("ms per transaction (batch of 1)")
    ax[1].set_title("Scoring latency by runtime")
    ax[1].legend(frameon=False)
    for a in (ax[0], ax[1], ax2):
        a.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="fraudtf")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="train, evaluate, export and benchmark")
    e.add_argument("--n-cards", type=int, default=2500)
    e.add_argument("--days", type=int, default=90)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--n-boot", type=int, default=200)
    e.add_argument("--out", default="results/eval.json")
    e.add_argument("--plot", default="docs/eval.png")
    a = p.parse_args(argv)
    from .pipeline import run_eval

    res = run_eval(a.n_cards, a.days, a.seed, a.n_boot)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    Path(a.plot).parent.mkdir(parents=True, exist_ok=True)
    plot(res, a.plot)
    print(json.dumps({k: res[k] for k in ("tf_mlp", "tf_function_vs_eager_ms_per_step", "export", "latency_batch1")}, indent=2))
    print(f"\nwrote {a.out} and {a.plot} in {res['runtime_s']} s")


if __name__ == "__main__":
    main()
