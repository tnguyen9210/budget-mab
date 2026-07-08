"""Plot performance regret / ln(B/c_min), reproducing Figure 1."""

import sys, os, pickle, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REGIME_TITLES = {
    "homogeneous": "Homogeneous arms",
    "moderate":    "Diverse arms",
    "extreme":     "Extremely diverse arms",
}

ALGO_STYLE = {
    "KUBE":                 dict(color="blue",   ls="-",  lw=2,   label="KUBE"),
    "Fractional KUBE":      dict(color="red",    ls="--", lw=2,   label="Fractional KUBE"),
    "UCB1":                 dict(color="green",  ls="-.", lw=1.5, label="UCB1"),
    "Random":               dict(color="purple", ls=":",  lw=1.5, label="Random"),
    "eps-first (eps=0.05)": dict(color="orange", ls="-.", lw=1,   label=r"$\varepsilon$-first (0.05)"),
    "eps-first (eps=0.1)":  dict(color="brown",  ls="--", lw=1,   label=r"$\varepsilon$-first (0.10)"),
    "eps-first (eps=0.15)": dict(color="gray",   ls=":",  lw=1,   label=r"$\varepsilon$-first (0.15)"),
}


def plot(results_path: str, out_dir: str):
    with open(results_path, "rb") as f:
        all_results = pickle.load(f)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    regimes = list(all_results.keys())
    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 5))
    if len(regimes) == 1:
        axes = [axes]

    for ax, regime in zip(axes, regimes):
        data = all_results[regime]
        budgets  = np.array(data["budgets"])
        c_min    = data["c_min"]
        log_norm = np.log(budgets / c_min)   # ln(B/c_min)

        algo_data = data["algorithms"]

        # O(B^{2/3} (ln B)^{-1}) reference — scale to match ε-first level
        ref = budgets**(2/3) / np.log(budgets)
        eps_y = np.array(algo_data.get(
            "eps-first (eps=0.1)", next(iter(algo_data.values()))
        )["regret"]) / log_norm
        ref = ref / ref[-1] * eps_y[-1] * 1.3
        ax.plot(budgets, ref, color="black", lw=2,
                label=r"$O(B^{2/3}(\ln B)^{-1})$")

        for algo_name, style in ALGO_STYLE.items():
            if algo_name not in algo_data:
                continue
            y     = np.array(algo_data[algo_name]["regret"])   / log_norm
            y_err = np.array(algo_data[algo_name]["regret_std"]) / log_norm
            kw = {k: v for k, v in style.items() if k != "label"}
            ax.plot(budgets, y, label=style["label"], **kw)
            ax.fill_between(budgets,
                            y - 1.96 * y_err, y + 1.96 * y_err,
                            alpha=0.15, color=style["color"])

        ax.set_title(REGIME_TITLES.get(regime, regime))
        ax.set_xlabel("Budget size")
        ax.set_ylabel(r"Performance regret / $\ln(B/c_{\min})$")
        # Cap y-axis at 25 to match paper's scale and show detail
        ax.set_ylim(bottom=0, top=25)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.04), fontsize=9)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = Path(out_dir) / f"figure1.{ext}"
        plt.savefig(p, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"Saved {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="../results/results.pkl")
    parser.add_argument("--out_dir", default="../results")
    args = parser.parse_args()
    plot(args.results, args.out_dir)
