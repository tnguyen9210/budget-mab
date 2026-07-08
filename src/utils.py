"""Shared helpers used by run_experiments.py and plot_results.py."""
import os
import subprocess

import numpy as np


def git_commit() -> str:
    """Current HEAD commit hash, or "unknown" outside a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def reference_curve(budgets: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Paper's O(B^(2/3)/ln B) growth-rate guide (normalised regret).

    The paper states this as a growth-rate shape only -- Theorem 8
    proves a lower bound of C*ln(B) for some unspecified worst-case C,
    it does NOT fix the constant for this B^(2/3)/ln(B) curve. `anchor`
    (normalised regret of some series, e.g. eps-first(eps=0.1)) is
    used to scale the curve to sit near the data at the largest
    budget -- this scaling is an arbitrary plotting choice, not
    something the paper specifies.
    """
    ref = budgets ** (2 / 3) / np.log(budgets)
    return ref / ref[-1] * anchor[-1] * 1.3


def print_summary(all_results: dict) -> None:
    """Print raw and normalised (regret / ln(B/c_min)) regret."""
    for regime, data in all_results.items():
        budgets = np.array(data["budgets"])
        c_min = data["c_min"]
        log_norm = np.log(budgets / c_min)
        algo_data = data["algorithms"]

        col_w = 16
        name_w = max(
            [len("O(B^(2/3)/lnB) [ref]")]
            + [len(n) for n in algo_data] + [9]
        ) + 2
        print(f"\n=== {regime} (c_min={c_min}) ===")
        header = "algorithm".ljust(name_w) + "".join(
            f"B={b}".ljust(col_w) for b in budgets
        )
        print(header)
        for algo_name, vals in algo_data.items():
            regret = np.array(vals["regret"])
            norm = regret / log_norm
            cells = "".join(
                f"{r:7.2f} ({n:5.2f})".ljust(col_w)
                for r, n in zip(regret, norm)
            )
            print(f"{algo_name.ljust(name_w)}{cells}")

        anchor_name = "eps-first (eps=0.1)"
        if anchor_name in algo_data:
            anchor = np.array(algo_data[anchor_name]["regret"]) / log_norm
            ref = reference_curve(budgets, anchor)
            cells = "".join(
                f"{'':7} ({r:5.2f})".ljust(col_w) for r in ref
            )
            print(f"{'O(B^(2/3)/lnB) [ref]'.ljust(name_w)}{cells}")

        # Normalised-only view, for a direct comparison against the
        # reference curve above without the raw-regret clutter.
        print(f"\n--- {regime}: normalised regret only ---")
        print(header)
        for algo_name, vals in algo_data.items():
            norm = np.array(vals["regret"]) / log_norm
            cells = "".join(f"{n:<16.2f}" for n in norm)
            print(f"{algo_name.ljust(name_w)}{cells}")
        if anchor_name in algo_data:
            cells = "".join(f"{r:<16.2f}" for r in ref)
            print(f"{'O(B^(2/3)/lnB) [ref]'.ljust(name_w)}{cells}")
    print(
        "\n(raw regret with normalised regret in parentheses; "
        "reference curve is normalised-only, scaled to "
        f"{anchor_name!r} at the largest budget -- see "
        "reference_curve() docstring, the constant is not from "
        "the paper)"
    )
