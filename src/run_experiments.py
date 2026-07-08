"""Experiment runner for budget-limited MAB (Tran-Thanh et al. 2012).

Reproduces Figure 1:
  - K=10 arms, truncated Gaussian rewards, mu[i] ~ Uniform[10,20]
  - Three cost regimes: homogeneous, moderately diverse, extremely diverse
  - Budget swept over a grid
  - One fixed bandit instance per regime; n_seeds varies only reward noise
  - Performance regret divided by ln(B/c_min)

Usage:
    python run_experiments.py [--n_seeds N] [--n_budgets N] [--K N] [--n_workers N]
"""

import argparse
import sys
import os
import numpy as np
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
from algorithms import KUBE, FractionalKUBE, UCB1, Random, EpsilonFirst


def make_costs(K: int, regime: str, rng: np.random.Generator) -> np.ndarray:
    """Paper Section 5 cost distributions, rounded to integers >= 1."""
    if regime == "homogeneous":
        costs = rng.uniform(5, 10, size=K)
    elif regime == "moderate":
        costs = rng.uniform(1, 10, size=K)
    elif regime == "extreme":
        costs = rng.uniform(1, 20, size=K)
    else:
        raise ValueError(regime)
    return np.maximum(costs.round().astype(float), 1.0)


def make_mu(K: int, rng: np.random.Generator) -> np.ndarray:
    """mu[i] ~ Uniform[10,20] (paper Section 5)."""
    return rng.uniform(10, 20, size=K)


# Top-level so ProcessPoolExecutor can pickle it.
def _run_one(algo_name, algo_eps, mu, costs, budget, seed):
    """Run a single trial. Returns regret (float).

    algo_name / algo_eps encode which algorithm to instantiate so the
    function is picklable without passing a live algo object.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from env import BudgetMAB
    from algorithms import KUBE, FractionalKUBE, UCB1, Random, EpsilonFirst

    algo_map = {
        "KUBE":           lambda: KUBE(),
        "Fractional KUBE": lambda: FractionalKUBE(),
        "UCB1":           lambda: UCB1(),
        "Random":         lambda: Random(),
        "eps-first":      lambda: EpsilonFirst(algo_eps),
    }
    algo = algo_map[algo_name]()

    rng = np.random.default_rng(seed)
    env = BudgetMAB(mu=mu, costs=costs, budget=budget, rng=rng)
    reward = algo.run(env)
    oracle = env.oracle_reward()
    return oracle - reward   # regret


def _algo_key(algo):
    """Return (name_key, eps) for _run_one dispatch."""
    if isinstance(algo, EpsilonFirst):
        return ("eps-first", algo.eps)
    return (algo.name, None)


def run_experiment(
    K: int = 10,
    n_seeds: int = 500,
    budgets=None,
    regimes=("homogeneous", "moderate", "extreme"),
    instance_seed: int = 42,
    results_dir: str = "../results",
    n_workers: Optional[int] = None,   # None → os.cpu_count()
):
    if budgets is None:
        budgets = np.linspace(500, 4000, 15, dtype=int)

    algorithms = [
        KUBE(),
        FractionalKUBE(),
        UCB1(),
        Random(),
        EpsilonFirst(0.05),
        EpsilonFirst(0.10),
        EpsilonFirst(0.15),
    ]

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    all_results = {}

    for regime in regimes:
        print(f"\n=== Regime: {regime} ===")

        inst_rng = np.random.default_rng(instance_seed)
        mu = make_mu(K, inst_rng)
        costs = make_costs(K, regime, inst_rng)
        c_min = float(costs.min())

        print(f"  mu_norm: {(mu/40).round(3)}")
        print(f"  costs:   {costs}")
        print(f"  c_min={c_min}, initial_phase_cost={costs.sum():.0f}")
        I_star = int(np.argmax(mu / costs))
        print(f"  I*={I_star}, density*={mu[I_star]/40/costs[I_star]:.4f}")

        regime_results = {
            algo.name: {"regret": [], "regret_std": []}
            for algo in algorithms
        }

        for B in budgets:
            print(f"  B={int(B)}", end="", flush=True)

            for algo in algorithms:
                name_key, eps = _algo_key(algo)

                # Submit all seeds in parallel
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = {
                        pool.submit(_run_one, name_key, eps, mu, costs, float(B), s): s
                        for s in range(n_seeds)
                    }
                    regrets = [f.result() for f in as_completed(futures)]

                regime_results[algo.name]["regret"].append(float(np.mean(regrets)))
                regime_results[algo.name]["regret_std"].append(
                    float(np.std(regrets) / np.sqrt(n_seeds))
                )
                print(".", end="", flush=True)
            print()

        all_results[regime] = {
            "budgets": budgets.tolist(),
            "c_min": c_min,
            "algorithms": regime_results,
        }

    out_path = Path(results_dir) / "results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved {out_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds",    type=int,  default=500)
    parser.add_argument("--n_budgets",  type=int,  default=15)
    parser.add_argument("--K",          type=int,  default=10)
    parser.add_argument("--n_workers",  type=int,  default=None)
    parser.add_argument("--results_dir", type=str, default="../results")
    args = parser.parse_args()

    budgets = np.linspace(500, 4000, args.n_budgets, dtype=int)
    run_experiment(
        K=args.K,
        n_seeds=args.n_seeds,
        budgets=budgets,
        results_dir=args.results_dir,
        n_workers=args.n_workers,
    )
