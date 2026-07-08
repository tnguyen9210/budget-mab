"""Experiment runner for budget-limited MAB (Tran-Thanh et al. 2012).

Reproduces Figure 1:
  - K=10 arms, truncated Gaussian rewards, mu[i] ~ Uniform[10,20]
  - Three cost regimes: homogeneous, moderately diverse, and
    extremely diverse
  - Budget swept over a grid
  - One fixed bandit instance per regime; n_seeds varies only the
    reward noise
  - Performance regret divided by ln(B/c_min)

Usage (from the repo root, Hydra-style):
    python run_experiments.py --config-name default
    python run_experiments.py --config-name default n_seeds=10
    python run_experiments.py --config-name smoke regimes=[homogeneous]

Each run writes results/<name>.pkl plus a results/<name>.json
provenance sidecar (git commit, config used, timestamp) -- see
docs/issues.md for why this matters (a prior results.pkl of unknown
provenance could not be trusted for comparison).
"""

import json
import sys
import os
import time
import numpy as np
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)
from algorithms import (
    KUBE, FractionalKUBE, UCB1, Random, EpsilonFirst, BonusMode,
)
from exp_config import ExpConfig
from utils import git_commit, print_summary

cs = ConfigStore.instance()
cs.store(name="exp_schema", node=ExpConfig)


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
def _run_one(algo_kind, algo_arg, mu, costs, budget, seed):
    """Run a single trial. Returns regret (float).

    algo_kind / algo_arg encode which algorithm to instantiate so the
    function is picklable without passing a live algo object:
      - "eps-first" / eps (float)
      - "KUBE" | "Fractional KUBE" | "UCB1" / bonus_mode (BonusMode)
      - "Random" / None
    """
    import sys, os
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    )
    from env import BudgetMAB
    from algorithms import KUBE, FractionalKUBE, UCB1, Random, EpsilonFirst

    algo_map = {
        "KUBE":            lambda: KUBE(algo_arg),
        "Fractional KUBE": lambda: FractionalKUBE(algo_arg),
        "UCB1":            lambda: UCB1(algo_arg),
        "Random":          lambda: Random(),
        "eps-first":       lambda: EpsilonFirst(algo_arg),
    }
    algo = algo_map[algo_kind]()

    rng = np.random.default_rng(seed)
    env = BudgetMAB(mu=mu, costs=costs, budget=budget, rng=rng)
    reward = algo.run(env)
    oracle = env.oracle_reward()
    return oracle - reward   # regret


def _algo_key(algo):
    """Return (kind, arg) for _run_one dispatch -- see its docstring."""
    if isinstance(algo, EpsilonFirst):
        return ("eps-first", algo.eps)
    if isinstance(algo, Random):
        return ("Random", None)
    # KUBE / FractionalKUBE / UCB1 all carry a bonus_mode
    base = algo.name.split(" (")[0]
    return (base, algo.bonus_mode)


# Keyed by algo.name so conf/*.yaml's `algorithms:` list (a list of
# these keys) selects a subset without editing this file. Each of
# KUBE/FractionalKUBE/UCB1 appears once per BonusMode (see
# docs/decisions/ucb-exploration-bonus-scale.md for what the modes
# mean); Random and eps-first are unaffected by bonus_mode.
ALGO_REGISTRY = {
    algo.name: algo
    for algo in (
        Random(),
        UCB1(),
        EpsilonFirst(0.05),
        EpsilonFirst(0.10),
        EpsilonFirst(0.15),
        KUBE(),
        FractionalKUBE(),
        *(
            cls(mode)
            for cls in (KUBE, FractionalKUBE, UCB1)
            for mode in (BonusMode.TRUE_VAR, BonusMode.EST_VAR)
        ),
    )
}


def run_experiment(
    budgets,
    algorithms,
    K: int = 10,
    n_seeds: int = 500,
    regimes=("homogeneous", "moderate", "extreme"),
    instance_seed: int = 42,
    results_dir: str = "results",
    n_workers: Optional[int] = None,   # None → os.cpu_count()
    config_name: str = "results",
    config_used: Optional[dict] = None,
):
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
                algo_kind, algo_arg = _algo_key(algo)

                # Submit all seeds in parallel
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = {
                        pool.submit(
                            _run_one, algo_kind, algo_arg, mu, costs,
                            float(B), s,
                        ): s
                        for s in range(n_seeds)
                    }
                    regrets = [f.result() for f in as_completed(futures)]

                regime_results[algo.name]["regret"].append(
                    float(np.mean(regrets))
                )
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

    out_path = Path(results_dir) / f"{config_name}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved {out_path}")

    provenance = {
        "config_name": config_name,
        "config": config_used or {},
        "git_commit": git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "resolved_params": {
            "K": K,
            "n_seeds": n_seeds,
            "budgets": budgets.tolist(),
            "regimes": list(regimes),
            "instance_seed": instance_seed,
            "n_workers": n_workers,
        },
    }
    sidecar_path = Path(results_dir) / f"{config_name}.json"
    with open(sidecar_path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"Saved {sidecar_path}")

    print_summary(all_results)
    return all_results


@hydra.main(config_path="conf", config_name="default", version_base=None)
def main(cfg: ExpConfig):
    root_dir = hydra.utils.get_original_cwd()
    results_dir = os.path.join(root_dir, cfg.results_dir)

    try:
        algorithms = [ALGO_REGISTRY[name] for name in cfg.algorithms]
    except KeyError as e:
        raise ValueError(
            f"Unknown algorithm {e} in cfg.algorithms; valid keys: "
            f"{sorted(ALGO_REGISTRY)}"
        ) from e

    budgets = np.array([4000, 10000])
    run_experiment(
        budgets=budgets,
        algorithms=algorithms,
        K=cfg.K,
        n_seeds=cfg.n_seeds,
        regimes=tuple(cfg.regimes),
        instance_seed=cfg.instance_seed,
        results_dir=results_dir,
        n_workers=cfg.n_workers,
        config_name=cfg.name,
        config_used=OmegaConf.to_container(cfg, resolve=True),
    )


if __name__ == "__main__":
    main()
