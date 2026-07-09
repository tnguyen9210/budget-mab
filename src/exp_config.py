"""Structured config for run_experiments.py (Hydra/OmegaConf).

Mirrors the llm-reasoning-methods repo's conf/ pattern: a typed
ExpConfig dataclass registered with Hydra's ConfigStore, composed from
conf/<name>.yaml files via `defaults:`.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExpConfig:
    """Top-level experiment config.

    Field names match run_experiments.run_experiment's kwargs
    one-to-one so main() can pass cfg straight through.
    """
    name: str = "results"
    K: int = 10
    n_seeds: int = 500
    regimes: List[str] = field(
        default_factory=lambda: ["homogeneous", "moderate", "extreme"]
    )
    # Subset of ALGO_REGISTRY keys (run_experiments.py) to run; e.g.
    # ["KUBE", "Fractional KUBE"] to skip the baselines. Names must
    # match algo.name exactly (EpsilonFirst's Python float repr drops
    # trailing zeros: eps=0.10 -> "eps-first (eps=0.1)"). UCB1/KUBE/
    # Fractional KUBE default to their est_var bonus mode (variance-
    # aware exploration beats the plain Hoeffding bound in this
    # environment -- see
    # docs/decisions/ucb-exploration-bonus-scale.md); override with
    # the bare name (e.g. "KUBE") or "... (true_var)" to compare.
    algorithms: List[str] = field(
        default_factory=lambda: [
            "Random", "UCB1 (est_var)",
            "eps-first (eps=0.05)", "eps-first (eps=0.1)",
            "eps-first (eps=0.15)",
            "KUBE (est_var)", "Fractional KUBE (est_var)",
        ]
    )
    instance_seed: int = 42
    n_workers: Optional[int] = 1
    results_dir: str = "results"
    # Budget grid swept per regime. Historically hardcoded in
    # run_experiments.py's main() as [4000, 10000]; kept as that same
    # default here so existing configs (which don't set this field)
    # reproduce identical behavior.
    budgets: List[float] = field(default_factory=lambda: [4000, 10000])
