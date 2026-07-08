"""KUBE, Fractional KUBE, UCB1, and Random for budget-limited MAB.

All algorithms follow the same interface:
    run(env) -> total_reward (float)

Reference: Tran-Thanh et al., "Knapsack based Optimal Policies for
Budget-Limited Multi-Armed Bandits", arXiv:1204.1909.
"""
from enum import Enum
from typing import Callable, Optional

import numpy as np
from env import BudgetMAB


# ----------------------------------------------------------------------
# Running per-arm statistics (mean + variance), shared by every
# UCB-based policy (see docs/issues.md#duplicated-run-skeleton)
# ----------------------------------------------------------------------

class ArmStats:
    """Running mean and (Welford) variance estimate, per arm."""

    def __init__(self, K: int):
        self.n = np.zeros(K, dtype=int)
        self.mean = np.zeros(K)
        self._m2 = np.zeros(K)   # sum of squared deviations

    def update(self, arm: int, reward: float) -> None:
        self.n[arm] += 1
        delta = reward - self.mean[arm]
        self.mean[arm] += delta / self.n[arm]
        self._m2[arm] += delta * (reward - self.mean[arm])

    def sample_var(self) -> np.ndarray:
        """Unbiased variance estimate; 0 where n<2 (too few samples)."""
        return np.where(self.n > 1, self._m2 / np.maximum(self.n - 1, 1), 0.0)


# ----------------------------------------------------------------------
# Exploration-bonus modes (see docs/decisions/ucb-exploration-bonus-scale.md)
# ----------------------------------------------------------------------

class BonusMode(str, Enum):
    """Which variance the UCB confidence radius is scaled by.

    NONE:     worst-case Hoeffding bound for [0,1]-bounded rewards
              (sqrt(2 ln t / n)) -- ignores the arms' actual variance.
    TRUE_VAR: oracle -- plug in env.arm_var directly. Not something a
              real algorithm could know; useful as a diagnostic upper
              bound on how well variance-aware exploration could do.
    EST_VAR:  online empirical-Bernstein -- use each arm's running
              sample variance, estimated from collected rewards.
    """
    NONE = "none"
    TRUE_VAR = "true_var"
    EST_VAR = "est_var"


def _bonus_none(stats: ArmStats, t: int, arm_var=None) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(stats.n > 0, np.sqrt(2 * np.log(t) / stats.n), np.inf)


def _bonus_true_var(
    stats: ArmStats, t: int, arm_var: np.ndarray,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        bonus = np.sqrt(2 * arm_var * np.log(t) / stats.n)
        return np.where(stats.n > 0, bonus, np.inf)


def _bonus_est_var(stats: ArmStats, t: int, arm_var=None) -> np.ndarray:
    """Variance-scaled bound using each arm's running sample variance.

    Deliberately NOT the textbook empirical-Bernstein bound (which
    adds a "+ 3 ln(t) / n" small-sample bias-correction term) -- that
    term is O(1/n) with a constant far larger than the O(sqrt(1/n))
    variance term at the n this environment actually reaches (n in
    the tens-to-hundreds per arm at these budgets), so it dominated
    and made this WORSE than BonusMode.NONE in practice (regret 106
    vs 78 for KUBE at B=4000). Dropping it and using the bare
    variance-scaled sqrt term (same functional form as TRUE_VAR, with
    the arm's own running estimate in place of the oracle variance)
    fixed it: regret 9.8, between NONE and the TRUE_VAR oracle bound,
    as expected. See docs/decisions/ucb-exploration-bonus-scale.md.
    """
    var_hat = stats.sample_var()
    with np.errstate(divide="ignore", invalid="ignore"):
        bonus = np.sqrt(2 * var_hat * np.log(t) / stats.n)
        return np.where(stats.n > 0, bonus, np.inf)


BONUS_FNS: dict = {
    BonusMode.NONE: _bonus_none,
    BonusMode.TRUE_VAR: _bonus_true_var,
    BonusMode.EST_VAR: _bonus_est_var,
}


def ucb_values(
    stats: ArmStats, t: int, bonus_mode: BonusMode, arm_var=None,
) -> np.ndarray:
    """UCB index: mean[i] + confidence_bonus[i] (mode picks the bonus).

    Arms with n[i]=0 not yet pulled get an inf bonus so they're always
    preferred until pulled once.
    """
    return stats.mean + BONUS_FNS[bonus_mode](stats, t, arm_var)


def _bonus_mode_name(base: str, bonus_mode: BonusMode) -> str:
    """e.g. ("KUBE", NONE) -> "KUBE"; ("KUBE", EST_VAR) -> "KUBE (est_var)"."""
    if bonus_mode == BonusMode.NONE:
        return base
    return f"{base} ({bonus_mode.value})"


def cheapest_feasible_arm(costs: np.ndarray, residual: float) -> Optional[int]:
    """Cheapest arm affordable at `residual`, or None if none are."""
    feasible = np.where(costs <= residual)[0]
    if len(feasible) == 0:
        return None
    return int(feasible[np.argmin(costs[feasible])])


# ----------------------------------------------------------------------
# Density-ordered greedy (Algorithm 1, Step 9)
# ----------------------------------------------------------------------

def density_ordered_greedy(
    ucb_vals: np.ndarray,
    costs: np.ndarray,
    residual_budget: float,
) -> np.ndarray:
    """Approximate unbounded knapsack via density-ordered greedy.

    Items are sorted by density = ucb_val / cost (descending).
    We greedily fill the knapsack with as many units of the highest-
    density item as fit, then move to the next, etc.

    Returns m[i]: how many times arm i appears in the solution set M*.
    """
    K = len(ucb_vals)
    density = ucb_vals / costs          # shape (K,)
    order = np.argsort(-density)        # descending

    m = np.zeros(K, dtype=int)
    rem = residual_budget
    for i in order:
        if costs[i] <= rem:
            units = int(rem // costs[i])
            m[i] = units
            rem -= units * costs[i]
        if rem < costs.min():
            break
    return m


# ----------------------------------------------------------------------
# Shared UCB-policy run loop
# ----------------------------------------------------------------------
# KUBE, FractionalKUBE, and UCB1 differ ONLY in how they turn a vector
# of UCB values into a single arm choice (`select_arm`); everything
# else -- the round-robin init phase, the affordability fallback, the
# stats update, the stopping condition -- is identical. See
# docs/issues.md#duplicated-run-skeleton for why this used to be
# copy-pasted three times (and how that let two real bugs diverge
# silently between siblings).

SelectArmFn = Callable[[np.ndarray, float], int]


def _run_ucb_loop(
    env: BudgetMAB,
    select_arm: SelectArmFn,
    bonus_mode: BonusMode,
) -> float:
    """Run the shared KUBE/FractionalKUBE/UCB1 skeleton to exhaustion.

    select_arm(ucb, residual) -> arm index, called only after the
    initial round-robin phase (t > K); it may assume `ucb` already
    reflects the current bonus_mode.
    """
    K = env.K
    stats = ArmStats(K)
    arm_var = env.arm_var if bonus_mode == BonusMode.TRUE_VAR else None
    residual = env.budget
    t = 0
    total_reward = 0.0

    while env.is_feasible(residual):
        t += 1

        if t <= K:
            arm = t - 1
        else:
            ucb = ucb_values(stats, t, bonus_mode, arm_var)
            arm = select_arm(ucb, residual)
            if arm is None:
                break

        if env.costs[arm] > residual:
            fallback = cheapest_feasible_arm(env.costs, residual)
            if fallback is None:
                break
            arm = fallback

        reward = env.pull(arm)
        total_reward += reward
        stats.update(arm, reward)
        residual -= env.costs[arm]

    return total_reward


# ----------------------------------------------------------------------
# KUBE (Algorithm 1)
# ----------------------------------------------------------------------

class KUBE:
    """Knapsack-based Upper Confidence Bound Exploration.

    Complexity per step: O(K log K) due to the sort in
    density-ordered greedy.
    """

    def __init__(self, bonus_mode: BonusMode = BonusMode.NONE):
        self.bonus_mode = BonusMode(bonus_mode)
        self.name = _bonus_mode_name("KUBE", self.bonus_mode)

    def run(self, env: BudgetMAB) -> float:
        def select_arm(ucb, residual):
            # Solve approximate knapsack to get M*(B_t)
            m = density_ordered_greedy(ucb, env.costs, residual)
            if m.sum() == 0:
                # No arm fits even a single pull -- shouldn't happen
                # if is_feasible passed, but guard anyway
                return None
            # Randomly select arm with probability m[i] / sum(m)
            probs = m / m.sum()
            return int(env.rng.choice(env.K, p=probs))

        return _run_ucb_loop(env, select_arm, self.bonus_mode)


# ----------------------------------------------------------------------
# Fractional KUBE (Section 3.3)
# ----------------------------------------------------------------------

class FractionalKUBE:
    """Fractional KUBE: fractional relaxation, not greedy knapsack.

    At each t, the arm with highest UCB density (ucb[i]/c[i]) among
    currently-AFFORDABLE arms is selected deterministically
    (restricted to affordable arms up front, like UCB1, rather than
    argmax-then-fallback -- see
    docs/issues.md#fractional-kube-fallback-discards-ranking for why
    the unrestricted version silently picked the wrong arm near
    budget exhaustion). Complexity per step: O(K).
    """

    def __init__(self, bonus_mode: BonusMode = BonusMode.NONE):
        self.bonus_mode = BonusMode(bonus_mode)
        self.name = _bonus_mode_name("Fractional KUBE", self.bonus_mode)

    def run(self, env: BudgetMAB) -> float:
        def select_arm(ucb, residual):
            # Restricting BEFORE the argmax preserves the ranking
            # among affordable arms, unlike picking the single best
            # arm over ALL arms and falling back to "cheapest" if it
            # turns out to be unaffordable.
            ucb_density = ucb / env.costs
            affordable = np.where(env.costs <= residual)[0]
            return int(affordable[np.argmax(ucb_density[affordable])])

        return _run_ucb_loop(env, select_arm, self.bonus_mode)


# ----------------------------------------------------------------------
# UCB1 adapted to budget setting
# ----------------------------------------------------------------------

class UCB1:
    """Standard UCB1 ignoring costs.

    Treats all arms as cost-1 for the index, but still pays actual
    costs. This is a natural budget-unaware baseline.
    """

    def __init__(self, bonus_mode: BonusMode = BonusMode.NONE):
        self.bonus_mode = BonusMode(bonus_mode)
        self.name = _bonus_mode_name("UCB1", self.bonus_mode)

    def run(self, env: BudgetMAB) -> float:
        def select_arm(ucb, residual):
            affordable = np.where(env.costs <= residual)[0]
            return int(affordable[np.argmax(ucb[affordable])])

        return _run_ucb_loop(env, select_arm, self.bonus_mode)


# ----------------------------------------------------------------------
# Random policy
# ----------------------------------------------------------------------

class Random:
    """Uniformly random selection among affordable arms each step."""

    name = "Random"

    def run(self, env: BudgetMAB) -> float:
        residual = env.budget
        total_reward = 0.0

        while env.is_feasible(residual):
            affordable = np.where(env.costs <= residual)[0]
            if len(affordable) == 0:
                break
            arm = int(env.rng.choice(affordable))
            reward = env.pull(arm)
            total_reward += reward
            residual -= env.costs[arm]

        return total_reward


# ----------------------------------------------------------------------
# Epsilon-first baseline (from Tran-Thanh et al. 2010)
# ----------------------------------------------------------------------

class EpsilonFirst:
    """Budget-limited epsilon-first policy.

    Spends eps*B on pure exploration (round-robin), then exploits the
    empirically best arm for the remaining (1-eps)*B budget.
    """

    def __init__(self, eps: float):
        self.eps = eps
        self.name = f"eps-first (eps={eps})"

    def run(self, env: BudgetMAB) -> float:
        K = env.K
        stats = ArmStats(K)
        residual = env.budget
        total_reward = 0.0

        # Exploration phase: round-robin, falling back to the cheapest
        # affordable arm (like the other policies) when the
        # round-robin arm itself is unaffordable, instead of ending
        # exploration early while cheaper arms remain affordable -- see
        # docs/issues.md#epsilon-first-exploration-early-break.
        t = 0
        explore_floor = (1 - self.eps) * env.budget
        while env.is_feasible(residual) and residual > explore_floor:
            arm = t % K
            if env.costs[arm] > residual:
                fallback = cheapest_feasible_arm(env.costs, residual)
                if fallback is None:
                    break
                arm = fallback
            reward = env.pull(arm)
            total_reward += reward
            stats.update(arm, reward)
            residual -= env.costs[arm]
            t += 1

        # Exploitation phase: pull the best arm by density of the
        # empirical mean (mean / cost, consistent with the
        # budget-limited framing).
        if stats.n.sum() > 0:
            density = np.where(stats.n > 0, stats.mean / env.costs, 0.0)
            best_arm = int(np.argmax(density))
            while (env.is_feasible(residual)
                   and env.costs[best_arm] <= residual):
                reward = env.pull(best_arm)
                total_reward += reward
                residual -= env.costs[best_arm]

        return total_reward
