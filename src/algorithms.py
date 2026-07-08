"""KUBE, Fractional KUBE, UCB1, and Random for budget-limited MAB.

All algorithms follow the same interface:
    run(env) -> total_reward (float)

Reference: Tran-Thanh et al., "Knapsack based Optimal Policies for
Budget-Limited Multi-Armed Bandits", arXiv:1204.1909.
"""
from typing import Optional

import numpy as np
from env import BudgetMAB


# ---------------------------------------------------------------------------
# Shared helpers (see docs/issues.md#duplicated-run-skeleton)
# ---------------------------------------------------------------------------

def ucb_values(mu_hat: np.ndarray, n: np.ndarray, t: int) -> np.ndarray:
    """UCB index: mu_hat[i] + sqrt(2 ln t / n[i]).

    Arms with n[i]=0 not yet pulled; give them inf UCB so they're
    always preferred until pulled once.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return mu_hat + np.sqrt(2 * np.log(t) / np.where(n > 0, n, np.inf))


def update_mean(mu_hat: np.ndarray, n: np.ndarray, arm: int, reward: float) -> None:
    """Incremental update of the running mean estimate for `arm`."""
    n[arm] += 1
    mu_hat[arm] += (reward - mu_hat[arm]) / n[arm]


def cheapest_feasible_arm(costs: np.ndarray, residual: float) -> Optional[int]:
    """Cheapest arm affordable at `residual`, or None if none are."""
    feasible = np.where(costs <= residual)[0]
    if len(feasible) == 0:
        return None
    return int(feasible[np.argmin(costs[feasible])])


# ---------------------------------------------------------------------------
# Density-ordered greedy (Algorithm 1, Step 9)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# KUBE (Algorithm 1)
# ---------------------------------------------------------------------------

class KUBE:
    """Knapsack-based Upper Confidence Bound Exploration.

    Complexity per step: O(K log K) due to sort in density-ordered greedy.
    """

    name = "KUBE"

    def run(self, env: BudgetMAB) -> float:
        K = env.K
        mu_hat = np.zeros(K)    # empirical mean estimates
        n = np.zeros(K, dtype=int)  # pull counts
        residual = env.budget
        t = 0
        total_reward = 0.0

        while env.is_feasible(residual):
            t += 1

            # Initial phase: pull each arm once (steps 6-7 of Algorithm 1)
            if t <= K:
                arm = t - 1
            else:
                ucb = ucb_values(mu_hat, n, t)

                # Solve approximate knapsack to get M*(B_t)
                m = density_ordered_greedy(ucb, env.costs, residual)
                total_m = m.sum()

                if total_m == 0:
                    # No arm fits even a single pull — shouldn't happen if
                    # is_feasible passed, but guard anyway
                    break

                # Randomly select arm with probability m[i] / sum(m)
                probs = m / total_m
                arm = int(env.rng.choice(K, p=probs))

            if env.costs[arm] > residual:
                # Arm not affordable; skip (shouldn't occur in initial phase
                # if budget >= K * max_cost, but handle gracefully; density_
                # ordered_greedy is itself cost-aware, so this is a defensive
                # fallback, not a routinely-hit path — see docs/issues.md)
                fallback = cheapest_feasible_arm(env.costs, residual)
                if fallback is None:
                    break
                arm = fallback

            reward = env.pull(arm)
            total_reward += reward

            # Update estimates (steps 12-13)
            update_mean(mu_hat, n, arm, reward)
            residual -= env.costs[arm]

        return total_reward


# ---------------------------------------------------------------------------
# Fractional KUBE (Section 3.3)
# ---------------------------------------------------------------------------

class FractionalKUBE:
    """Fractional KUBE: uses fractional relaxation instead of greedy knapsack.

    At each t, the arm with highest UCB density (ucb[i]/c[i]) among
    currently-AFFORDABLE arms is selected deterministically (restricted
    to affordable arms up front, like UCB1, rather than argmax-then-
    fallback — see docs/issues.md#fractional-kube-fallback-discards-ranking
    for why the unrestricted version silently picked the wrong arm near
    budget exhaustion). Complexity per step: O(K).
    """

    name = "Fractional KUBE"

    def run(self, env: BudgetMAB) -> float:
        K = env.K
        mu_hat = np.zeros(K)
        n = np.zeros(K, dtype=int)
        residual = env.budget
        t = 0
        total_reward = 0.0

        while env.is_feasible(residual):
            t += 1

            if t <= K:
                arm = t - 1
            else:
                # Fractional relaxation: pick the highest-ucb-density arm
                # among those currently affordable (restricting BEFORE the
                # argmax preserves the ranking among affordable arms, unlike
                # picking the single best arm over ALL arms and falling back
                # to "cheapest" if it turns out to be unaffordable).
                ucb_density = ucb_values(mu_hat, n, t) / env.costs
                affordable = np.where(env.costs <= residual)[0]
                arm = int(affordable[np.argmax(ucb_density[affordable])])

            if env.costs[arm] > residual:
                fallback = cheapest_feasible_arm(env.costs, residual)
                if fallback is None:
                    break
                arm = fallback

            reward = env.pull(arm)
            total_reward += reward

            update_mean(mu_hat, n, arm, reward)
            residual -= env.costs[arm]

        return total_reward


# ---------------------------------------------------------------------------
# UCB1 adapted to budget setting
# ---------------------------------------------------------------------------

class UCB1:
    """Standard UCB1 ignoring costs (treats all arms as cost-1 for index,
    but still pays actual costs). This is a natural budget-unaware baseline.
    """

    name = "UCB1"

    def run(self, env: BudgetMAB) -> float:
        K = env.K
        mu_hat = np.zeros(K)
        n = np.zeros(K, dtype=int)
        residual = env.budget
        t = 0
        total_reward = 0.0

        while env.is_feasible(residual):
            t += 1

            if t <= K:
                arm = t - 1
            else:
                ucb = ucb_values(mu_hat, n, t)
                # Among arms that are currently affordable
                affordable = np.where(env.costs <= residual)[0]
                arm = int(affordable[np.argmax(ucb[affordable])])

            if env.costs[arm] > residual:
                fallback = cheapest_feasible_arm(env.costs, residual)
                if fallback is None:
                    break
                arm = fallback

            reward = env.pull(arm)
            total_reward += reward

            update_mean(mu_hat, n, arm, reward)
            residual -= env.costs[arm]

        return total_reward


# ---------------------------------------------------------------------------
# Random policy
# ---------------------------------------------------------------------------

class Random:
    """Selects uniformly at random among affordable arms at each step."""

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


# ---------------------------------------------------------------------------
# Epsilon-first baseline (from Tran-Thanh et al. 2010)
# ---------------------------------------------------------------------------

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
        mu_hat = np.zeros(K)
        n = np.zeros(K, dtype=int)
        residual = env.budget
        total_reward = 0.0

        # Exploration phase: round-robin, falling back to the cheapest
        # affordable arm (like the other policies) when the round-robin
        # arm itself is unaffordable, instead of ending exploration early
        # while cheaper arms remain affordable — see
        # docs/issues.md#epsilon-first-exploration-early-break.
        t = 0
        while env.is_feasible(residual) and residual > (1 - self.eps) * env.budget:
            arm = t % K
            if env.costs[arm] > residual:
                fallback = cheapest_feasible_arm(env.costs, residual)
                if fallback is None:
                    break
                arm = fallback
            reward = env.pull(arm)
            total_reward += reward
            update_mean(mu_hat, n, arm, reward)
            residual -= env.costs[arm]
            t += 1

        # Exploitation phase: pull best arm (by density of empirical mean)
        if n.sum() > 0:
            # Use density = mu_hat / cost, consistent with budget-limited framing
            density = np.where(n > 0, mu_hat / env.costs, 0.0)
            best_arm = int(np.argmax(density))
            while env.is_feasible(residual) and env.costs[best_arm] <= residual:
                reward = env.pull(best_arm)
                total_reward += reward
                residual -= env.costs[best_arm]

        return total_reward
