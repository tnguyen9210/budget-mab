"""Budget-limited MAB environment (Tran-Thanh et al. 2012)."""
import numpy as np
from scipy.stats import truncnorm

R_MAX = 40.0  # 2 * max(mu) = 2*20; normalises rewards to [0,1]


class BudgetMAB:
    """K-armed bandit with heterogeneous arm costs and a fixed budget.

    Reward model (paper Section 5):
      mu[i] ~ Uniform[10,20],
      reward ~ TruncGaussian(mu[i], sqrt(mu[i]/2)),
      support [0, 2*mu[i]].

    The underlying (pre-truncation) normal has mean mu[i] and
    sigma=sqrt(mu[i]/2); reward is drawn from that normal truncated to
    [0, 2*mu[i]] via inverse-CDF sampling (scipy.stats.truncnorm), not
    clipped -- clipping would pile probability mass onto the boundary
    and distort the variance away from the paper's stated sigma^2.

    Internally all rewards are divided by R_MAX=40 so they lie in [0,1],
    making the Hoeffding-based UCB bound sqrt(2 ln t / n) valid.

    Oracle and algorithm rewards are BOTH on the normalised [0,1] scale,
    so regret = oracle_norm - algo_reward_norm is consistent.
    """

    def __init__(
        self,
        mu: np.ndarray,       # raw means in [10, 20]
        costs: np.ndarray,    # integer costs >= 1
        budget: float,
        rng: np.random.Generator,
    ):
        assert mu.shape == costs.shape
        assert np.all(costs >= 1)
        self.mu_raw = mu.copy()
        self.mu = mu / R_MAX          # normalised means in [0.25, 0.5]
        self.costs = costs.copy()
        self.budget = budget
        self.rng = rng
        self.K = len(mu)

    def pull(self, arm: int) -> float:
        """Return normalised reward in [0, 1]."""
        mu_i = self.mu_raw[arm]
        sigma = np.sqrt(mu_i / 2.0)
        lo, hi = 0.0, 2.0 * mu_i
        a, b = (lo - mu_i) / sigma, (hi - mu_i) / sigma
        raw = truncnorm.rvs(
            a, b, loc=mu_i, scale=sigma, random_state=self.rng,
        )
        return float(raw) / R_MAX

    @property
    def arm_var(self) -> np.ndarray:
        """True per-arm reward variance, normalised (Section 5) scale.

        Oracle information -- only the BonusMode.TRUE_VAR bandit
        algorithms read this; it is never used to compute mu_hat/n.
        """
        return (self.mu_raw / 2.0) / (R_MAX ** 2)

    def is_feasible(self, residual: float) -> bool:
        return residual >= self.costs.min()

    def oracle_reward(self) -> float:
        """Expected total normalised reward of the optimal policy.

        The optimal (full-information) policy pulls
        I* = argmax mu_norm[i]/c[i] as many times as the budget
        allows.  Both oracle and algo rewards are on the same
        [0,1] scale.
        """
        I_star = int(np.argmax(self.mu / self.costs))
        n_pulls = int(self.budget // self.costs[I_star])
        return n_pulls * self.mu[I_star]
