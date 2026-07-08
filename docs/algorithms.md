# Algorithm registry

Map of the policies implemented in `src/algorithms.py`, what each
does, and where the paper-vs-code correspondence lives. This file is
an index, not the spec — the paper is
[`paper.pdf`](../paper.pdf) (Tran-Thanh, Chapman, Rogers, Jennings,
"Knapsack based Optimal Policies for Budget-Limited Multi-Armed
Bandits", arXiv:1204.1909).

Known issues in the implementations below are tracked in
[issues.md](issues.md), not duplicated here.

## Problem setting (`src/env.py`)

`BudgetMAB`: K arms, each with a fixed unknown mean reward `mu[i]` and
a fixed known cost `costs[i]`. An agent pulls arms sequentially; each
pull consumes `costs[arm]` from a fixed `budget`. The agent stops when
remaining budget is less than the cheapest arm's cost
(`is_feasible`). Reward model: `mu[i] ~ Uniform[10,20]`, pulls draw
from `Normal(mu[i], sqrt(mu[i]/2))` clipped to `[0, 2*mu[i]]`
(docstring calls this "TruncGaussian" — see
[issues.md#env-truncated-gaussian](issues.md#env-truncated-gaussian)
for the clip-vs-truncate distinction). Rewards are normalized by
`R_MAX=40` so they lie in `[0,1]`ish, matching the Hoeffding-style UCB
bound `sqrt(2 ln t / n)` used throughout.

`oracle_reward()`: the expected total reward of the best fixed policy
— pull the single highest-density arm (`argmax(mu/costs)`) as many
times as the budget allows. This is what every algorithm's regret is
measured against (`regret = oracle_reward() - algo.run(env)`).

## Active variants (`src/algorithms.py`)

| Algorithm | Class | Selection rule | Per-step cost |
|---|---|---|---|
| KUBE | `KUBE` | Approximate unbounded knapsack over UCB values (`density_ordered_greedy`), then samples an arm with probability proportional to its knapsack allocation | O(K log K) |
| Fractional KUBE | `FractionalKUBE` | Deterministic argmax of UCB-density (`ucb/cost`) over all arms | O(K) |
| UCB1 (budget-aware) | `UCB1` | Standard UCB1 index, restricted to currently-affordable arms | O(K) |
| Random | `Random` | Uniform over affordable arms | O(K) |
| Epsilon-first | `EpsilonFirst` | Round-robin explore for `eps*budget`, then exploit the best density-by-empirical-mean arm for the rest | O(K) |

All five share the same `run(env) -> total_reward` interface
(`src/algorithms.py:1-8`) and follow the same overall shape: an
initial round-robin phase (pull each arm once), then a selection rule
that runs until the budget is exhausted. The UCB formula, the
"find cheapest feasible arm" fallback, and the incremental mean
update are factored into shared helpers
(`ucb_values`, `cheapest_feasible_arm`, `update_mean`,
`src/algorithms.py:19-40`) used by all four UCB-based classes — see
[issues.md#duplicated-run-skeleton](issues.md#duplicated-run-skeleton)
for what's still duplicated (each class's own `while` loop/phase
structure) and why.

### KUBE (Algorithm 1 in the paper)

Each step: compute UCB values for all arms, solve an approximate
unbounded knapsack (`density_ordered_greedy`, sorts arms by
`ucb/cost` descending and greedily fills the residual budget),
then samples one arm from the resulting allocation with probability
proportional to how many "units" of it the knapsack fit. The
knapsack step is itself cost-aware, so by construction it never
proposes an arm the residual budget can't afford — the post-hoc
"arm not affordable" fallback (`cheapest_feasible_arm`, called from
`src/algorithms.py:123`) is genuinely defensive here, not a
routinely-hit path.

### Fractional KUBE (paper §3.3)

The cheaper O(K) relaxation of KUBE: instead of solving the knapsack,
deterministically pick the arm with the highest UCB-density **among
currently-affordable arms** (restricted before the argmax, matching
UCB1's pattern — see
[issues.md#fractional-kube-fallback-discards-ranking](issues.md#fractional-kube-fallback-discards-ranking)
for the earlier version that argmaxed over all arms first and why
that discarded ranking information near budget exhaustion).

### UCB1 (budget-unaware baseline, budget-aware selection)

Standard UCB1 index, but arm selection is restricted to the
currently-affordable subset before taking the argmax
(`src/algorithms.py:221`) — this is what makes UCB1's own
"unaffordable" fallback essentially unreachable in normal operation
(and the pattern Fractional KUBE's fix now follows too).

### Random

Uniform selection among currently-affordable arms. The simplest
possible baseline; no UCB bookkeeping.

### Epsilon-first (Tran-Thanh et al. 2010, cited as a baseline)

Two phases: round-robin exploration until `eps*budget` is spent,
then exploit the empirically-best arm (by `mu_hat/cost` density) for
the remaining budget. This is structurally different from the other
four (two phases instead of one selection rule run to exhaustion).
The exploration phase falls back to the cheapest affordable arm when
the round-robin arm is unaffordable, same as the other three — see
[issues.md#epsilon-first-exploration-early-break](issues.md#epsilon-first-exploration-early-break)
for the earlier version that broke out of exploration early instead.

## Experiment pipeline

- `src/run_experiments.py`: reproduces the paper's Figure 1 — 3 cost
  regimes (`homogeneous`, `moderate`, `extreme`, via `make_costs`),
  a budget sweep (`np.linspace(500, 4000, 15)` by default), `n_seeds`
  (default 500) independent noise draws per (regime, budget, algo)
  cell, run in parallel via `ProcessPoolExecutor`. Regret is per-seed
  as `oracle_reward() - algo.run(env)`, then mean/SEM'd across seeds.
- `src/plot_results.py`: loads `results/results.pkl`, plots
  `regret / ln(budget/c_min)` per regime (the paper's normalized
  performance-regret axis), with a hardcoded
  `O(B^(2/3)/ln B)` reference curve scaled to match the
  `eps-first (eps=0.1)` series.

Both scripts assume they're run from `src/` (relative paths like
`../results` resolve to the repo-root `results/` dir).
