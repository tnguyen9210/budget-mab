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
(`is_feasible`). Reward model (paper Section 5): `mu[i] ~
Uniform[10,20]`, pulls draw from a proper truncated Gaussian
(`Normal(mu[i], sqrt(mu[i]/2))` truncated to `[0, 2*mu[i]]` via
`scipy.stats.truncnorm`, not clipped — clipping would distort the
variance away from the paper's stated `sigma^2`). Rewards are
normalized by `R_MAX=40` so they lie in `[0,1]`.

`arm_var`: the true per-arm reward variance on the normalized scale
(`(mu_raw/2) / R_MAX**2`) — oracle information, used only by
`BonusMode.TRUE_VAR` (below), never by `mu_hat`/`n` bookkeeping.

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

`KUBE`/`FractionalKUBE`/`UCB1` each take a `bonus_mode:
BonusMode` constructor argument (default `BonusMode.NONE`), picking
which variance the UCB exploration bonus is scaled by — see
[decisions/ucb-exploration-bonus-scale.md](decisions/ucb-exploration-bonus-scale.md)
for the full derivation and why the default (worst-case Hoeffding
bound) over-explores in this environment:

| `BonusMode` | Bonus formula | What it needs |
|---|---|---|
| `NONE` (default) | `sqrt(2 ln t / n)` | nothing — worst-case `[0,1]` bound |
| `TRUE_VAR` | `sqrt(2 * arm_var * ln t / n)` | `env.arm_var` (oracle; diagnostic only) |
| `EST_VAR` | `sqrt(2 * var_hat * ln t / n)` | `ArmStats.sample_var()` (online, no oracle) |

`ALGO_REGISTRY` (`run_experiments.py`) instantiates all three modes
for each of `KUBE`/`FractionalKUBE`/`UCB1`; `.name` disambiguates
non-default modes (e.g. `"KUBE (est_var)"`).

All five algorithm classes share the same `run(env) -> total_reward`
interface. `KUBE`/`FractionalKUBE`/`UCB1` share a single loop,
`_run_ucb_loop(env, select_arm, bonus_mode)` — the initial
round-robin phase, the UCB computation, the "find cheapest feasible
arm" fallback, and the stats update all live there exactly once; each
class supplies only its own `select_arm(ucb, residual) -> arm`
callback, its actual point of difference. `ArmStats` (running
mean + Welford variance) replaces the old separate `mu_hat`/`n`
arrays. `EpsilonFirst`'s two-phase structure doesn't fit this shared
loop (see its own section below) and keeps its own `run()`. See
[issues.md#duplicated-run-skeleton](issues.md#duplicated-run-skeleton)
for the history — this used to be three independently-copied loops,
which is how two earlier real bugs (below) diverged silently between
siblings.

### KUBE (Algorithm 1 in the paper)

Each step: compute UCB values for all arms, solve an approximate
unbounded knapsack (`density_ordered_greedy`, sorts arms by
`ucb/cost` descending and greedily fills the residual budget),
then samples one arm from the resulting allocation with probability
proportional to how many "units" of it the knapsack fit. The
knapsack step is itself cost-aware, so by construction it never
proposes an arm the residual budget can't afford — the post-hoc
"arm not affordable" fallback (`cheapest_feasible_arm`, called from
`_run_ucb_loop`) is genuinely defensive here, not a routinely-hit
path.

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
currently-affordable subset before taking the argmax — this is what
makes UCB1's own "unaffordable" fallback essentially unreachable in
normal operation (and the pattern Fractional KUBE's fix now follows
too).

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

- `run_experiments.py` (repo root): reproduces the paper's Figure 1 —
  3 cost regimes (`homogeneous`, `moderate`, `extreme`, via
  `make_costs`), a fixed budget grid (`[4000, 10000]`, hardcoded in
  `main()`), `n_seeds` (default 500) independent noise draws per
  (regime, budget, algo) cell, run in parallel via
  `ProcessPoolExecutor`. Regret is per-seed as
  `oracle_reward() - algo.run(env)`, then mean/SEM'd across seeds.
  Hydra-based (see README): `--config-name <name>` selects a
  `conf/<name>.yaml` preset (schema: `src/exp_config.py:ExpConfig`),
  any field overridable inline (`key=value`), including
  `algorithms=[...]` to select a subset of `ALGO_REGISTRY`'s keys.
  Writes `results/<name>.pkl` plus a `results/<name>.json` provenance
  sidecar (git commit, config used, resolved parameters), and prints
  a summary table (`utils.print_summary`) at the end of the run.
- `plot_results.py` (repo root): loads a `results/<name>.pkl`, plots
  `regret / ln(budget/c_min)` per regime (the paper's normalized
  performance-regret axis), with a
  `O(B^(2/3)/ln B)` reference curve (`utils.reference_curve`) scaled
  to match the `eps-first (eps=0.1)` series.

Both scripts run from the repo root (relative paths like `results`
resolve against it). Library code (`env.py`, `algorithms.py`,
`utils.py`, `exp_config.py`) stays in `src/`.
