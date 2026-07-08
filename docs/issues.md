# Known issues

Findings from a code review of the initial commit (`4acbcf4`),
covering `src/algorithms.py`, `src/env.py`, `src/run_experiments.py`,
`src/plot_results.py`. Each entry: what's wrong, a concrete
failure scenario, verification status, and open questions to
resolve before fixing.

Status legend: **confirmed** (verified against the code with a
concrete reachable scenario) / **plausible** (real defect, but
severity or reachability not fully pinned down).

---

## Epsilon-first exploration early break {#epsilon-first-exploration-early-break}

**Status:** confirmed · **File:** `src/algorithms.py:258` ·
**Severity:** correctness (affects regret comparisons involving
`EpsilonFirst`)

`EpsilonFirst.run()`'s exploration-phase loop breaks unconditionally
when the round-robin arm is unaffordable:

```python
arm = t % K
if env.costs[arm] > residual:
    break
```

Every other algorithm (`KUBE`, `FractionalKUBE`, `UCB1`) falls back to
the cheapest currently-affordable arm instead of aborting when its
selection is unaffordable (e.g. `src/algorithms.py:91-98`). Epsilon-
first is the only one of the four that doesn't.

**Concrete failure scenario** (verified by trace): `costs=[1,1,50]`,
`budget=200`, `eps=0.9` (explore boundary: `residual > 20`). Round-
robin cycles arm0/arm1/arm2; by `t=11`, `residual=42` and the cycle
lands on the cost-50 arm — the loop breaks immediately, even though
`residual=42` is still well above the boundary (20) and arms 0/1
(cost 1 each) remain trivially affordable. Exploration stops dozens
of steps early while the other three algorithms would substitute a
cheap arm and keep exploring.

A narrower, more severe sub-case: if the *very first* round-robin arm
(`t=0`, `arm=0`) is unaffordable relative to the **full** budget
(`costs[0] > budget`), `n.sum()` stays 0 after the loop, the
`if n.sum() > 0:` exploitation guard is never entered, and the
algorithm returns `total_reward=0`, abandoning the entire budget. This
sub-case is **not reachable** in this repo's actual experiment grid
(`make_costs`'s max cost is 20 in the "extreme" regime; the budget
grid's minimum is 500 — 20 ≪ 500), so it's a real latent bug but not
one that's silently corrupted any current results.

**Open question:** what's the intended fix? Two options with
different implications:
1. Give `EpsilonFirst`'s exploration phase the same
   "fall back to cheapest affordable arm" pattern the other three
   already have (simplest, most consistent with the rest of the
   codebase).
2. Something more faithful to the *specific* eps-first baseline
   definition — worth checking whether the cited "Tran-Thanh et al.
   2010" eps-first source specifies its own affordability handling,
   since this baseline's exact behavior may matter for how it's
   meant to compare against KUBE/Fractional KUBE in Figure 1.

---

## Fractional KUBE's fallback discards its own ranking {#fractional-kube-fallback-discards-ranking}

**Status:** confirmed, scoped to `FractionalKUBE` only (does not
apply to `KUBE`) · **File:** `src/algorithms.py:141` ·
**Severity:** correctness (degrades `FractionalKUBE`'s selection
quality near budget exhaustion)

`UCB1` restricts its argmax to affordable arms *before* selecting:

```python
affordable = np.where(env.costs <= residual)[0]
arm = int(affordable[np.argmax(ucb[affordable])])
```

`FractionalKUBE` does not — it argmaxes UCB-density over **all** K
arms, then only checks affordability after the fact:

```python
ucb_density = (...) / env.costs
arm = int(np.argmax(ucb_density))   # unrestricted
# ... later, if unaffordable:
feasible = np.where(env.costs <= residual)[0]
arm = int(feasible[np.argmin(env.costs[feasible])])   # cheapest, not best-density
```

When the single best-density arm is unaffordable, the fallback picks
the **cheapest** feasible arm, not the *best-density-among-affordable*
arm — silently discarding the UCB ranking for every other candidate.

**Concrete failure scenario:** `costs = [1]*9 + [15]`, `K=10`,
`residual=5`. If the cost-15 arm has accumulated the highest UCB
density (e.g. pulled least, so it has the largest exploration bonus),
it gets selected, found unaffordable, and the fallback picks the
lowest-index cost-1 arm by tie-break — ignoring which of the 9
affordable arms actually has the highest density among them. This
fires routinely near budget exhaustion, not just in rare edge cases.

**Why KUBE is unaffected:** `KUBE`'s own selection mechanism
(`density_ordered_greedy`) is passed `residual` directly and only
ever allocates knapsack mass to arms that fit within it — so by the
time `KUBE.run()`'s own affordability guard could fire, it's already
a "shouldn't occur" defensive check (as the code's own comment
says), not a routinely-hit path the way `FractionalKUBE`'s is.

**Open question:** should the fix restrict `FractionalKUBE`'s argmax
to affordable arms first (matching `UCB1`'s pattern exactly), or
should the fallback itself be smarter (pick highest-density among
feasible, not cheapest)? The former is more consistent with the
paper's own fractional-relaxation framing (§3.3) — worth checking
against the paper's exact algorithm statement before changing.

---

## Duplicated `run()` skeleton across KUBE/FractionalKUBE/UCB1 {#duplicated-run-skeleton}

**Status:** confirmed · **File:** `src/algorithms.py:58-203` ·
**Severity:** maintainability (very likely the root cause of the two
bugs above)

`KUBE.run()`, `FractionalKUBE.run()`, and `UCB1.run()` each
re-implement, nearly identically:
- the initial round-robin phase (`if t <= K: arm = t - 1`),
- the `np.errstate`-guarded UCB formula,
- the "find cheapest feasible arm" affordability fallback,
- the incremental mean update
  (`mu_hat[arm] += (reward - mu_hat[arm]) / n[arm]`).

Only the arm-selection rule genuinely differs between the three
classes. `EpsilonFirst` duplicates the mean-update logic too, though
its two-phase structure is different enough that it's not a clean
fit for the same base class without more thought.

**Why this matters beyond style:** both confirmed bugs above are
exactly the kind of divergence this duplication invites — one
sibling's fallback/guard logic silently drifting from the others'
because there's no single place enforcing they stay consistent. A
shared base class (e.g. a `_BaseUCBPolicy` with a
`select_arm(t, mu_hat, n, residual, env)` hook, and shared
`_cheapest_feasible_arm` / `_ucb_values` / `_update_mean` helpers)
would mean the affordability-fallback bug, at least, could only be
introduced once, not independently in each subclass.

**Open question:** worth doing this refactor before or after fixing
the two bugs above? Doing it first would let both fixes land in one
shared place rather than three; doing it after means the current
(buggy) behavior is easier to diff against for a "did the fix change
anything else" sanity check. Lean toward refactor-first, but flagging
for a decision rather than assuming.

---

## Wasted `ProcessPoolExecutor` / `oracle_reward()` recomputation

**Status:** confirmed · **File:** `src/run_experiments.py:69,131` ·
**Severity:** efficiency only (no effect on correctness/results)

`run_experiment()` creates a new `ProcessPoolExecutor` inside the
innermost `(regime, budget, algo)` loop — 3×15×7 = 315 times total —
instead of once for the whole sweep. Each creation spawns up to
`os.cpu_count()` fresh worker processes, each paying interpreter
startup and module re-import cost.

Separately, `env.oracle_reward()` depends only on `(mu, costs,
budget)` — not on the per-seed RNG — but is recomputed inside
`_run_one` for every one of the 500 (default `n_seeds`) seed calls
per `(regime, budget, algo)` cell, i.e. ~52,500 redundant
recomputations per regime when only 15 distinct values (one per
budget) are actually needed.

**Impact:** pure wasted wall-clock, no effect on any reported number.
Worth fixing if experiment turnaround time matters, otherwise low
priority.

**Suggested fix:** hoist the `ProcessPoolExecutor` creation outside
all three loops (create once, submit all jobs across the whole
sweep, or at least once per regime); compute `oracle_reward()` once
per `(regime, budget)` outside the seed loop and pass the scalar into
`_run_one`, or subtract it after aggregating.

---

## Fragile silent fallback in the reference-curve plot

**Status:** plausible · **File:** `src/plot_results.py:47` ·
**Severity:** low (affects plot correctness only if the algorithm
list changes)

The `O(B^(2/3)/ln B)` reference curve is scaled against a hardcoded
key:

```python
eps_y = np.array(algo_data.get(
    "eps-first (eps=0.1)", next(iter(algo_data.values()))
)["regret"]) / log_norm
```

If `"eps-first (eps=0.1)"` is ever missing (e.g. `EpsilonFirst(0.10)`
dropped from `run_experiments.py`'s algorithm list, or a partial
`results.pkl` is loaded), this silently falls back to whatever
algorithm happens to be first in dict-insertion order (currently
`KUBE`) — rescaling the reference line against a semantically
different algorithm's regret magnitude with no warning. Currently
harmless since the key always exists in the full pipeline, but a
silent-wrong-plot risk if the algorithm list ever changes without a
matching update here.

**Suggested fix:** either raise/warn if the expected key is missing,
or make the reference-curve anchor an explicit, named parameter
rather than an implicit dict-order fallback.

---

## `env.py` docstring says "TruncGaussian" but implements clipping {#env-truncated-gaussian}

**Status:** plausible, low practical impact · **File:** `src/env.py:41`

```python
raw = self.rng.normal(self.mu_raw[arm], sigma)
raw = float(np.clip(raw, 0.0, 2.0 * self.mu_raw[arm]))
```

This clips samples to `[0, 2*mu]`; it doesn't resample or use
inverse-CDF sampling, so it isn't a properly renormalized truncated
Gaussian (a true truncated Gaussian's mean would still equal `mu_raw`
exactly; a clipped Gaussian's mean is only approximately `mu_raw`,
biased by however much mass piles up at the two boundaries).

**Quantified impact:** across the full `mu_raw ∈ [10, 20]` range used
here, the clip boundaries sit at least 4.47σ from the mean (at
`mu_raw=10`, `sigma=sqrt(5)≈2.236`) up to 6.32σ (at `mu_raw=20`,
`sigma=sqrt(10)≈3.162`). `P(clip)` ranges from ~8e-6 down to ~5e-10 —
negligible in practice; `oracle_reward()`'s assumption that the true
mean equals `mu_raw/R_MAX` is only *approximately*, not *exactly*,
correct, but the gap is far below the noise floor of any experiment
this codebase runs (`n_seeds=500`).

**Suggested fix:** cosmetic only — rename the docstring's
"TruncGaussian" to "clipped Gaussian" (or actually implement rejection
sampling / inverse-CDF truncation if exact-mean matching to `mu_raw`
is ever load-bearing for a theoretical claim being verified
empirically). Not urgent.

---

## Ruled out during review (no fix needed)

**`np.linspace(500, 4000, 15, dtype=int)` budget-grid truncation** —
initially flagged as a risk that `dtype=int` casting could collapse
adjacent budget breakpoints into duplicates. Verified numerically:
`(4000-500)/14 = 250.0` exactly, so all 15 default breakpoints land
on exact integers before casting — no duplicates, no precision loss.
A non-default `--n_budgets` override that doesn't evenly divide 3500
could theoretically hit this (e.g. `--n_budgets 10` gives a
non-integer step), but even the one non-evenly-dividing case checked
during review didn't actually produce a collision. Not a real bug in
any usage this codebase currently has.
