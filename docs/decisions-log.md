# Design decisions log

Append-only chronological record of decisions git history can't
show: cross-cutting design choices that span multiple files, and
deliberate omissions — things chosen *not* to be built, and why.
Newest first. One `##` section per decision. Titles carry one or two
area prefixes (`Area:` or `Area, Area:`) so skimming groups by eye
and `grep '^## .*Area'` gives a per-topic view.

Every decision gets an entry here, always — this file is the
chronological spine. When a decision is substantial enough to need a
table, multiple named alternatives, or an open still-unresolved
scaffold, it also gets a standalone file in
[decisions/](decisions/); the log entry then carries a one-line
pointer to it rather than repeating the full writeup.

## 2026-07-08 — Algorithms: three selectable UCB bonus modes
(`BonusMode`) implemented; shared KUBE/FractionalKUBE/UCB1 run loop
extracted; a second empirical-Bernstein bias-term bug found and fixed

**Context:** follow-up to the entry directly below, which diagnosed
(but did not yet fix) that the UCB bonus over-explores relative to
this environment's true reward variance. Decided against hardcoding
a single `/40` constant (environment-specific, doesn't generalize) in
favor of making the exploration bonus's variance source a first-class,
selectable axis, so `UCB1`/`KUBE`/`FractionalKUBE` can each be run and
compared under three modes.

**Decision:** add `BonusMode` (`NONE` / `TRUE_VAR` / `EST_VAR`) and a
`BONUS_FNS` registry in `src/algorithms.py`; each of
`UCB1`/`KUBE`/`FractionalKUBE` takes `bonus_mode` as a constructor
arg (default `NONE`, preserving prior behavior/names). `TRUE_VAR`
reads a new `env.arm_var` property (the paper's known
`sigma_i=sqrt(mu_i/2)`, oracle information); `EST_VAR` uses a new
`ArmStats` class (Welford mean+variance) to estimate each arm's
variance online from collected rewards. `ALGO_REGISTRY`
(`run_experiments.py`) now includes all three modes for each of the
three algorithms (9 entries) alongside the unaffected `Random`/
`eps-first` variants.

**Also refactored (same change, not a separate decision):** collapsed
`KUBE.run()`/`FractionalKUBE.run()`/`UCB1.run()` — previously three
near-identical copies of the same loop, flagged as a duplication risk
in [issues.md#duplicated-run-skeleton](../issues.md#duplicated-run-skeleton)
and the likely root cause of two earlier real bugs — into one shared
`_run_ucb_loop(env, select_arm, bonus_mode)`, with each class now
supplying only its `select_arm` callback (its actual point of
difference). Verified byte-identical regret to the pre-refactor
per-class loops under `BonusMode.NONE`, same seeds, confirming the
extraction is behavior-preserving.

**A second bug, found while validating `EST_VAR`:** the standard
empirical-Bernstein bound (`sqrt(2*var_hat*ln(t)/n) + 3*ln(t)/n`)
made `EST_VAR` perform worse than even the un-fixed `NONE` bonus (KUBE
regret 105.8 vs 77.9 at B=4000) — the `+3 ln(t)/n` bias-correction
term is `O(1/n)` vs the variance term's `O(sqrt(1/n))`, and at this
environment's actual per-arm sample counts (tens to low hundreds) the
bias term is 4–10x larger, drowning out the variance-awareness
entirely. **Fix:** drop the bias term; `EST_VAR` is now the bare
`sqrt(2*var_hat*ln(t)/n)`. Verified this restores the expected
ordering at every tested budget (KUBE at B=4000: `NONE` 78.2,
`EST_VAR` 9.8, `TRUE_VAR` (oracle) 5.1).

**Revisit if:** the `n=1`-gives-zero-variance edge case (same shape
as the `inf`-on-unpulled bug, one step later) is ever observed to
cause real arm starvation — not seen in any tested budget so far, so
not fixed defensively. Full writeup, all four ablation ranges, and
the empirical tables:
[decisions/ucb-exploration-bonus-scale.md](decisions/ucb-exploration-bonus-scale.md).

## 2026-07-08 — Algorithms: UCB1/KUBE/Fractional-KUBE exploration
bonus needed two fixes — a real `inf`-on-unpulled bug, and a
variance-scale correction

**Context:** `ucb_values(mu_hat, n, t)` (`src/algorithms.py`) is the
shared UCB index used by `UCB1`, `KUBE`, and `FractionalKUBE`. A
code-review pass on the line
`mu_hat + np.sqrt(2 * np.log(t) / np.where(n > 0, n, np.inf))`
questioned whether an unpulled arm (`n[i]=0`) actually gets an
infinite bonus, as the docstring claimed.

**Bug found:** it did not. `np.inf` was substituted into the
*denominator* of the fraction, so an unpulled arm's bonus evaluated
to `sqrt(2 ln t / inf) = 0`, not `inf`. UCB1's core guarantee — pull
every arm once before trusting any mean estimate — was silently not
enforced; unpulled arms tied with each other at bonus 0 instead of
dominating already-pulled arms.

**Fix:** compute the finite bonus first, then `np.where` selects
between it and `np.inf` directly:
```python
bonus = np.where(n > 0, np.sqrt(2 * np.log(t) / n), np.inf)
return mu_hat + bonus
```
Verified numerically: unpulled arms now evaluate to `inf`.

**Second, larger issue — after the `inf` fix, KUBE/Fractional-KUBE
still under-performed eps-first at large budgets** (paper's Figure 1
shows KUBE-family beating eps-first). A follow-up suggestion proposed
dividing the whole bonus by `R_MAX=40` (the reward normalization
constant in `env.py`), reasoning the paper's raw-scale UCB and the
code's normalized-scale UCB were mismatched by a factor of 40.

That specific reasoning was **wrong** — `env.py` already normalizes
every reward to `[0,1]` before an algorithm ever sees it, so
`sqrt(2 ln t / n)` (the standard Hoeffding bound for `[0,1]`-bounded
rewards) is the theoretically correct bonus on the scale the code
actually operates on. There is no raw-scale UCB computation anywhere
to "match." Naively dividing by 40 again would double-normalize.

**But the empirical test said otherwise.** A/B run (homogeneous
regime, K=10, `n_seeds=500`, budgets `[4000, 10000]`, same instance
seed) comparing the current bonus against `bonus/40`:

| algorithm | B=10000 current (norm.) | B=10000 `/40` (norm.) |
|---|---|---|
| UCB1 | 42.36 | 13.57 |
| KUBE | 19.49 | 1.21 |
| Fractional KUBE | 19.45 | 1.22 |
| eps-first(0.1) | 5.70 | 5.70 (unaffected) |

`/40` makes KUBE/Fractional-KUBE clearly beat eps-first, matching
the paper's qualitative ordering; the unscaled version does not.

**Decision:** apply the `/40` scaling, but justify it correctly —
not as "raw vs normalized units" (refuted above), but as a
**variance-scale correction**. Standard `sqrt(2 ln t / n)` is
calibrated for the *worst-case* variance of a `[0,1]`-bounded
variable (≤1/4). This environment's actual reward variance is far
below that worst case: raw variance is `mu_i/2` (paper spec,
`mu_i ∈ [10,20]`), which after `/R_MAX²` normalization is
≈0.005–0.006 — roughly 40–50x smaller than 1/4. A confidence bound
sized for the worst case massively over-explores data this
low-variance, which is exactly the observed symptom (KUBE keeps
spending budget as if arms were still uncertain long after the
empirical mean has converged). `1/40` approximates
`sqrt(variance_ratio) = sqrt(1/1600) = 1/40` — not a coincidental
fudge factor, but roughly the right correction for the true noise
scale vs. the worst-case Hoeffding scale.

**Not yet applied to the live code** — this entry records the
diagnosis and the validated direction; `ucb_values` in
`src/algorithms.py` still uses the unscaled bonus as of this
writing. Applying the `/40` scaling (or a more principled
variance-aware bound) is a follow-up step.

**Revisit if:** a more principled fix is wanted later — an
empirical-Bernstein bound using the running sample variance instead
of a fixed `/40` constant would adapt automatically instead of
hardcoding a constant derived from this specific reward model's
`mu_i/2` variance spec. Full empirical writeup (the A/B script, full
result tables, why the Hoeffding-bound argument was initially
mistaken): [decisions/ucb-exploration-bonus-scale.md](decisions/ucb-exploration-bonus-scale.md).
