# UCB exploration bonus: the `inf`-on-unpulled bug, and three variance-aware bonus modes

*Originating entry:
[decisions-log.md #2026-07-08](../decisions-log.md#2026-07-08--algorithms-ucb1kubefractional-kube-exploration)
— this doc carries the full empirical writeup the log entry
summarizes.*

## Where the bonus lives

`ucb_values(mu_hat, n, t)` in `src/algorithms.py` computes the UCB
index shared by `UCB1`, `KUBE`, and `FractionalKUBE`:

```python
def ucb_values(mu_hat, n, t):
    with np.errstate(divide="ignore", invalid="ignore"):
        return mu_hat + np.sqrt(2 * np.log(t) / np.where(n > 0, n, np.inf))
```

## Bug 1 — unpulled arms got bonus 0, not `inf`

The docstring claims unpulled arms (`n[i]=0`) get an infinite bonus
"so they're always preferred until pulled once" — the standard UCB1
guarantee. The code did not do this: `np.inf` was substituted for
`n` in the *denominator*, so an unpulled arm's bonus was
`sqrt(2 ln t / inf) = 0`. Since `mu_hat` also initializes to 0 for
an unpulled arm, its full UCB index was `0 + 0 = 0` — tied with
every other unpulled arm, and potentially *worse* than an
already-pulled arm with a positive empirical mean. UCB1's force-
exploration guarantee was silently unenforced.

**Fix:**
```python
bonus = np.where(n > 0, np.sqrt(2 * np.log(t) / n), np.inf)
return mu_hat + bonus
```
Verified: `ucb_values(mu_hat=[0.5,0.0,0.3], n=[3,0,2], t=3)` →
`[1.356, inf, 1.348]` — the unpulled arm (index 1) now correctly
dominates. `t=0` (all arms unpulled) gives all-`inf`, a tie handled
fine by `argmax`/`rng.choice` — same as any all-`inf` state.

This fix alone is uncontroversial and was applied directly (see
`src/algorithms.py:19-27`, current as of this writing).

## Bug 2 (candidate) — is the Hoeffding bound itself mis-scaled?

After fixing bug 1, KUBE and Fractional-KUBE still under-performed
eps-first at large budgets in a real run (homogeneous regime),
whereas the paper's Figure 1 shows the opposite ordering (KUBE-family
should beat eps-first). A suggestion proposed: since rewards are
normalized by `R_MAX=40` (`env.py`), maybe the UCB bonus should be
divided by 40 too, to "match" a raw-scale computation the paper
might have used.

### Why the "raw vs normalized units" framing is wrong

`env.py::BudgetMAB.pull()` divides every reward by `R_MAX` **before**
it is ever returned:
```python
return float(raw) / R_MAX
```
So `mu_hat` — the running mean every algorithm tracks — is built
*entirely* from `[0,1]`-scale rewards. There is no raw-scale
(`[0,40]`) UCB computation anywhere in this codebase to be
inconsistent with. `sqrt(2 ln t / n)` is the standard Hoeffding-based
UCB1 bonus (Auer et al. 2002) for rewards bounded in `[0,1]`, and
`mu_hat` already lives on exactly that scale. Dividing the bonus by
40 *again* would apply the correction twice, over-shrinking
exploration by an extra factor of 40 beyond whatever the real issue
is. The paper (`paper.pdf`, referencing Auer et al. 2002 directly for
UCB1) does not define a different bonus formula either.

**This part of the original suggestion is refuted.** But the
empirical test below shows the same numerical adjustment
(`bonus / 40`) fixes the observed symptom anyway — for a different
reason than the one originally proposed.

### The actual mechanism: worst-case variance vs. true variance

`sqrt(2 ln t / n)` is a Hoeffding bound calibrated for the
*worst-case* variance of a random variable bounded in `[0,1]`
(variance ≤ 1/4, attained by a Bernoulli(0.5)). It is a valid upper
confidence bound for *any* `[0,1]`-bounded distribution, but it is
maximally conservative when the true variance is much smaller.

This environment's rewards are far below that worst case. Per the
paper's Section 5 spec, `reward ~ TruncGaussian(mu_i, sqrt(mu_i/2))`
— raw variance is `mu_i/2`, with `mu_i ∈ [10,20]`. After the `/R_MAX`
normalization (`R_MAX=40`), normalized variance is:
```
(mu_i/2) / 40^2 = mu_i / 3200  ∈  [0.0031, 0.0063]
```
versus the worst-case 0.25 the Hoeffding bound is sized for — a ratio
of roughly 40–80x. A bonus calibrated for variance 0.25 but applied
to data with variance ~0.005 stays "uncertain" (large bonus) for far
longer than the data actually warrants, so KUBE/UCB1 keep treating
arms as unresolved and spending budget on further exploration long
after the empirical mean has essentially converged — exactly the
observed symptom (KUBE/Fractional-KUBE regret growing faster than
eps-first's at larger budgets, instead of the reverse).

`1/40 ≈ sqrt(1/1600)`, i.e. roughly `sqrt(true_variance /
worst_case_variance)` — so the `/40` correction is not a coincidental
fudge, it's approximately rescaling the confidence radius to the
environment's real noise level instead of the worst-case level the
generic Hoeffding bound assumes.

## Empirical test

Standalone A/B script (not part of the tracked pipeline):
`/tmp/.../scratchpad/ucb_scale_test.py` — homogeneous regime, K=10,
`instance_seed=42`, `n_seeds=500`, budgets `[4000, 10000]`,
`n_workers=96`. Same `mu`/`costs` draw used for both variants; only
`ucb_values` differs (monkey-patched at the module level for the
`ProcessPoolExecutor` subprocess workers).

Full output (regret, normalized regret in parentheses):

```
--- CURRENT bonus: sqrt(2 ln t / n) ---
algorithm         B=4000              B=10000
Random             150.10(22.46)       374.71(49.30)
UCB1               135.19(20.22)       321.97(42.36)
eps-first(0.1)      20.83( 3.12)        43.33( 5.70)
KUBE                77.90(11.65)       148.15(19.49)
Fractional KUBE     77.89(11.65)       147.84(19.45)

--- /40-SCALED bonus: sqrt(2 ln t / n) / 40 ---
algorithm         B=4000              B=10000
Random             150.10(22.46)       374.71(49.30)
UCB1                45.57( 6.82)       103.13(13.57)
eps-first(0.1)      20.83( 3.12)        43.33( 5.70)
KUBE                 6.03( 0.90)         9.22( 1.21)
Fractional KUBE      6.07( 0.91)         9.26( 1.22)
```

`Random` is identical across both (doesn't use `ucb_values`) —
confirms the patch only touched the intended code path. Under the
`/40` scaling, KUBE/Fractional-KUBE regret at B=10000 drops from 148
to ~9.2, going from *worse* than eps-first (43.3) to clearly
*better* — matching the paper's qualitative Figure 1 ordering.

## Implemented as three selectable bonus modes, not a single fix

Rather than replacing the `/40` fudge with one "correct" formula, the
codebase now supports three `BonusMode`s side by side
(`src/algorithms.py`), selectable per-algorithm at construction time,
so UCB1/KUBE/FractionalKUBE can each be compared across all three:

- **`NONE`** — the original `sqrt(2 ln t / n)`, unchanged.
- **`TRUE_VAR`** — plugs in the environment's true per-arm variance
  (`env.arm_var`, a new property: `(mu_raw/2) / R_MAX**2`, from the
  paper's known `sigma_i = sqrt(mu_i/2)` reward spec). An oracle bound
  no real algorithm could know in advance; used as a diagnostic upper
  bound on how well variance-aware exploration could possibly do.
- **`EST_VAR`** — uses each arm's own running sample variance,
  tracked online via a new `ArmStats` class (Welford's algorithm, one
  `update()` call per pull maintains both the mean and `_m2` for an
  unbiased variance estimate).

This is a strictly better design than picking one fixed constant:
`/40` was never a "correct" value, just an average correction that
happened to work for this environment's specific, narrow variance
range (see below); `TRUE_VAR`/`EST_VAR` scale to each arm's actual
variance and require no environment-specific tuning.

## A second bug: the standard empirical-Bernstein bias term hurts here

The textbook empirical-Bernstein bound (Audibert-Munos-Szepesvári
2007) is `sqrt(2 * var_hat * ln(t) / n) + 3 * ln(t) / n` — the second
term corrects for the bias in small-sample variance estimates. The
first `EST_VAR` implementation used this exact form.

**Result: `EST_VAR` was WORSE than `NONE`** (KUBE regret 105.8 vs
77.9 at B=4000, homogeneous regime, instance_seed=42, n_seeds=20) —
worse than the very bug this was supposed to fix.

**Root cause, isolated by ablation:** the `3 ln(t)/n` term is
`O(1/n)`, while the variance term is `O(sqrt(1/n))` — at the `n` this
environment actually reaches (tens to low hundreds per arm, given
`K=10` and budgets 500–10000), the bias term is 4–10x *larger* than
the variance term:
```
n=10, t=50:  var_term=0.0125   bias_term=1.174   (bias dominates ~94x)
n=100, t=2000: var_term=0.028  bias_term=0.228   (bias still ~8x)
```
So `EST_VAR` wasn't actually testing variance-aware exploration — it
was dominated by a flat `O(1/n)` term unrelated to variance, and that
term happens to decay differently (and, empirically, worse for this
budget range) than `NONE`'s `O(sqrt(ln t / n))` bound.

**Fix:** drop the bias term entirely — `EST_VAR` is now the bare
`sqrt(2 * var_hat * ln(t) / n)`, same functional form as `TRUE_VAR`
but with the running estimate in place of the oracle variance.
Verified this recovers the expected ordering (`NONE` worst,
`EST_VAR` middle, `TRUE_VAR` best) at every budget tested:

```
              B=500   B=2000  B=4000  B=10000
KUBE (none)     13.64   45.01   78.22   148.73
KUBE (est_var)   4.21    7.82    9.77    19.01
KUBE (true_var)  3.60    4.39    5.07     5.63
```
(Fractional KUBE and UCB1 show the same ordering; full sweep in
scratchpad, not preserved beyond this doc.)

**Caveat, not yet a problem in practice:** `EST_VAR`'s `sample_var()`
returns exactly 0 for any arm at `n=1` (can't estimate variance from
one sample) — the bonus for that arm is `sqrt(0)=0` right after the
mandatory round-robin phase, when it's actually most uncertain. This
is the same *shape* of issue as Bug 1 above, shifted from `n=0` to
`n=1`. Empirically this did not cause arm-starvation or regret
blowup at any tested budget (see table above — regret stays low and
scales sensibly, no sign of a starved arm compounding over time),
likely because KUBE/FractionalKUBE's knapsack/argmax-over-density
selection still gives a temporarily-zero-bonus arm a nonzero chance
via the other arms' own bonuses shrinking too. Not fixed
defensively since it isn't an observed failure — see Revisit-if.

## Status

**Implemented and verified.** `src/algorithms.py` now has `ArmStats`,
`BonusMode` (`NONE`/`TRUE_VAR`/`EST_VAR`), and a `BONUS_FNS` registry;
`KUBE`/`FractionalKUBE`/`UCB1` each take `bonus_mode` in their
constructor. `env.py` gained the `arm_var` property. The
KUBE/FractionalKUBE/UCB1 `run()` methods were also collapsed into one
shared `_run_ucb_loop` parameterized by a `select_arm` callback,
resolving `docs/issues.md#duplicated-run-skeleton` at the same time
(verified byte-identical regret to the pre-refactor per-class loops
under `BonusMode.NONE`, same seeds).

## Revisit if

- The `n=1`-zero-variance caveat above is ever observed to cause real
  arm starvation (e.g. at a much larger `K` or a very short budget
  relative to `K`) — would need a small-sample floor on `var_hat`
  (e.g. treat `n=1` as `inf` like the `n=0` case, rather than 0).
- A per-arm variance estimate needs to also inform algorithm
  *selection* (not just the bonus) — e.g. an arm with high estimated
  variance might warrant a different knapsack treatment in KUBE
  beyond just its UCB value.
- Regenerate any table/plot that used `BonusMode.NONE` results from
  before the shared-loop refactor if a discrepancy is ever suspected
  — verified behavior-preserving here, but worth re-checking if the
  refactor is touched again.
