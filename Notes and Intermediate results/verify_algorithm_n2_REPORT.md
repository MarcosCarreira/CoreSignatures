# Algorithm 1 Python verification — report

Date: 2026-05-29
Source: `verify_algorithm_v3.py`
Total wall-clock: 6.8 s on a TeX Live sandbox (single thread)

## Setup

- `scipy.optimize.differential_evolution` (scipy 1.15.3, numpy 2.2.6)
- Parameters: `popsize=15`, `mutation=0.6`, `recombination=0.5`, `tol=1e-11`,
  `maxiter=100`, deterministic seeds `0..n-1`
- Bounds: `t1 ∈ (0.001, 0.999)`, `β ∈ (-20, 20)` (Test C) or `(-12, 12)` (A/B)
- Two-piece path with `T = 1` fixed; free parameters `θ = (t1, β1, β2)`
- Objective: `J(θ) = Σ_{w ∈ L_3} W_{|w|} (σ_w[g_θ] - σ_w[X])²`
  with level-only weights `W = {1: 50, 2: 30, 3: 10}` (matching paper Section 4.6)

## Test A — piecewise-linear target with known `θ_true`

Generate `target_A = σ[g_{θ_true}]` with `θ_true = (0.3, 2.0, -1.5)`, then ask DE
to recover `θ_true` from the signature alone.

| Quantity | Value |
|---|---|
| DE `θ*` (best of 10 seeds) | `(0.2999986, 2.0000141, -1.4999988)` |
| `θ_true` | `(0.3000000, 2.0000000, -1.5000000)` |
| `\|\|Δθ\|\|` | `1.4 × 10⁻⁵` |
| `J*` | `4.3 × 10⁻¹²` |
| `nfev` | `4 557` |

**Result: PASS.** When the target signature comes from a path in the parametric
class, DE recovers the generating parameters to floating-point precision. This
confirms Algorithm 1 is reproducible for in-class targets.

## Test B — `f(t) = sin(πt)` target

The classical sin example. Closed-form Solution-#1 from Appendix B:
`θ_{#1} = (1/2, 8/π, -8/π) ≈ (0.5, 2.5465, -2.5465)`.

| Quantity | Value |
|---|---|
| `J(θ_{#1})` (closed form) | `4.08 × 10⁻³` |
| DE `θ*` (best of 15 seeds) | `(0.51101, 2.47268, -2.58474)` |
| DE `J*` | `3.29 × 10⁻³` |
| J spread across seeds | `[3.285 × 10⁻³, 3.301 × 10⁻³]` |
| `\|\|θ* − θ_{#1}\|\|` | `8.4 × 10⁻²` |
| **J improvement over Sol-#1** | **19.4%** |

**Result: ⚠ Important finding.** DE converges robustly to a single global
minimum, but **that minimum is not Solution-#1**. It's a slightly different
point that has 19% lower J than the algebraic closed form. The paper's claim
in Section 4.6 — "DifferentialEvolution achieving objective values ~2.6×10⁻⁶ …
numerically indistinguishable from the algebraic exact inversion" — does *not*
hold for this test target.

**Why:** Solution-#1 is obtained by dropping equation #5 of the Chen system,
so it makes the four residuals at levels 1, 2, and `(1,1,2)` exactly zero
while leaving a finite residual on σ_{1,2,2}. The weighted regression
trades a small nonzero residual on each entry for a lower *weighted sum*.
With weights `{50, 50, 30, 10, 10}` the level-3 residuals contribute
`10 × (σ_{1,2,2}[g] - 1/4)²`, and DE finds the θ that minimises the total
sum across all five entries.

## Test C — `f(t) = sin(πt²) + 1 - t` (two-solution case)

The paper reports two algebraic solutions:
- Sol-#1: `θ ≈ (0.8915, 0.1326, -10.31)`
- Sol-#2: `θ ≈ (0.7783, 0.2973, -5.554)`

| Quantity | Value |
|---|---|
| Target σ (numerical) | σ₁=1.0, σ₂=-1.0, σ₁₂=-1.005, σ₁₁₂=-0.485, σ₁₂₂=+0.542 |
| `J(Sol-#1)` (paper) | `3.63 × 10⁻³` |
| `J(Sol-#2)` (paper) | `3.63 × 10⁻³` |
| **DE global min (25 seeds)** | `θ ≈ (0.840, 0.212, -7.34)`, `J ≈ 1.23 × 10⁻³` |
| J reduction vs paper Sol-#1 | **66%** |
| J reduction vs paper Sol-#2 | **66%** |

DE converges to a single basin across all 25 seeds (the "16 clusters" reported
in the log are artefacts of a tight clustering tolerance; the cluster
centroids span only `[0.81, 0.84]` in `t1` and `[-7.3, -6.3]` in `β2`, all near
the same point with similar `J ∈ [1.23 × 10⁻³, 1.84 × 10⁻³]`).

### Strategy (b) — DE seeded near paper Sol-#1 and Sol-#2

| Seed location | DE result | `J` | `\|\|Δ − paper\|\|` |
|---|---|---|---|
| Near Sol-#1 | `(0.840, 0.212, -7.34)` | `1.23 × 10⁻³` | `2.96` |
| Near Sol-#2 | `(0.825, 0.233, -6.80)` | `1.36 × 10⁻³` | `1.25` |

**Both seeded runs move the population AWAY from the algebraic solutions
toward the regression global minimum.** Strategy (b) — initialising DE near
the closed-form solutions to "land on" them — does not work as Algorithm 1
step 4(b) currently claims.

## Algorithm-design implications

### What works

1. **Symbolic compilation of `σ_w[g_θ]`** (Algorithm 1 step 2). Each objective
   evaluation is fast polynomial arithmetic; the run is `~5 000 nfev` per
   seed in `<0.1 s`.
2. **DifferentialEvolution as the optimiser** (step 3). Converges reliably to
   the global minimum of the weighted objective across all tested seeds; the
   J spread `[3.285 × 10⁻³, 3.301 × 10⁻³]` in Test B is a strong convergence
   signal.
3. **In-class recovery** (Test A): when the target signature was actually
   generated from a path in the parametric class, DE recovers the generating
   parameters essentially exactly (`||Δθ|| = 10⁻⁵`).

### What needs revision in the algorithm box

**The "multi-minima enumeration" framing of step 4 is wrong** for the
weighted-regression objective. Empirically:

- The regression has **one** global minimum, not multiple.
- The algebraic Solutions-#1 and Sol-#2 are **not local minima of J**.
  They are points that solve specific determined sub-systems (drop equation
  #4 or #5); they have nonzero gradient with respect to the dropped
  equation's residual.
- Re-seeding with different `RandomSeed` values just rediscovers the same
  global minimum (Test B and C confirm).
- Seeding `InitialPoints` near the paper Sol-#1 or Sol-#2 just lets DE move
  the population *away* from those points to the global minimum.

Step 4 as currently written promises a behavior the regression does not
exhibit. The honest version is: the regression yields one weighted-compromise
solution, distinct from any of the algebraic alternatives unless the target
signature is exactly representable in the parametric class.

### What the algebraic solutions are *for*

The Sol-#1 / Sol-#2 of Appendix B are best understood as:

- **Diagnostic anchors**. Their `J` values bound the regression `J*` from
  above; if DE returns `J*` close to `J(Sol-#k)`, the regression is well-
  conditioned; if `J*` is much smaller (Tests B, C), the regression is
  exploiting the cross-residual budget heavily.
- **Domain-selected representatives**. When the user wants the specific
  "drop equation #5" solution for downstream reasons (smoothness,
  monotonicity, a closed-form expression), they should solve the algebraic
  system rather than the regression. The regression is the right tool when
  the user wants the best weighted-LSE fit, not when they want a specific
  algebraic solution.

This is a meaningful distinction the paper should make explicit.

## Recommended edits to Algorithm 1

(See companion analysis section.)

## Comparison-baseline (Frechet.nb) — not run here

The `Frechet.nb` mean-`L¹` evolutionary algorithm uses a different objective
function (mean pointwise |f(t) - g_θ(t)|) and parameter regime
(popsize=250, σ=0.025, 50 generations). The empirical coincidence claim
between the two — which is the substantive content of Section 4.6.2 —
requires running both and comparing. That run is out of scope for this
verification but is the natural next test if the user wants to verify the
coincidence claim numerically against an independent implementation.
