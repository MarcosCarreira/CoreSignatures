# `wilmott_n4_comparison.py` — Wilmott claim verification

**Purpose.** Verify the Wilmott article's specific empirical claim:
> *"On the target f(t) = sin(2π t²), the algorithm converges to interior
> breakpoints that agree to four significant digits with the breakpoints
> found by the algebraic core-signature inversion."*

The Wilmott "algorithm" is the **mean-L¹ evolutionary algorithm** in
`Frechet.nb`; the "algebraic core-signature inversion" is the
**8-equation Chen-identity polynomial system** solved in
`CoreSignatures202412.nb`.

This script ports **both** algorithms to Python (single file, no external
notebooks needed), runs them against the same target, and reports the
significant-digit agreement on the interior breakpoint triple (t₁, t₂, t₃).

## How to run in PyCharm

1. Open this folder (`2026/`) as a PyCharm project.
2. Open `wilmott_n4_comparison.py`. Right-click → Run.
3. Default settings run the **full Frechet.nb baseline** (population = 250,
   generations = 50, 5 ES seeds, 200 algebraic multi-starts). Wall-clock
   ≈ 2–5 minutes on a modern laptop.
4. For a fast smoke test (under-converged, not for verification):
   ```
   python wilmott_n4_comparison.py --quick
   ```

## What's inside

The script is self-contained and has six logical sections:

1. **Path-signature machinery** (`piecewise_signature`,
   `line_segment_signature`, `chen_concat`) — explicit Chen-identity
   concatenation of n=4 piecewise-linear segments to produce the
   core signature {σ_(1,), σ_(2,), σ_(1,2), σ_(1,1,2), σ_(1,2,2),
   σ_(1,1,1,2), σ_(1,1,2,2), σ_(1,2,2,2)}.

2. **Target core signature** (`sin_2pi_t2_target`) — fast cumulative
   trapezoidal integration on a 40,000-point grid. Verified against the
   closed form σ_(1,2) = −FS(2)/2 where FS is the Fresnel sine integral
   (matches to ~10⁻⁶ on the default grid).

3. **Algorithm A — algebraic Chen-system inversion** (`algebraic_inversion`)
   — port of `CoreSignatures202412.nb chen4z4core`. Solves the
   7-equation-in-7-unknowns nonlinear polynomial system (level-1 σ_(1,)=T
   is identically satisfied) via multi-start `scipy.optimize.least_squares`
   with the Levenberg-Marquardt method. Residuals are scaled by k! for
   numerical conditioning. Each found solution is deduplicated by
   Euclidean distance; tolerance for "real solution" is residual < 10⁻⁷.

4. **Algorithm B — mean-L¹ (μ+λ)-ES** (`frechet_es`) — port of `Frechet.nb`
   with the documented parameters: popsize=250, σ=0.025, 50 generations,
   elite-plus-mutation selection with fresh random "islands" injected each
   generation. Operates on the 6-dimensional space of 3 interior (t, y)
   breakpoint coordinates with pinned endpoints (0,0) and (1,0).
   Fitness is mean-|f(t) − g(t)| on 1001 sample points.

5. **Comparison runner** — for each algebraic solution, reports breakpoint
   agreement with the best ES result: relative deltas, significant-digit
   estimate, verdict against the 4-sig-figs Wilmott claim.

6. **CLI** (`argparse`) — see `--help` for options.

## What to expect

**The full Frechet.nb baseline takes a few minutes** because the ES makes
~12,500 function evaluations per seed (popsize × generations). On 5
independent seeds × 12,500 evaluations × ~50 µs each = ~3 seconds per seed
× 5 = 15 s for the ES; plus ~30 s for the algebraic multi-start; plus
overhead. Total ~ 60 s in the optimistic case, possibly 3-5 minutes if
the ES is slowed by many small-segment paths.

**Expected outputs:**

- The algebraic solve should find between 1 and 4 distinct real solutions
  (ALS26 Table 3 has PRdeg = 4 for piecewise-linear at d=2, k=4; some may
  be complex for the specific sin(2πt²) target). My sandbox run found one:
  θ ≈ (0.1037, 0.5296, 0.8868, β=(0.244, 2.636, −6.599, 10.690)).

- Each ES seed should converge to mean-L¹ ≪ 0.01 with the full
  Frechet.nb settings. If the cross-seed convergence is robust (J spread
  tight) and the converged ES interior-t values match the algebraic
  solution to ≥4 significant digits, **the Wilmott claim is confirmed**.

- If the ES converges robustly but to different (t₁, t₂, t₃) than the
  algebraic solution, the Wilmott claim is **weaker than stated**; the
  paper's empirical observation may have been on a slightly different
  configuration (different popsize, different bounds, different sigma).

## Sandbox partial result (under-converged, DO NOT rely on for verdict)

```
Algebraic Sol#1:  t = (0.1037, 0.5296, 0.8868)
ES (popsize=60, 15 gens):  t_interior = (0.5726, 0.8082, 0.9311)
                          mean-L¹ = 7.6 × 10⁻²  ← not converged
```

These do not agree to 4 sig digits, but the ES had not converged at
sandbox settings. **Run with defaults (popsize=250, 50 generations) for
the actual verdict.**

## If the ES converges to a different basin than the algebraic Sol

That's a real finding: the mean-L¹ minimum and the Chen-system algebraic
root are *not* the same point in parameter space. The Wilmott article
would need a softer claim ("the algorithms converge to qualitatively
similar piecewise-linear approximations") rather than a strict
4-sig-digit numerical agreement. Send me the output and we'll decide
together what edit (if any) to make to the Wilmott `.tex`.

## Files referenced

- `verify_algorithm_v3.py` — Test A/B/C for n=2 inversion (already ran in
  sandbox, confirms Algorithm 1 reproducibility at n=2).
- `verify_algorithm_REPORT.md` — Findings from the n=2 verification.
- `verify_n4_wilmott.py` — Earlier n=4 single-algorithm test (regression
  only, not the Wilmott comparison).
- `wilmott_n4_comparison.py` — **This script: the actual Wilmott
  claim test.**

## Requirements

- Python 3.9+
- `numpy >= 1.20`
- `scipy >= 1.7`

`pip install numpy scipy`
