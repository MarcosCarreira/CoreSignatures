"""
wilmott_n4_comparison.py
========================

Verify the Wilmott claim:
    For f(t) = sin(2π t²) on [0, 1] with n=4 segments, the interior
    breakpoints (t1, t2, t3) from the algebraic Chen-identity inversion
    agree to four significant figures with the breakpoints from the
    mean-L¹ evolutionary algorithm.

This is a Python port of:
    - CoreSignatures202412.nb  → algebraic Chen system (`chen4z4core`)
    - Frechet.nb               → mean-L¹ (μ+λ)-ES (`genpts`, `mutate`,
                                  `next`, `poplist = NestList[next, ...]`)

Both algorithms operate on the same 7-parameter space:
    θ = (t1, t2, t3, β1, β2, β3, β4)
with the constraint 0 < t1 < t2 < t3 < 1 and T = t1 + (t2 - t1) + (t3 - t2)
+ (1 - t3) = 1.

For the time-augmented path X(t) = (t, f(t)), the y-coordinate of an
interior breakpoint at time t_i is determined by the slope sequence:
    y_i = β_1 * t_1 + Σ_{j=2..i} β_j * (t_j - t_{j-1})

So an interior breakpoint is (t_i, y_i) with y_i = (cumulative slope sum).

Usage
-----
PyCharm: open in interpreter, hit Run.
CLI:
    python wilmott_n4_comparison.py
    python wilmott_n4_comparison.py --seeds 20 --report verbose

Requires: numpy>=1.20, scipy>=1.7
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
from scipy.optimize import differential_evolution, fsolve

PI = math.pi
T = 1.0
N_SEGMENTS = 4
TRUNCATION_LEVEL = 4
LEVEL_WEIGHTS = {1: 50.0, 2: 30.0, 3: 10.0, 4: 3.0}

# Lyndon words on {1, 2} through length 4 (the 8 core entries)
CORE_WORDS = [
    (1,), (2,),
    (1, 2),
    (1, 1, 2), (1, 2, 2),
    (1, 1, 1, 2), (1, 1, 2, 2), (1, 2, 2, 2),
]


# ---------------------------------------------------------------------------
# 1.  Path-signature machinery — explicit Chen-identity concatenation
# ---------------------------------------------------------------------------

_ALL_WORDS = [()]
for _k in range(1, TRUNCATION_LEVEL + 1):
    for _w in product((1, 2), repeat=_k):
        _ALL_WORDS.append(_w)

_SPLITS = {w: [(w[:j], w[j:]) for j in range(len(w) + 1)] for w in _ALL_WORDS}


def line_segment_signature(slope_y: float, dt: float) -> dict:
    """σ_w of a single time-augmented segment with slope (1, slope_y), duration dt.

    Closed form: σ_{i_1, ..., i_k} = (∏_j slope_{i_j}) · dt^k / k!
    where slope_1 = 1 (time) and slope_2 = slope_y.
    """
    sig = {(): 1.0}
    for w in _ALL_WORDS:
        if not w:
            continue
        prod = 1.0
        for i in w:
            prod *= (1.0 if i == 1 else slope_y)
        sig[w] = prod * (dt ** len(w)) / math.factorial(len(w))
    return sig


def chen_concat(s1: dict, s2: dict) -> dict:
    """σ_w[X*Y] = Σ_{w = uv} σ_u[X] · σ_v[Y]."""
    return {w: sum(s1[u] * s2[v] for u, v in _SPLITS[w]) for w in _ALL_WORDS}


def piecewise_signature(theta: Sequence[float]) -> dict:
    """Core signature of the 4-segment time-augmented PL path parametrised by θ."""
    t1, t2, t3, b1, b2, b3, b4 = theta
    dts = (t1, t2 - t1, t3 - t2, 1.0 - t3)
    s = line_segment_signature(b1, dts[0])
    for slope, dt in zip((b2, b3, b4), dts[1:]):
        s = chen_concat(s, line_segment_signature(slope, dt))
    return {w: s[w] for w in CORE_WORDS}


def core_signature_residuals(theta: Sequence[float], target: dict,
                               drop_trivial: bool = True) -> np.ndarray:
    """Return the residual vector σ_w[g_θ] − σ_w[X].

    If drop_trivial is True (default), σ_(1,) = T = 1 is automatically
    satisfied by the parametrisation and is removed, giving 7 non-trivial
    residuals in 7 unknowns (exactly determined). This is the right system
    to feed to a Newton-type solver.
    """
    g = piecewise_signature(theta)
    if drop_trivial:
        words = [w for w in CORE_WORDS if w != (1,)]
    else:
        words = CORE_WORDS
    return np.array([g[w] - target[w] for w in words])


def core_signature_residuals_scaled(theta, target):
    """Same as core_signature_residuals but scaled by level for better
    conditioning. Level-k residuals are scaled by k! so they have
    comparable magnitudes across levels."""
    g = piecewise_signature(theta)
    words = [w for w in CORE_WORDS if w != (1,)]
    return np.array([math.factorial(len(w)) * (g[w] - target[w]) for w in words])


# ---------------------------------------------------------------------------
# 2.  Target core signature for f(t) = sin(2π t²)
#     Fast grid-based cumulative trapezoid integration; verified against the
#     closed-form -FS(2)/2 for σ_{1,2}.
# ---------------------------------------------------------------------------

def target_core_signature(f, fp, T_: float = T, N: int = 40000) -> dict:
    t_grid = np.linspace(0.0, T_, N + 1)
    dt = t_grid[1] - t_grid[0]
    fp_vals = fp(t_grid)
    h = {(): np.ones_like(t_grid)}
    for w in _ALL_WORDS:
        if not w:
            continue
        parent = w[:-1]
        i = w[-1]
        integrand = h[parent] * (1.0 if i == 1 else fp_vals)
        cum = np.zeros_like(t_grid)
        cum[1:] = np.cumsum((integrand[:-1] + integrand[1:]) / 2) * dt
        h[w] = cum
    return {w: float(h[w][-1]) for w in CORE_WORDS}


def sin_2pi_t2_target(N: int = 40000) -> dict:
    f = lambda t: np.sin(2 * PI * t * t)
    fp = lambda t: 4 * PI * t * np.cos(2 * PI * t * t)
    return target_core_signature(f, fp, T_=1.0, N=N)


# ---------------------------------------------------------------------------
# 3.  Algorithm A — Algebraic Chen-system inversion
#     Port of CoreSignatures202412.nb `chen4z4core` solve at level 4:
#     8 polynomial equations in 8 unknowns (t1,t2,t3,t4,β1,β2,β3,β4) with
#     constraint t1+t2+t3+t4 = z1 = 1. Use multi-start Newton to enumerate
#     real solutions.
# ---------------------------------------------------------------------------

def algebraic_inversion(target: dict, n_starts: int = 80, seed: int = 0,
                        verbose: bool = False, progress_every: int = 100) -> list:
    """Return a list of (θ, residual_norm) for each real solution found.

    Multi-start scipy.optimize.least_squares from random initial points.
    The level-4 core-signature system is exactly determined (7 non-trivial
    equations in 7 unknowns after fixing T=1), so it has finitely many
    algebraic solutions; per ALS26 Table 3 the path recovery degree is
    4 for d=2, k=4, m=(1,1,1,1).
    """
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed)
    solutions = []
    t_stage = time.time()
    for trial in range(n_starts):
        ts = np.sort(rng.uniform(0.05, 0.95, size=3))
        bs = rng.uniform(-12.0, 12.0, size=4)
        x0 = np.concatenate([ts, bs])
        try:
            res = least_squares(
                core_signature_residuals_scaled, x0, args=(target,),
                method='lm', xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=4000,
            )
        except Exception:
            continue
        x_root = res.x
        t1_, t2_, t3_ = x_root[0], x_root[1], x_root[2]
        if not (0.0 < t1_ < t2_ < t3_ < 1.0):
            continue
        residual = np.linalg.norm(core_signature_residuals(x_root, target))
        if residual > 1e-7:
            continue
        already = any(np.linalg.norm(x_root - np.array(s[0])) < 1e-4
                      for s in solutions)
        if not already:
            solutions.append((tuple(x_root.tolist()), float(residual)))
            if verbose:
                print(f"  trial {trial:4d}: new solution, ||res||={residual:.2e}, "
                      f"θ=({t1_:.5f},{t2_:.5f},{t3_:.5f}, "
                      f"β=({x_root[3]:+.3f},{x_root[4]:+.3f},"
                      f"{x_root[5]:+.3f},{x_root[6]:+.3f}))")
        if verbose and progress_every > 0 and (trial + 1) % progress_every == 0:
            print(f"  ... {trial+1}/{n_starts} trials done, {len(solutions)} unique "
                  f"sol(s) so far ({time.time()-t_stage:.1f}s)")
    solutions.sort(key=lambda s: s[1])
    return solutions


# ---------------------------------------------------------------------------
# 4.  Algorithm B — Mean-L¹ evolutionary algorithm (port of Frechet.nb)
#
#     Frechet.nb parameter settings (extracted from the notebook):
#       popsize       = 250
#       generations   = 50
#       σ (mutation)  = 0.025  (Gaussian noise on (t,y) of interior points)
#       eps           = 0.5    (boundary slack used by filterbounds)
#       Path: fixed endpoints (0,0) and (1,0); n_interior = n_segments - 1
#            interior points each parametrised as (t_i, y_i) ∈ [0,1]×[-2,2]
#       Fitness: mean over 1001 sample points of |f(t_k) - g_θ(t_k)|
#       Selection: top `best` survivors retained as elite; mutated copies
#            and fresh-random "islands" combined, sorted, top popsize kept.
# ---------------------------------------------------------------------------

@dataclass
class FrechetESConfig:
    popsize: int = 250
    generations: int = 50
    sigma: float = 0.025
    sigma_start: float = None       # if set, σ decays linearly from sigma_start to sigma over generations
    best: int = 25      # elite count per generation
    n_islands: int = 25  # fresh random injections per generation
    eps: float = 0.5    # filterbounds slack
    rangex: tuple = (0.0, 1.0)
    rangey: tuple = (-2.0, 2.0)
    n_segments: int = 4
    n_sample: int = 1001
    # Optional: per-coordinate sigma (t may want larger jiggle than y)
    sigma_t_mult: float = 1.0       # multiplier on σ for t-coordinate
    sigma_y_mult: float = 1.0       # multiplier on σ for y-coordinate


def _interior_to_pwlin(t_interior: np.ndarray, y_interior: np.ndarray,
                       endpoints=((0.0, 0.0), (1.0, 0.0))):
    """Return arrays of (t, y) breakpoints including pinned endpoints."""
    ts = np.concatenate(([endpoints[0][0]], t_interior, [endpoints[1][0]]))
    ys = np.concatenate(([endpoints[0][1]], y_interior, [endpoints[1][1]]))
    order = np.argsort(ts)
    return ts[order], ys[order]


def _eval_pwlin(t_query: np.ndarray, ts: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.interp(t_query, ts, ys)


def meandist(f, t_interior, y_interior, T_=1.0, n_sample=1001):
    """Mean L¹ distance over n_sample equispaced t in [0, T_]."""
    ts, ys = _interior_to_pwlin(t_interior, y_interior)
    t_query = np.linspace(0.0, T_, n_sample)
    g = _eval_pwlin(t_query, ts, ys)
    return float(np.mean(np.abs(f(t_query) - g)))


def _genpts(rng: np.random.Generator, cfg: FrechetESConfig):
    """Generate one random interior-point configuration: (t_interior, y_interior)."""
    n_int = cfg.n_segments - 1
    ts = rng.uniform(cfg.rangex[0], cfg.rangex[1], size=n_int)
    ys = rng.uniform(cfg.rangey[0], cfg.rangey[1], size=n_int)
    ts.sort()
    return ts, ys


def _mutate(t_interior, y_interior, rng, cfg: FrechetESConfig, sigma_eff: float = None):
    """Gaussian σ mutation on each (t, y) interior coordinate, with bound filter.

    sigma_eff overrides cfg.sigma if provided (for σ scheduling)."""
    σ = sigma_eff if sigma_eff is not None else cfg.sigma
    n_int = len(t_interior)
    new_t = t_interior + rng.normal(0.0, σ * cfg.sigma_t_mult, size=n_int)
    new_y = y_interior + rng.normal(0.0, σ * cfg.sigma_y_mult, size=n_int)
    new_t = np.clip(new_t, cfg.rangex[0] - cfg.eps, cfg.rangex[1] + cfg.eps)
    new_y = np.clip(new_y, cfg.rangey[0] - cfg.eps, cfg.rangey[1] + cfg.eps)
    new_t.sort()
    for i in range(1, n_int):
        if new_t[i] - new_t[i - 1] < 1e-4:
            new_t[i] = new_t[i - 1] + 1e-4
    return new_t, new_y


def frechet_es(f, cfg: FrechetESConfig = FrechetESConfig(), seed: int = 0,
               verbose: bool = False) -> tuple:
    """Run the Frechet.nb-style (μ+λ)-ES. Return (best_t_interior, best_y_interior, best_score)."""
    rng = np.random.default_rng(seed)
    # Initial population
    pop = [_genpts(rng, cfg) for _ in range(cfg.popsize)]
    scores = [meandist(f, t, y, n_sample=cfg.n_sample) for (t, y) in pop]
    for gen in range(cfg.generations):
        # Compute σ for this generation (linear schedule if sigma_start set)
        if cfg.sigma_start is not None:
            frac = gen / max(cfg.generations - 1, 1)
            sigma_eff = cfg.sigma_start * (1 - frac) + cfg.sigma * frac
        else:
            sigma_eff = cfg.sigma
        order = np.argsort(scores)
        pop = [pop[i] for i in order]
        scores = [scores[i] for i in order]
        elite = pop[: cfg.best]
        copies_per_elite = max(1, (cfg.popsize - cfg.best - cfg.n_islands) // cfg.best)
        mutations = []
        for (t, y) in elite:
            for _ in range(copies_per_elite):
                mutations.append(_mutate(t, y, rng, cfg, sigma_eff=sigma_eff))
        islands = [_genpts(rng, cfg) for _ in range(cfg.n_islands)]
        all_pop = elite + mutations + islands
        all_scores = [meandist(f, t, y, n_sample=cfg.n_sample) for (t, y) in all_pop]
        order = np.argsort(all_scores)[: cfg.popsize]
        pop = [all_pop[i] for i in order]
        scores = [all_scores[i] for i in order]
        if verbose and (gen % 20 == 0 or gen == cfg.generations - 1):
            print(f"  gen {gen:4d}: best mean-L¹ = {scores[0]:.5e}  (σ={sigma_eff:.4f})")
    best_t, best_y = pop[0]
    return best_t, best_y, scores[0]


# ---------------------------------------------------------------------------
# 5.  Comparison runner
# ---------------------------------------------------------------------------

def _y_at_breakpoints(theta) -> np.ndarray:
    """Given algebraic θ = (t1,t2,t3,β1,β2,β3,β4), return interior (t_i, y_i)
    where y_i = cumulative path height at t_i."""
    t1, t2, t3, b1, b2, b3, b4 = theta
    y1 = b1 * t1
    y2 = y1 + b2 * (t2 - t1)
    y3 = y2 + b3 * (t3 - t2)
    return np.array([t1, t2, t3]), np.array([y1, y2, y3])


def report_agreement(label, breakpoints_A, breakpoints_B):
    """Compute and print significant-digit agreement between two breakpoint arrays."""
    A = np.asarray(breakpoints_A)
    B = np.asarray(breakpoints_B)
    deltas = np.abs(A - B)
    rel = deltas / np.maximum(np.abs(A), 1e-12)
    print(f"\n{label}")
    for i, (a, b, d, r) in enumerate(zip(A, B, deltas, rel)):
        digits = -np.log10(max(r, 1e-15))
        print(f"  t{i+1}:  algebraic={a:+.6f}   ES={b:+.6f}   |Δ|={d:.2e}   "
              f"rel={r:.2e}   (~{digits:.1f} sig.dig)")
    worst_rel = float(np.max(rel))
    worst_digits = -np.log10(max(worst_rel, 1e-15))
    print(f"\nWorst-case relative agreement: {worst_rel:.2e} "
          f"(~{worst_digits:.1f} significant digits)")
    return worst_digits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # === Default settings: comprehensive Wilmott-claim verification ===
    # Aggressive ES (σ-scheduled, more generations, more seeds), verbose
    # reporting, full algebraic multi-start. Wall-clock ~15-25 min on laptop.
    # Override with --frechet-original for the literal Frechet.nb parameters,
    # or --quick for a fast sandbox test.
    parser.add_argument("--seeds-algebraic", type=int, default=400,
                        help="Multi-start random initialisations for algebraic solve "
                             "(default 400)")
    parser.add_argument("--seeds-es", type=int, default=10,
                        help="Independent ES restarts (default 10 for tight "
                             "cross-seed convergence diagnostic)")
    parser.add_argument("--es-popsize", type=int, default=250,
                        help="ES population size (default 250 = Frechet.nb)")
    parser.add_argument("--es-generations", type=int, default=200,
                        help="ES generation count (default 200; Frechet.nb uses 50)")
    parser.add_argument("--es-sigma", type=float, default=0.005,
                        help="ES final mutation σ (default 0.005; Frechet.nb uses 0.025 constant)")
    parser.add_argument("--es-sigma-start", type=float, default=0.10,
                        help="ES initial mutation σ; linearly decays to --es-sigma. "
                             "Default 0.10 → 0.005 over generations. Set to --es-sigma "
                             "for constant σ.")
    parser.add_argument("--es-best", type=int, default=25,
                        help="ES elite count per generation (default 25)")
    parser.add_argument("--es-islands", type=int, default=25,
                        help="Fresh-random injections per generation (default 25)")
    parser.add_argument("--es-sigma-t-mult", type=float, default=2.5,
                        help="Multiplier on σ for t-coordinate mutations (default 2.5)")
    parser.add_argument("--frechet-original", action="store_true",
                        help="Match Frechet.nb settings exactly: popsize=250, "
                             "generations=50, σ=0.025 constant (no schedule), "
                             "t-mult=1.0, 5 seeds. Use this to reproduce the "
                             "notebook's behaviour exactly.")
    parser.add_argument("--quick", action="store_true",
                        help="Quick diagnostic run: popsize=80, generations=25, "
                             "1 ES seed, 50 algebraic multi-starts. "
                             "For sandbox testing only; not enough to verify the claim.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-generation progress prints (still shows "
                             "per-seed totals and the final comparison).")
    args = parser.parse_args(argv)

    # Override flags
    if args.frechet_original:
        args.es_popsize = 250
        args.es_generations = 50
        args.es_sigma = 0.025
        args.es_sigma_start = 0.025  # no schedule
        args.es_sigma_t_mult = 1.0
        args.seeds_es = 5
        args.seeds_algebraic = 200
        print("[--frechet-original mode: popsize=250 × gen=50 × 5 seeds × σ=0.025 constant]\n")
    elif args.quick:
        args.seeds_algebraic = 50
        args.es_popsize = 80
        args.es_generations = 25
        args.seeds_es = 1
        args.es_sigma_start = 0.05
        print("[--quick mode: under-converged settings, for testing only]\n")

    # Verbose by default (--quiet to suppress per-generation prints)
    args.verbose = not args.quiet

    t0 = time.time()
    print("=" * 72)
    print("Wilmott n=4 comparison:  sin(2π t²) target")
    print("=" * 72)
    print(f"Config:  algebraic_multi_starts={args.seeds_algebraic}, "
          f"ES_seeds={args.seeds_es}, ES_popsize={args.es_popsize}, "
          f"ES_gen={args.es_generations}")
    es_evals = args.es_popsize * args.es_generations * args.seeds_es
    # Wall-clock estimate: algebraic ~0.5s per multi-start, ES ~50µs per eval
    est_alg_sec = args.seeds_algebraic * 0.5
    est_es_sec = es_evals * 5e-5
    print(f"ES function evaluations ≈ {es_evals:,}")
    print(f"Estimated wall-clock: algebraic ~{est_alg_sec/60:.1f} min + "
          f"ES ~{est_es_sec/60:.1f} min  →  total ~{(est_alg_sec+est_es_sec)/60:.1f} min")
    print()

    # ----- Target signature -----
    t_stage = time.time()
    print("Computing target core signature (8 entries)...")
    target = sin_2pi_t2_target(N=40000)
    for w in CORE_WORDS:
        print(f"  σ_{w}: {target[w]:+.6f}")
    print(f"  (target sig in {time.time()-t_stage:.2f}s)")

    # ----- Algorithm A: Algebraic Chen-system inversion -----
    t_stage = time.time()
    print(f"\nRunning algebraic Chen inversion ({args.seeds_algebraic} multi-starts)...")
    algebraic_solutions = algebraic_inversion(target, n_starts=args.seeds_algebraic,
                                               verbose=args.verbose)
    print(f"  Found {len(algebraic_solutions)} algebraically distinct solutions "
          f"({time.time()-t_stage:.1f}s)")
    for idx, (theta, res) in enumerate(algebraic_solutions[: min(6, len(algebraic_solutions))]):
        t1, t2, t3 = theta[0], theta[1], theta[2]
        b1, b2, b3, b4 = theta[3:]
        print(f"  Sol#{idx+1}: t=({t1:.5f}, {t2:.5f}, {t3:.5f}), "
              f"β=({b1:+.3f},{b2:+.3f},{b3:+.3f},{b4:+.3f}), ||res||={res:.2e}")

    if not algebraic_solutions:
        print("\nNo algebraic solutions found. Try increasing --seeds-algebraic.")
        return 1

    # ----- Algorithm B: Mean-L¹ evolutionary strategy (Frechet.nb port) -----
    cfg = FrechetESConfig(
        popsize=args.es_popsize,
        generations=args.es_generations,
        sigma=args.es_sigma,
        sigma_start=args.es_sigma_start,
        best=args.es_best,
        n_islands=args.es_islands,
        sigma_t_mult=args.es_sigma_t_mult,
    )
    t_stage = time.time()
    print(f"\nRunning Frechet.nb mean-L¹ ES ({args.seeds_es} seeds, "
          f"popsize={cfg.popsize}, generations={cfg.generations}, "
          f"σ∈[{cfg.sigma}, {cfg.sigma_start or cfg.sigma}], "
          f"t-mult={cfg.sigma_t_mult}, best={cfg.best}, islands={cfg.n_islands})...")
    f_target = lambda t: np.sin(2 * PI * t * t)
    es_results = []
    for seed in range(args.seeds_es):
        tx = time.time()
        t_int, y_int, score = frechet_es(f_target, cfg=cfg, seed=seed,
                                          verbose=args.verbose)
        es_results.append((t_int, y_int, score, seed))
        print(f"  seed {seed}: best mean-L¹ = {score:.5e}  "
              f"t_interior = ({t_int[0]:.5f}, {t_int[1]:.5f}, {t_int[2]:.5f})  "
              f"({time.time()-tx:.1f}s; cumulative {time.time()-t_stage:.0f}s)")
    print(f"  (ES total: {time.time()-t_stage:.1f}s for {args.seeds_es} seeds)")

    best_es = min(es_results, key=lambda r: r[2])
    es_breakpoints = best_es[0]
    es_y = best_es[1]
    es_score = best_es[2]

    # ----- Diagnostic: mean-L¹ at each algebraic solution -----
    print("\nDiagnostic: mean-L¹ at each algebraic solution (lower = better path fit)")
    print(f"  Best ES mean-L¹: {es_score:.5e}")
    for idx, (theta_alg, _res) in enumerate(algebraic_solutions[:4]):
        alg_t, alg_y = _y_at_breakpoints(theta_alg)
        alg_score = meandist(f_target, alg_t, alg_y, n_sample=cfg.n_sample)
        ratio = alg_score / es_score if es_score > 0 else float('inf')
        print(f"  Algebraic Sol#{idx+1} mean-L¹: {alg_score:.5e}  "
              f"({ratio:.2f}× best ES)")

    # ----- Cross-seed agreement (only meaningful if multiple seeds converged) -----
    es_t_array = np.array([r[0] for r in es_results])
    es_scores = np.array([r[2] for r in es_results])
    # Filter seeds within 20% of the best score
    mask = es_scores <= 1.20 * es_scores.min()
    if mask.sum() >= 2:
        filt = es_t_array[mask]
        print(f"\nCross-seed convergence ({mask.sum()}/{len(es_results)} within 20% of best):")
        for i in range(3):
            vals = filt[:, i]
            span = vals.max() - vals.min()
            rel = span / max(abs(vals.mean()), 1e-12)
            print(f"  t{i+1}: mean={vals.mean():+.6f}, range={span:.2e}, rel={rel:.2e}")

    # ----- Comparison -----
    print("\n" + "=" * 72)
    print("Breakpoint agreement between algorithms")
    print("=" * 72)
    digits_per_solution = []
    for idx, (theta_alg, _res) in enumerate(algebraic_solutions[: min(4, len(algebraic_solutions))]):
        alg_breakpoints = np.array(theta_alg[:3])
        d = report_agreement(f"Algebraic Sol#{idx+1} vs best ES result:",
                              alg_breakpoints, es_breakpoints)
        digits_per_solution.append(d)

    print("\n" + "=" * 72)
    print("Verdict")
    print("=" * 72)
    best_match_digits = max(digits_per_solution) if digits_per_solution else 0.0
    target_digits = 4
    print(f"Best algebraic-solution-to-ES agreement: ~{best_match_digits:.1f} significant digits")
    print(f"Wilmott claim: 'four significant digits'  → "
          f"{'CONFIRMED' if best_match_digits >= target_digits else 'WEAKER than claimed'}")

    print(f"\nTotal wall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
