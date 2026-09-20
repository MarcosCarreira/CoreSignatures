"""
coresig.py -- core signatures of time-augmented paths, and their inversion.

Companion module to "Core Signatures and Inversions" (M. C. S. Carreira,
Wilmott; full version SSRN, DOI 10.2139/ssrn.6879202). Self-contained:
numpy + scipy only. Alphabet {1, 2}: letter 1 is time, letter 2 is the
value coordinate of X(t) = (t, f(t)).

Contents
--------
Words                      Lyndon words (the core index) and Chen splits, level <= m
segment_signature          closed form for one line segment, eq. (3) of the paper
chen                       Chen's identity, componentwise, eq. (5)
piecewise_signature        signature of a piecewise-linear path
signature_of_samples       signature of the linear interpolant of (t_i, y_i)
core_vector                the core entries as a vector, in level-then-lex order
two_piece_inversion        the two solutions of the two-segment Chen system
variance_equalising_weights   W_l = 1 / max_{|w|=l} |sigma_w|^2, normalised W_1 = 1
weighted_regression        n-segment inversion as weighted nonlinear least squares
sqrt_time, curve_path      the sqrt(tau) axis for term structures

Conventions
-----------
A piecewise-linear path on [0, T] with n segments is parametrised by
theta = (t_1, ..., t_{n-1}, beta_1, ..., beta_n): interior breakpoints and
segment slopes. The path starts at (0, 0); the level y(t) is stored
separately (the signature sees only dy).

Run the file to execute the self-test (a few seconds).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import brentq, differential_evolution, least_squares

# --------------------------------------------------------------------------
# Settings (edit here, not in the functions)
# --------------------------------------------------------------------------
WORKING_WEIGHTS = {1: 50.0, 2: 50.0, 3: 30.0, 4: 10.0, 5: 10.0, 6: 3.0, 7: 3.0, 8: 3.0}
DEFAULT_STARTS = 24          # multi-start count for the local solver
DEFAULT_SEED = 20260913
DT_MIN_FRACTION = 0.05      # minimum segment length as a fraction of T (avoids empty segments)
N_GRID_SMOOTH = 40_000       # grid for the signature of a smooth target


# --------------------------------------------------------------------------
# 1. Words: the core index
# --------------------------------------------------------------------------
def is_lyndon(w: tuple) -> bool:
    """w is Lyndon iff it is strictly smaller than every non-trivial rotation."""
    return all(w < w[k:] + w[:k] for k in range(1, len(w)))


def lyndon_words(m: int) -> list[tuple]:
    """Lyndon words on {1, 2} of length 1..m, level then lex order (the core index C_m)."""
    out = []
    for n in range(1, m + 1):
        out.extend(w for w in itertools.product((1, 2), repeat=n) if is_lyndon(w))
    return out


@dataclass
class Words:
    """All words of length 0..m, their Chen splits, and the core (Lyndon) index."""
    m: int
    all: list[tuple] = field(init=False)
    splits: dict = field(init=False)
    core: list[tuple] = field(init=False)

    def __post_init__(self):
        self.all = [()] + [w for n in range(1, self.m + 1)
                           for w in itertools.product((1, 2), repeat=n)]
        self.splits = {w: [(w[:j], w[j:]) for j in range(len(w) + 1)] for w in self.all}
        self.core = lyndon_words(self.m)

    def by_level(self, l: int) -> list[tuple]:
        return [w for w in self.core if len(w) == l]


# --------------------------------------------------------------------------
# 2. Segment signature and Chen's identity
# --------------------------------------------------------------------------
def segment_signature(slope: float, dt: float, words: Words) -> dict:
    """sigma_w of the segment (1, slope) of duration dt: prod(slopes) dt^|w| / |w|!."""
    sig = {(): 1.0}
    for w in words.all[1:]:
        n2 = sum(1 for c in w if c == 2)
        sig[w] = slope ** n2 * dt ** len(w) / math.factorial(len(w))
    return sig


def chen(s1: dict, s2: dict, words: Words) -> dict:
    """Chen's identity, componentwise: z_w = sum_j x_{w[:j]} y_{w[j:]}."""
    return {w: sum(s1[u] * s2[v] for u, v in words.splits[w]) for w in words.all}


def piecewise_signature(breakpoints: Sequence[float], slopes: Sequence[float],
                        words: Words) -> dict:
    """Signature of the path with slope slopes[i] on (breakpoints[i], breakpoints[i+1])."""
    dts = np.diff(np.asarray(breakpoints, float))
    if len(dts) != len(slopes):
        raise ValueError("need one slope per interval")
    sig = segment_signature(float(slopes[0]), float(dts[0]), words)
    for b, dt in zip(slopes[1:], dts[1:]):
        sig = chen(sig, segment_signature(float(b), float(dt), words), words)
    return sig


def signature_of_samples(t: np.ndarray, y: np.ndarray, words: Words) -> dict:
    """Signature of the linear interpolant of the samples (t_i, y_i)."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    return piecewise_signature(t, np.diff(y) / np.diff(t), words)


def signature_of_smooth(f: Callable, fp: Callable, words: Words, T: float = 1.0,
                        N: int = N_GRID_SMOOTH) -> dict:
    """Signature of X(t) = (t, f(t)) by iterated trapezoid integration (fp = f')."""
    t = np.linspace(0.0, T, N + 1); dt = t[1] - t[0]
    fpv = np.asarray(fp(t), float)
    h = {(): np.ones_like(t)}
    for w in words.all[1:]:
        integrand = h[w[:-1]] * (1.0 if w[-1] == 1 else fpv)
        cum = np.zeros_like(t)
        cum[1:] = np.cumsum((integrand[:-1] + integrand[1:]) / 2) * dt
        h[w] = cum
    return {w: float(h[w][-1]) for w in words.all}


def core_vector(sig: dict, words: Words) -> np.ndarray:
    """The core entries of a signature dict as a vector (level-then-lex order)."""
    return np.array([sig[w] for w in words.core], float)


# --------------------------------------------------------------------------
# 3. Two segments: the closed system of Table 3
# --------------------------------------------------------------------------
def _two_piece_given_t1(t1: float, z: dict, T: float) -> tuple[float, float]:
    """Solve equations 2 and 3 (levels 1-2) for the slopes, given the breakpoint."""
    a = T - t1
    A = np.array([[t1, a], [0.5 * t1 ** 2, 0.5 * a ** 2 + t1 * a]])
    b1, b2 = np.linalg.solve(A, [z[(2,)], z[(1, 2)]])
    return float(b1), float(b2)


def two_piece_inversion(target: dict, drop: int = 5, n_scan: int = 400) -> list[dict]:
    """
    Two-segment inversion of a level-3 core signature {z1, z2, z12, z112, z122}.

    T = z1 fixes the duration; equations 2-3 fix the slopes for each breakpoint
    t1; the remaining level-3 equation (4 or 5, the other one being `drop`) is
    solved for t1 by a bracketing scan. Returns every root found, as dicts with
    keys t1, beta1, beta2, T, and the residual of the dropped equation.
    """
    words = Words(3)
    T = target[(1,)]
    keep = (1, 1, 2) if drop == 5 else (1, 2, 2)
    dropped = (1, 2, 2) if drop == 5 else (1, 1, 2)

    def resid(t1):
        b1, b2 = _two_piece_given_t1(t1, target, T)
        s = piecewise_signature([0.0, t1, T], [b1, b2], words)
        return s[keep] - target[keep]

    grid = np.linspace(1e-6 * T, T * (1 - 1e-6), n_scan)
    vals = np.array([resid(g) for g in grid])
    roots = []
    for i in range(len(grid) - 1):
        if np.sign(vals[i]) != np.sign(vals[i + 1]):
            t1 = brentq(resid, grid[i], grid[i + 1], xtol=1e-14)
            b1, b2 = _two_piece_given_t1(t1, target, T)
            s = piecewise_signature([0.0, t1, T], [b1, b2], words)
            roots.append(dict(t1=t1, beta1=b1, beta2=b2, T=T,
                              residual_dropped=s[dropped] - target[dropped]))
    return roots


# --------------------------------------------------------------------------
# 4. Weights
# --------------------------------------------------------------------------
def variance_equalising_weights(target: dict, words: Words, floor: float = 0.1) -> dict:
    """
    W_l = 1 / max_{|w|=l} |sigma_w[X]|^2, normalised so that W_1 = 1.
    A level whose entries all vanish (a symmetric target: sigma_12 = 0 for sin(2 pi t))
    would get an infinite weight; its scale is floored at `floor` times the size
    A^l / l! that a path of length scale A has at level l, with A read off the target.
    """
    A = max((abs(target[w]) * math.factorial(len(w))) ** (1.0 / len(w)) for w in words.core)
    W = {}                                              # A: the path's length scale, entries ~ A^l / l!
    for l in range(1, words.m + 1):
        mx = max(abs(target[w]) for w in words.by_level(l))
        W[l] = 1.0 / max(mx, floor * A ** l / math.factorial(l)) ** 2
    scale = W[1]
    return {l: W[l] / scale for l in W}


def working_weights(words: Words) -> dict:
    """The level-only working values of the paper's Section 4.3 (decreasing with level)."""
    return {l: WORKING_WEIGHTS[l] for l in range(1, words.m + 1)}


# --------------------------------------------------------------------------
# 5. n segments: weighted nonlinear least squares in signature space
# --------------------------------------------------------------------------
def theta_to_path(theta: np.ndarray, n: int, T: float) -> tuple[np.ndarray, np.ndarray]:
    """theta = (t_1..t_{n-1}, beta_1..beta_n) -> breakpoints [0, t_1, .., T], slopes."""
    t_int = np.sort(np.asarray(theta[:n - 1], float))
    return np.concatenate([[0.0], t_int, [T]]), np.asarray(theta[n - 1:], float)


def breakpoint_values(theta: np.ndarray, n: int, T: float) -> np.ndarray:
    """y at the breakpoints (path starts at 0): cumulative slope x duration."""
    bp, sl = theta_to_path(theta, n, T)
    return np.concatenate([[0.0], np.cumsum(sl * np.diff(bp))])


def weighted_regression(target: dict, n: int, words: Words, weights: dict | None = None,
                        n_starts: int = DEFAULT_STARTS, seed: int = DEFAULT_SEED,
                        global_search: bool = False, slope_bound: float | None = None,
                        dt_min: float | None = None, verbose: bool = False) -> dict:
    """
    Invert a core signature into an n-segment path by weighted least squares:
        theta* = argmin sum_w W_|w| (sigma_w[g_theta] - z_w)^2  over Lyndon w, |w| <= m.
    weights: {level: W_l}; default variance-equalising from the target (Section 4.5).
    Multi-start local solver (scipy least_squares); global_search=True runs
    differential evolution first and polishes the best point locally.
    dt_min: minimum segment length (default DT_MIN_FRACTION * T); shorter
    segments are penalised so the solver cannot empty a segment.
    Returns dict: theta, breakpoints, slopes, y_breakpoints, cost, n_starts_converged.
    """
    T = target[(1,)]
    W = weights if weights is not None else variance_equalising_weights(target, words)
    sw = np.array([math.sqrt(W[len(w)]) for w in words.core])
    z = core_vector(target, words)
    rng = np.random.default_rng(seed)
    if slope_bound is None:
        slope_bound = 10.0 * max(1.0, abs(target[(2,)]) / T,
                                 6.0 * abs(target[(1, 2)]) / T ** 2)
    if dt_min is None:
        dt_min = DT_MIN_FRACTION * T
    scale = float(np.sqrt(np.sum((sw * z) ** 2))) or 1.0

    def residuals(theta):
        bp, sl = theta_to_path(theta, n, T)
        dts = np.diff(bp)
        if np.any(dts <= 0):
            return 1e3 * np.ones_like(z)
        s = piecewise_signature(bp, sl, words)
        r = sw * (core_vector(s, words) - z)
        short = np.clip(dt_min - dts, 0.0, None)          # penalty for segments shorter than dt_min
        r[0] += scale * float(np.sum(short)) / dt_min
        return r

    lo = np.concatenate([dt_min * np.ones(n - 1), -slope_bound * np.ones(n)])
    hi = np.concatenate([(T - dt_min) * np.ones(n - 1), slope_bound * np.ones(n)])

    def random_start():
        t_int = np.sort(rng.uniform(dt_min, T - dt_min, n - 1))
        while np.any(np.diff(np.concatenate([[0.0], t_int, [T]])) < dt_min):
            t_int = np.sort(rng.uniform(dt_min, T - dt_min, n - 1))
        sl = rng.normal(target[(2,)] / T, 0.5 * slope_bound / 3, n)
        return np.concatenate([t_int, np.clip(sl, -slope_bound, slope_bound)])

    starts = [random_start() for _ in range(n_starts)]
    if global_search:
        de = differential_evolution(lambda th: float(np.sum(residuals(th) ** 2)),
                                    list(zip(lo, hi)), seed=seed, tol=1e-10,
                                    maxiter=300, popsize=25, polish=False)
        starts = [de.x] + starts
        if verbose:
            print(f"    DE cost {de.fun:.3e}", flush=True)

    best, converged = None, 0
    for k, th0 in enumerate(starts):
        try:
            r = least_squares(residuals, th0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14,
                              gtol=1e-14, max_nfev=4000)
        except Exception:
            continue
        if r.success:
            converged += 1
        if best is None or r.cost < best.cost:
            best = r
        if verbose:
            print(f"    start {k + 1:2d}/{len(starts)}: cost {r.cost:.3e}{'' if r.success else ' (not converged)'}"
                  f"  best so far {best.cost:.3e}", flush=True)
    bp, sl = theta_to_path(best.x, n, T)
    return dict(theta=best.x, breakpoints=bp, slopes=sl,
                y_breakpoints=breakpoint_values(best.x, n, T),
                cost=float(best.cost), n_starts_converged=converged, weights=W)


# --------------------------------------------------------------------------
# 5b. General two-dimensional paths (not time-ordered): knots free in both coordinates
# --------------------------------------------------------------------------
def segment_signature_2d(dx: float, dy: float, words: Words) -> dict:
    """sigma_w of the straight segment with increment (dx, dy): dx^|w|_1 dy^|w|_2 / |w|!."""
    sig = {(): 1.0}
    for w in words.all[1:]:
        n2 = sum(1 for c in w if c == 2)
        sig[w] = dx ** (len(w) - n2) * dy ** n2 / math.factorial(len(w))
    return sig


def piecewise_signature_2d(points: np.ndarray, words: Words) -> dict:
    """Signature of the polygonal path through the rows of points (shape (k+1, 2))."""
    P = np.asarray(points, float)
    d = np.diff(P, axis=0)
    sig = segment_signature_2d(float(d[0, 0]), float(d[0, 1]), words)
    for dx, dy in d[1:]:
        sig = chen(sig, segment_signature_2d(float(dx), float(dy), words), words)
    return sig


def signature_of_curve_2d(xp: Callable, yp: Callable, words: Words, T: float = 1.0,
                          N: int = N_GRID_SMOOTH) -> dict:
    """Signature of the parametric path (x(t), y(t)) from its derivatives xp = x', yp = y'."""
    t = np.linspace(0.0, T, N + 1); dt = t[1] - t[0]
    dv = {1: np.asarray(xp(t), float), 2: np.asarray(yp(t), float)}
    h = {(): np.ones_like(t)}
    for w in words.all[1:]:
        integrand = h[w[:-1]] * dv[w[-1]]
        cum = np.zeros_like(t)
        cum[1:] = np.cumsum((integrand[:-1] + integrand[1:]) / 2) * dt
        h[w] = cum
    return {w: float(h[w][-1]) for w in words.all}


def weighted_regression_2d(target: dict, n: int, words: Words, weights: dict | None = None,
                           n_starts: int = DEFAULT_STARTS, seed: int = DEFAULT_SEED,
                           box: float = 1.5, length_penalty: float = 0.0, verbose: bool = False) -> dict:
    """
    Invert a core signature into an n-segment polygonal path in the plane, start at the
    origin and end at (z_1, z_2) (the level-1 entries), the n-1 interior knots free in
    both coordinates: theta = (x_1, y_1, ..., x_{n-1}, y_{n-1}), 2(n-1) unknowns against
    the core entries of level >= 2. Random starts inside a box of half-width
    `box` x the chord length around the chord's midpoint. Without time-ordering the
    truncated signature admits far-away polygons with nearly the target's entries
    (long self-crossing excursions cancel); `length_penalty` adds the residual
    length_penalty * (path length / chord - 1), which favours shorter polygons.
    Segment durations do not enter the objective: the fit matches the planar
    signature of the ordered vertices and returns no traversal times.
    Returns dict: points (n+1, 2), theta, cost, n_starts_converged,
    weights, length_ratio.
    """
    W = weights if weights is not None else variance_equalising_weights(target, words)
    core = [w for w in words.core if len(w) >= 2]
    sw = np.array([math.sqrt(W[len(w)]) for w in core])
    z = np.array([target[w] for w in core])
    end = np.array([target[(1,)], target[(2,)]])
    L = max(float(np.hypot(*end)), 1e-12)
    rng = np.random.default_rng(seed)

    def points_of(theta):
        return np.vstack([[0.0, 0.0], np.asarray(theta, float).reshape(n - 1, 2), end])

    def length_ratio(theta):
        return float(np.sum(np.hypot(*np.diff(points_of(theta), axis=0).T))) / L

    def residuals(theta):
        s = piecewise_signature_2d(points_of(theta), words)
        r = sw * (np.array([s[w] for w in core]) - z)
        if length_penalty > 0:
            r = np.append(r, length_penalty * (length_ratio(theta) - 1.0))
        return r

    centre = end / 2
    lo = np.tile(centre - box * L, n - 1); hi = np.tile(centre + box * L, n - 1)
    best, converged = None, 0
    for k in range(n_starts):
        th0 = rng.uniform(lo, hi)
        try:
            r = least_squares(residuals, th0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=4000)
        except Exception:
            continue
        if r.success:
            converged += 1
        if best is None or r.cost < best.cost:
            best = r
        if verbose:
            print(f"    start {k + 1:2d}/{n_starts}: cost {r.cost:.3e}{'' if r.success else ' (not converged)'}"
                  f"  best so far {best.cost:.3e}", flush=True)
    return dict(points=points_of(best.x), theta=best.x, cost=float(best.cost),
                n_starts_converged=converged, weights=W, length_ratio=length_ratio(best.x))


# --------------------------------------------------------------------------
# 6. Term structures: the sqrt(tau) axis
# --------------------------------------------------------------------------
def sqrt_time(tau: np.ndarray) -> np.ndarray:
    """u = (sqrt(tau) - sqrt(tau_0)) / (sqrt(tau_max) - sqrt(tau_0)), in [0, 1]."""
    r = np.sqrt(np.asarray(tau, float))
    return (r - r[0]) / (r[-1] - r[0])


def curve_path(tau: np.ndarray, rate: np.ndarray, axis: str = "sqrt") -> tuple[np.ndarray, np.ndarray, float]:
    """
    A term structure as a time-augmented path starting at the origin.
    Returns (u, x - x0, x0): the axis in [0, 1] ('sqrt' or 'linear'), the rate
    relative to its first value, and the first value (stored separately).
    """
    tau = np.asarray(tau, float); rate = np.asarray(rate, float)
    if axis == "sqrt":
        u = sqrt_time(tau)
    elif axis == "linear":
        u = (tau - tau[0]) / (tau[-1] - tau[0])
    else:
        raise ValueError("axis must be 'sqrt' or 'linear'")
    return u, rate - rate[0], float(rate[0])


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def _self_test():
    import time
    t0 = time.time()
    w6 = Words(6)
    assert [len(w6.by_level(l)) for l in range(1, 7)] == [2, 1, 2, 3, 6, 9], "Witt counts"
    assert w6.core[:8] == [(1,), (2,), (1, 2), (1, 1, 2), (1, 2, 2),
                           (1, 1, 1, 2), (1, 1, 2, 2), (1, 2, 2, 2)]
    print("core index: 23 Lyndon words at m=6, per level 2,1,2,3,6,9  ok")

    # sin(pi t) on [0,1]: core signature {1, 0, -2/pi, -1/pi, 1/4} (paper, Section 4.1)
    w3 = Words(3)
    s = signature_of_smooth(np.sin, lambda t: np.pi * np.cos(np.pi * t), w3)
    ref = {(1,): 1.0, (2,): 0.0, (1, 2): -2 / np.pi, (1, 1, 2): -1 / np.pi, (1, 2, 2): 0.25}
    err = max(abs(s[w] - ref[w]) for w in ref)
    assert err < 1e-8, err
    print(f"sin(pi t) core signature matches the paper to {err:.1e}  ok")

    # two-piece inversion: symmetric tent with peak (1/2, 4/pi)
    roots = two_piece_inversion(s, drop=5)
    peak = [(r["t1"], r["beta1"] * r["t1"]) for r in roots]
    ok = any(abs(t1 - 0.5) < 1e-9 and abs(y - 4 / np.pi) < 1e-9 for t1, y in peak)
    assert ok, peak
    print(f"two-piece inversion of sin(pi t): peak {peak[0][0]:.6f}, {peak[0][1]:.6f} = (1/2, 4/pi)  ok")

    # variance-equalising weights on a smooth curve increase with level
    w5 = Words(5)
    s5 = signature_of_smooth(lambda t: np.sin(np.pi * t ** 2),
                             lambda t: 2 * np.pi * t * np.cos(np.pi * t ** 2), w5)
    W = variance_equalising_weights(s5, w5)
    assert all(W[l + 1] > W[l] for l in range(1, 5)), W
    print("variance-equalising weights on sin(pi t^2): " +
          ", ".join(f"W{l}={W[l]:.3g}" for l in W) + "  (increasing)  ok")

    # n = 4 regression at m = 5 (overdetermined: 14 core entries, 7 unknowns)
    res = weighted_regression(s5, n=4, words=w5, n_starts=8)
    print(f"4-segment inversion of sin(pi t^2) at level 5: cost {res['cost']:.2e}, "
          f"breakpoints {np.round(res['breakpoints'][1:-1], 4)}, "
          f"{res['n_starts_converged']}/8 starts converged")

    # exact recovery: a 3-segment path inverted at m = 4 (8 entries, 5 unknowns)
    w4 = Words(4)
    bp_true = [0.0, 0.3, 0.7, 1.0]; sl_true = [1.5, -0.8, 0.4]
    tgt = piecewise_signature(bp_true, sl_true, w4)
    res3 = weighted_regression(tgt, n=3, words=w4, n_starts=12)
    e_bp = np.max(np.abs(res3["breakpoints"] - bp_true)); e_sl = np.max(np.abs(res3["slopes"] - sl_true))
    assert e_bp < 1e-6 and e_sl < 1e-6, (res3["breakpoints"], res3["slopes"])
    print(f"exact recovery of a 3-segment path at level 4: max error {max(e_bp, e_sl):.1e}  ok")

    # sqrt axis
    u = sqrt_time(np.array([0.25, 1, 4, 9]))
    assert np.allclose(u, [0, 0.2, 0.6, 1.0]), u
    print("sqrt(tau) axis ok")
    print(f"self-test done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    _self_test()
