# from coresig.py -- imports: itertools, math, numpy as np, dataclass/field, scipy.optimize

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
