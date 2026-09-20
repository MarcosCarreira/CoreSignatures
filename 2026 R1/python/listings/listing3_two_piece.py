# from coresig.py -- imports: itertools, math, numpy as np, dataclass/field, scipy.optimize

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
