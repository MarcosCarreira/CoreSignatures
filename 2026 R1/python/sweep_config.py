"""
sweep_config.py -- configuration sweep for the Section 5 inversions.

For each curve and each (n segments, level m, minimum segment length), run the
variance-equalising weighted regression and record the pointwise fit error at
the market points (bp), the breakpoints, and the convergence count. Writes
  2026/_figs/sweep_config.csv   and   2026/_figs/sweep_config.tex
No arguments; prints one line per configuration with timing. Expected run time
5-8 minutes at N_STARTS = 12 (about 20 fits).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coresig import Words, signature_of_samples, variance_equalising_weights, weighted_regression, curve_path
from curves_load import load_di1, load_h15, curve_on

# --------------------------------------------------------------------------
# Settings (edit here)
# --------------------------------------------------------------------------
DI1_DATE = "2023-11-07"
H15_DATE = "2023-10-11"
CONFIGS = {                      # curve -> list of (n_segments, level m)
    "DI1":  [(4, 5), (4, 6), (5, 5), (5, 6)],
    "H.15": [(3, 4), (3, 5), (3, 6), (4, 5), (4, 6)],
}
DT_MIN_FRACTIONS = (0.05, 0.02)  # minimum segment length as a fraction of the path
N_STARTS = 12
SEED = 20260913
AXIS = "sqrt"
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_figs"))


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    curves = {"DI1": curve_on(load_di1(), DI1_DATE), "H.15": curve_on(load_h15(), H15_DATE)}
    dates = {"DI1": DI1_DATE, "H.15": H15_DATE}
    rows = []
    total = sum(len(v) for v in CONFIGS.values()) * len(DT_MIN_FRACTIONS)
    k = 0
    for name, (tau, rate) in curves.items():
        u, x, x0 = curve_path(tau, rate, axis=AXIS)
        for n, m in CONFIGS[name]:
            words = Words(m)
            target = signature_of_samples(u, x, words)
            W = variance_equalising_weights(target, words)
            n_res, n_unk = len(words.core) - 1, 2 * n - 1
            for frac in DT_MIN_FRACTIONS:
                k += 1; t1 = time.time()
                r = weighted_regression(target, n=n, words=words, weights=W, n_starts=N_STARTS,
                                        seed=SEED, dt_min=frac)
                fit = np.interp(u, r["breakpoints"], r["y_breakpoints"])
                e = 100.0 * (fit - x)
                bps = r["breakpoints"][1:-1]
                rows.append(dict(curve=name, date=dates[name], n=n, m=m, dt_min=frac,
                                 residuals=n_res, unknowns=n_unk,
                                 max_abs_bp=float(np.max(np.abs(e))), mean_abs_bp=float(np.mean(np.abs(e))),
                                 rms_bp=float(np.sqrt(np.mean(e ** 2))), cost=r["cost"],
                                 converged=r["n_starts_converged"],
                                 breakpoints=" ".join(f"{b:.3f}" for b in bps),
                                 at_bound=int(np.any(np.isclose(bps, frac, atol=1e-3)) or
                                              np.any(np.isclose(bps, 1 - frac, atol=1e-3)))))
                print(f"[{time.time() - t0:6.1f}s] {k:2d}/{total} {name:5s} n={n} m={m} dt_min={frac:.2f}: "
                      f"max {rows[-1]['max_abs_bp']:5.1f} bp, mean {rows[-1]['mean_abs_bp']:4.1f} bp, "
                      f"bp=[{rows[-1]['breakpoints']}], conv {r['n_starts_converged']}/{N_STARTS}"
                      f"{' (at bound)' if rows[-1]['at_bound'] else ''}  ({time.time() - t1:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "sweep_config.csv"), index=False)
    lines = [r"\begin{tabular}{@{}lrrrrrrrl@{}}", r"\toprule",
             r"Curve & $n$ & $m$ & $\Delta u_{\min}$ & res./unk. & max $|$err$|$ (bp) & mean $|$err$|$ (bp) & conv. & breakpoints ($u$) \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{r.curve} & {r.n} & {r.m} & {r.dt_min:.2f} & {r.residuals}/{r.unknowns} & {r.max_abs_bp:.1f} & "
                     f"{r.mean_abs_bp:.1f} & {r.converged}/{N_STARTS} & {r.breakpoints}{'*' if r.at_bound else ''} \\\\")
    lines += [r"\bottomrule", r"\multicolumn{9}{@{}l}{\footnotesize * a breakpoint sits at the minimum-segment-length bound.}", r"\end{tabular}"]
    with open(os.path.join(OUT_DIR, "sweep_config.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[{time.time() - t0:6.1f}s] written sweep_config.csv / sweep_config.tex to {OUT_DIR}")


if __name__ == "__main__":
    main()
