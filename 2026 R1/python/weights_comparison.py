"""
weights_comparison.py -- three weight schedules, same targets, pointwise error in path space.

Schedules: 'working'  = the level-only values {50,50,30,10,10,3,3,3} of the first experiments
           'variance' = variance-equalising, W_l = 1 / max_{|w|=l} sigma_w^2 (Section 4.5)
           'uniform'  = W_l = 1
Targets:   sin(pi t), sin(2 pi t), sin(pi t^2) at n = 4 and 5 segments, level 5;
           the DI1 curve of 2023-11-07 (n = 5) and the H.15 curve of 2023-10-11 (n = 4), level 5.
For each fit: mean and max pointwise |error| (function targets: on a 401-point grid, in the
target's units; curves: at the market points, in bp), signature-space cost, breakpoints,
starts converged. Writes 2026/_figs/weights_comparison.{csv,tex}.
No arguments. 30 fits at N_STARTS = 12, one line per start. A start under the working or uniform
schedule often runs to the iteration cap (about 17 s) because those schedules leave the problem
badly scaled; expect 40-70 minutes in total, most of it on those two schedules.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coresig import (Words, signature_of_smooth, signature_of_samples, variance_equalising_weights,
                     working_weights, weighted_regression, curve_path)
from curves_load import load_di1, load_h15, curve_on

# --------------------------------------------------------------------------
# Settings (edit here)
# --------------------------------------------------------------------------
N_STARTS = 12
SEED = 20260913
LEVEL = 5
DT_MIN = 0.02
SCHEDULES = ("working", "variance", "uniform")
FUNCTIONS = [
    ("sin(pi t)", lambda t: np.sin(np.pi * t), lambda t: np.pi * np.cos(np.pi * t), (4, 5)),
    ("sin(2 pi t)", lambda t: np.sin(2 * np.pi * t), lambda t: 2 * np.pi * np.cos(2 * np.pi * t), (4, 5)),
    ("sin(pi t^2)", lambda t: np.sin(np.pi * t ** 2), lambda t: 2 * np.pi * t * np.cos(np.pi * t ** 2), (4, 5)),
]
CURVES = [("DI1", "2023-11-07", 5), ("H.15", "2023-10-11", 4)]
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_figs"))


def log(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def schedule(name, target, words):
    if name == "working":
        return working_weights(words)
    if name == "variance":
        return variance_equalising_weights(target, words)
    return {l: 1.0 for l in range(1, words.m + 1)}


def fit(target, n, words, W, t0, label):
    log(f"{label}: {N_STARTS} starts", t0)
    r = weighted_regression(target, n=n, words=words, weights=W, n_starts=N_STARTS, seed=SEED,
                            dt_min=DT_MIN, verbose=True)
    return r


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    words = Words(LEVEL)
    rows = []
    tt = np.linspace(0, 1, 401)
    for name, f, fp, ns in FUNCTIONS:
        target = signature_of_smooth(f, fp, words)
        for n in ns:
            for sched in SCHEDULES:
                W = schedule(sched, target, words)
                r = fit(target, n, words, W, t0, f"{name} n={n} {sched}")
                g = np.interp(tt, r["breakpoints"], r["y_breakpoints"])
                e = np.abs(g - f(tt))
                rows.append(dict(target=name, n=n, schedule=sched, mean_err=e.mean(), max_err=e.max(),
                                 cost=r["cost"], converged=r["n_starts_converged"],
                                 breakpoints=" ".join(f"{b:.3f}" for b in r["breakpoints"][1:-1])))
                log(f"  -> mean |err| {e.mean():.4f}, max {e.max():.4f}, bp [{rows[-1]['breakpoints']}], "
                    f"conv {r['n_starts_converged']}/{N_STARTS}", t0)
    loaders = {"DI1": load_di1, "H.15": load_h15}
    for name, date, n in CURVES:
        tau, rate = curve_on(loaders[name](), date)
        u, x, x0 = curve_path(tau, rate, axis="sqrt")
        target = signature_of_samples(u, x, words)
        for sched in SCHEDULES:
            W = schedule(sched, target, words)
            r = fit(target, n, words, W, t0, f"{name} {date} n={n} {sched}")
            g = np.interp(u, r["breakpoints"], r["y_breakpoints"])
            e = 100.0 * np.abs(g - x)
            rows.append(dict(target=f"{name} {date}", n=n, schedule=sched, mean_err=e.mean(), max_err=e.max(),
                             cost=r["cost"], converged=r["n_starts_converged"],
                             breakpoints=" ".join(f"{b:.3f}" for b in r["breakpoints"][1:-1])))
            log(f"  -> mean |err| {e.mean():.1f} bp, max {e.max():.1f} bp, bp [{rows[-1]['breakpoints']}], "
                f"conv {r['n_starts_converged']}/{N_STARTS}", t0)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "weights_comparison.csv"), index=False)
    lines = [r"\begin{tabular}{@{}llrrrrl@{}}", r"\toprule",
             r"Target & $n$ & weights & mean $|$err$|$ & max $|$err$|$ & conv. & breakpoints \\", r"\midrule"]
    for _, r in df.iterrows():
        unit = " bp" if r.target.startswith(("DI1", "H.15")) else ""
        fmt = (lambda v: f"{v:.1f}") if unit else (lambda v: f"{v:.4f}")
        lines.append(f"{r.target} & {r.n} & {r.schedule} & {fmt(r.mean_err)}{unit} & {fmt(r.max_err)}{unit} & "
                     f"{r.converged}/{N_STARTS} & {r.breakpoints} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT_DIR, "weights_comparison.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"written weights_comparison.csv / .tex to {OUT_DIR}", t0)


if __name__ == "__main__":
    main()
