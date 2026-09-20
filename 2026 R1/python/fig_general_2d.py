"""
fig_general_2d.py -- Figure 3 of Section 4 redone with the regression: general 2-D paths.

  (a) the semicircle (cos(pi t), sin(pi t)), t in [0,1]: 3- and 5-segment regressions at level 5
  (b) the digit 5: a 7-knot polygon and its 5-segment regression at level 5
Variance-equalising weights; knots free in both coordinates (weighted_regression_2d), with the
length penalty LENGTH_PENALTY that favours shorter polygons; panel (a) also shows the
5-segment result without the penalty, a self-crossing polygon with a smaller signature residual.
Writes 2026/_figs/fig4_general_2d.{png,pdf} and general_2d.tex (knots and costs).

Digit-5 knots: read from python/data/digit5_points.csv, else 2026/Market_Data/digit5_points.csv (two columns x,y, one knot per
row, no header) when that file exists -- export it from the notebook -- otherwise the
approximate knots below, read off the original figure, are used and the log says so.
No arguments. Run time a few minutes (three regressions, 24 starts each).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coresig import (Words, signature_of_curve_2d, piecewise_signature_2d, variance_equalising_weights,
                     weighted_regression_2d)

# --------------------------------------------------------------------------
# Settings (edit here)
# --------------------------------------------------------------------------
N_STARTS = 24
SEED = 20260913
LEVEL = 5
LENGTH_PENALTY = 0.03         # residual lambda * (path length / chord - 1); 0 = signature residual only
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "_figs"))
DIGIT5_CSV = os.path.join(HERE, "data", "digit5_points.csv")   # repository copy
if not os.path.exists(DIGIT5_CSV):                                # project tree: exported from the notebook
    DIGIT5_CSV = os.path.normpath(os.path.join(HERE, "..", "Market_Data", "digit5_points.csv"))
DIGIT5_APPROX = np.array([[0, 0], [20, -3], [62, 12], [80, 47], [38, 53], [36, 77], [64, 95], [100, 93]], float)
FIG_W = 7.4
FONT = 11
plt.rcParams.update({"font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
                     "legend.fontsize": FONT - 1, "mathtext.fontset": "cm"})
COLS = {"target": "#1f3b73", "n3": "#c0392b", "n5": "#2e8b57", "n5raw": "#7f8c8d"}


def log(msg, t0):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def main():
    t0 = time.time()
    os.makedirs(FIG_DIR, exist_ok=True)
    words = Words(LEVEL)
    notes = []
    fig, ax = plt.subplots(2, 1, figsize=(FIG_W, 11.0))

    # (a) semicircle
    tt = np.linspace(0, 1, 401)
    target = signature_of_curve_2d(lambda t: -np.pi * np.sin(np.pi * t), lambda t: np.pi * np.cos(np.pi * t), words)
    # the path starts at (1, 0); the regression works from the origin, so shift by the start point
    start = np.array([1.0, 0.0])
    W = variance_equalising_weights(target, words)
    ax[0].plot(np.cos(np.pi * tt), np.sin(np.pi * tt), color=COLS["target"], lw=1.6, label="semicircle")
    for n, key, lam, style, tag in ((5, "n5raw", 0.0, "--", "no length penalty"),
                                    (3, "n3", LENGTH_PENALTY, "-", f"length penalty {LENGTH_PENALTY}"),
                                    (5, "n5", LENGTH_PENALTY, "-", f"length penalty {LENGTH_PENALTY}")):
        log(f"semicircle: {n} segments, {tag}, {N_STARTS} starts", t0)
        r = weighted_regression_2d(target, n=n, words=words, weights=W, n_starts=N_STARTS, seed=SEED,
                                   length_penalty=lam, verbose=True)
        P = r["points"] + start
        dense = np.vstack([np.linspace(P[i], P[i + 1], 50) for i in range(len(P) - 1)])
        dist = np.abs(np.hypot(dense[:, 0], dense[:, 1]) - 1.0)
        ax[0].plot(P[:, 0], P[:, 1], style, marker="o", color=COLS[key], lw=1.3, ms=4,
                   label=f"{n} segments, {tag}: max distance to the arc {dist.max():.3f}, length/chord {r['length_ratio']:.2f}")
        notes.append(f"% semicircle n={n} lambda={lam}: knots {np.round(P, 4).tolist()}; cost {r['cost']:.2e}; "
                     f"max radial error {dist.max():.4f}; length/chord {r['length_ratio']:.4f}; conv {r['n_starts_converged']}/{N_STARTS}")
        log(f"  -> knots {np.round(P[1:-1], 3).tolist()}, cost {r['cost']:.2e}, max radial error {dist.max():.3f}, "
            f"length/chord {r['length_ratio']:.3f} (arc {np.pi / 2:.3f}), conv {r['n_starts_converged']}/{N_STARTS}", t0)
    ax[0].set_aspect("equal"); ax[0].set_title(f"(a) the semicircle and its 3- and 5-segment regressions at level {LEVEL}")
    ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$y$"); ax[0].grid(alpha=0.3)
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=FONT - 2)

    # (b) digit 5
    if os.path.exists(DIGIT5_CSV):
        Q = np.loadtxt(DIGIT5_CSV, delimiter=",", ndmin=2)
        src = "digit5_points.csv"
    else:
        Q = DIGIT5_APPROX; src = "APPROXIMATE knots read off the original figure"
    log(f"digit 5: {len(Q) - 1}-segment target ({src})", t0)
    Q0 = Q - Q[0]
    target5 = piecewise_signature_2d(Q0, words)
    W5 = variance_equalising_weights(target5, words)
    ax[1].plot(Q[:, 0], Q[:, 1], "-o", color=COLS["target"], lw=1.6, ms=4, label=f"digit 5, {len(Q) - 1} segments")
    log(f"digit 5: 5 segments, {N_STARTS} starts", t0)
    r = weighted_regression_2d(target5, n=5, words=words, weights=W5, n_starts=N_STARTS, seed=SEED,
                               length_penalty=LENGTH_PENALTY, verbose=True)
    P = r["points"] + Q[0]
    ax[1].plot(P[:, 0], P[:, 1], "-o", color=COLS["n5"], lw=1.3, ms=4,
               label=f"5-segment regression, length penalty {LENGTH_PENALTY}")
    notes.append(f"% digit5 ({src}): knots {np.round(P, 3).tolist()}; cost {r['cost']:.2e}; conv {r['n_starts_converged']}/{N_STARTS}")
    log(f"  -> knots {np.round(P[1:-1], 2).tolist()}, cost {r['cost']:.2e}, conv {r['n_starts_converged']}/{N_STARTS}", t0)
    ax[1].set_aspect("equal"); ax[1].set_title(f"(b) the digit 5 and its 5-segment regression at level {LEVEL}")
    ax[1].set_xlabel("$x$"); ax[1].set_ylabel("$y$"); ax[1].grid(alpha=0.3)
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)

    fig.tight_layout(h_pad=2.5)
    out = os.path.join(FIG_DIR, "fig4_general_2d")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(FIG_DIR, "general_2d.tex"), "w") as fh:
        fh.write("\n".join(notes) + "\n")
    log(f"figure written to {FIG_DIR}", t0)


if __name__ == "__main__":
    main()
