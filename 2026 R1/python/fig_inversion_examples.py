"""
fig_inversion_examples.py -- the Section 4 example figures, regenerated with coresig.py.

  fig4_two_piece.{png,pdf}    three stacked panels: two-segment inversions (Listing 3) of
                              sin(pi t), sin(pi t^2), and sin(pi t^2) + 1 - t (both solutions)
  fig4_multi_piece.{png,pdf}  two stacked panels: 4- and 5-segment regressions (Listing 4)
                              of sin(pi t) and sin(2 pi t) at level 5, variance-equalising weights
Written to 2026/_figs/. No arguments. Run time about ten minutes (four regressions, 24 starts each).
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
from coresig import (Words, signature_of_smooth, two_piece_inversion, variance_equalising_weights,
                     weighted_regression, piecewise_signature, core_vector)

# --------------------------------------------------------------------------
# Settings (edit here)
# --------------------------------------------------------------------------
N_STARTS = 24
SEED = 20260913
LEVEL = 5
FIG_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_figs"))
FIG_W = 7.4
FONT = 11
plt.rcParams.update({"font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
                     "legend.fontsize": FONT - 1, "mathtext.fontset": "cm"})
COLS = {"target": "#1f3b73", "sol1": "#c0392b", "sol2": "#8e6b23", "n4": "#c0392b", "n5": "#2e8b57"}


def log(msg, t0):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


TARGETS_2P = [
    (r"$\sin(\pi t)$", lambda t: np.sin(np.pi * t), lambda t: np.pi * np.cos(np.pi * t)),
    (r"$\sin(\pi t^{2})$", lambda t: np.sin(np.pi * t ** 2), lambda t: 2 * np.pi * t * np.cos(np.pi * t ** 2)),
    (r"$\sin(\pi t^{2})+1-t$", lambda t: np.sin(np.pi * t ** 2) + 1 - t,
     lambda t: 2 * np.pi * t * np.cos(np.pi * t ** 2) - 1),
]
TARGETS_NP = [
    (r"$\sin(\pi t)$", lambda t: np.sin(np.pi * t), lambda t: np.pi * np.cos(np.pi * t)),
    (r"$\sin(2\pi t)$", lambda t: np.sin(2 * np.pi * t), lambda t: 2 * np.pi * np.cos(2 * np.pi * t)),
]


def tent(root, f0):
    """Breakpoints and values of a two-segment solution starting at f(0)."""
    t = np.array([0.0, root["t1"], root["T"]])
    y = f0 + np.array([0.0, root["beta1"] * root["t1"], root["beta1"] * root["t1"] + root["beta2"] * (root["T"] - root["t1"])])
    return t, y


def fig_two_piece(t0):
    words = Words(3)
    tt = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(3, 1, figsize=(FIG_W, 12.5))
    for k, (name, f, fp) in enumerate(TARGETS_2P):
        s = signature_of_smooth(f, fp, words)
        s = {w: (s[w] if w != () else 1.0) for w in s}
        f0 = float(f(0.0))
        ax[k].plot(tt, f(tt), color=COLS["target"], lw=1.6, label=f"target {name}")
        for drop, key, lab in ((5, "sol1", "Solution 1 (drops eq. 5)"), (4, "sol2", "Solution 2 (drops eq. 4)")):
            roots = two_piece_inversion(s, drop=drop)
            roots = [r for r in roots if 0 < r["t1"] < r["T"]]
            for j, r in enumerate(roots):
                t, y = tent(r, f0)
                ax[k].plot(t, y, "-o", color=COLS[key], lw=1.3, ms=4,
                           label=(lab + rf": $t_1$={r['t1']:.4f}, peak {y[1]:.4f}") if j == 0 else None)
                log(f"{name}: drop {drop}: t1 {r['t1']:.6f}, y1 {y[1]:.6f}, residual of dropped eq {r['residual_dropped']:.2e}", t0)
            if k < 2 and drop == 4:
                pass
        ax[k].set_title(f"({'abc'[k]}) two-segment inversion of {name}")
        ax[k].set_xlabel("$t$"); ax[k].set_ylabel("$f(t)$"); ax[k].grid(alpha=0.3)
        ax[k].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False)
    fig.tight_layout(h_pad=2.5)
    out = os.path.join(FIG_DIR, "fig4_two_piece")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)


def fig_multi_piece(t0):
    words = Words(LEVEL)
    tt = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(2, 1, figsize=(FIG_W, 8.2))
    rows = []
    for k, (name, f, fp) in enumerate(TARGETS_NP):
        target = signature_of_smooth(f, fp, words)
        W = variance_equalising_weights(target, words)
        ax[k].plot(tt, f(tt), color=COLS["target"], lw=1.6, label=f"target {name}")
        for n, key in ((4, "n4"), (5, "n5")):
            r = weighted_regression(target, n=n, words=words, weights=W, n_starts=N_STARTS, seed=SEED)
            fit = np.interp(tt, r["breakpoints"], r["y_breakpoints"])
            err = np.mean(np.abs(fit - f(tt)))
            ax[k].plot(r["breakpoints"], r["y_breakpoints"], "-o", color=COLS[key], lw=1.3, ms=4,
                       label=f"{n} segments: mean $|$error$|$ {err:.3f}")
            rows.append((name, n, r["breakpoints"][1:-1], r["cost"], err, r["n_starts_converged"]))
            log(f"{name}: n={n}: breakpoints {np.round(r['breakpoints'][1:-1], 4)}, cost {r['cost']:.2e}, "
                f"mean |err| {err:.4f}, conv {r['n_starts_converged']}/{N_STARTS}", t0)
        ax[k].set_title(f"({'ab'[k]}) 4- and 5-segment regressions of {name} at level {LEVEL}")
        ax[k].set_xlabel("$t$"); ax[k].set_ylabel("$f(t)$"); ax[k].grid(alpha=0.3)
        ax[k].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    fig.tight_layout(h_pad=2.5)
    out = os.path.join(FIG_DIR, "fig4_multi_piece")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(FIG_DIR, "inversion_examples.tex"), "w") as fh:
        for name, n, bp, cost, err, conv in rows:
            fh.write(f"% {name} n={n}: breakpoints {' '.join(f'{b:.4f}' for b in bp)}; cost {cost:.2e}; "
                     f"mean abs err {err:.4f}; conv {conv}/{N_STARTS}\n")


def main():
    t0 = time.time()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_two_piece(t0)
    log("two-piece figure written", t0)
    fig_multi_piece(t0)
    log(f"multi-piece figure written to {FIG_DIR}", t0)


if __name__ == "__main__":
    main()
