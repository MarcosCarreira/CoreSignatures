"""
fig_curve_inversion.py -- Section 5 figures and table: two term structures
inverted into a few line segments from their core signature.

Produces (in 2026/_figs/, PNG + PDF):
  fig5_di1_inversion      DI1 curve on DI1_DATE: (a) (tau, rate) with market points;
                          (b) (sqrt(tau) normalised, rate) with the N_SEG_DI1-segment
                          inversion at level M_DI1 overlaid, breakpoints marked.
  fig5_h15_inversion      the same for the H.15 Treasury curve on H15_DATE.
  fig5_level_sweep        fit error (bp) against truncation level, both curves.
  fig5_axis_weights       target-magnitude weight schedule under the
                          (tau, x) and (sqrt(tau), x) axes, both curves, log scale.
and the LaTeX fragment 2026/_figs/results_curve_inversion.tex with the results table.

No arguments: full-quality defaults. `python fig_curve_inversion.py --quick`
uses fewer starts for a smoke test. Prints per-step progress with timings.
Figures: stacked panels, never side by side; legends outside the axes;
fonts near text size; each panel about 0.8 of the text width.
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
from coresig import (Words, signature_of_samples, variance_equalising_weights,
                     weighted_regression, curve_path, sqrt_time, breakpoint_values)
from curves_load import load_di1, load_h15, curve_on

# --------------------------------------------------------------------------
# Settings (edit here)
# --------------------------------------------------------------------------
DI1_DATE = "2023-11-07"        # the worked example (last date in the DI1 file; the QuantMinds-2023 date)
H15_DATE = "2023-10-11"        # last date in the local H.15 file (append the Fed row to use 2023-11-07)
N_SEG_DI1, M_DI1 = 5, 5        # five segments, level 5: 13 non-trivial core entries, 9 unknowns (sweep_config 2026-09-13)
N_SEG_H15, M_H15 = 4, 5        # four segments, level 5: 13 non-trivial core entries, 7 unknowns
SWEEP_LEVELS = {"DI1": (5, 6), "H.15": (4, 5, 6)}   # per curve; level 4 is underdetermined for n=5, exact for n=4
DT_MIN = 0.02                  # minimum segment length as a fraction of the path (sweep_config: 0.02 avoids bound-pinned breakpoints)
AXIS = "sqrt"                  # path axis for the inversion: 'sqrt' or 'linear'
N_STARTS_FULL = 24
N_STARTS_QUICK = 4
SEED = 20260913
FIG_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_figs"))
FIG_W = 7.4                    # inches; ~0.8 of the A4 text width when printed
FONT = 11

plt.rcParams.update({"font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
                     "legend.fontsize": FONT - 1, "mathtext.fontset": "cm"})
COL_MKT, COL_FIT, COL_AXIS2 = "#1f3b73", "#c0392b", "#7f8c8d"


def log(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# One curve: path, target, inversion, errors
# --------------------------------------------------------------------------
def invert_curve(tau, rate, n_seg, m, n_starts, axis=AXIS):
    """Invert a market curve; returns a dict with the path, fit and errors in bp."""
    u, x, x0 = curve_path(tau, rate, axis=axis)
    words = Words(m)
    target = signature_of_samples(u, x, words)
    W = variance_equalising_weights(target, words)
    res = weighted_regression(target, n=n_seg, words=words, weights=W, n_starts=n_starts, seed=SEED, dt_min=DT_MIN)
    bp, yb = res["breakpoints"], res["y_breakpoints"]
    fit_at_market = np.interp(u, bp, yb)                    # piecewise-linear fit at the market points
    err_bp = 100.0 * (fit_at_market - x)                    # rates in %, so x100 -> basis points
    return dict(u=u, x=x, x0=x0, tau=tau, rate=rate, words=words, target=target, weights=W,
                breakpoints=bp, y_breakpoints=yb, cost=res["cost"],
                max_abs_bp=float(np.max(np.abs(err_bp))), mean_abs_bp=float(np.mean(np.abs(err_bp))),
                converged=res["n_starts_converged"], n_starts=n_starts, n_seg=n_seg, m=m)


def weight_ranges(tau, rate, m=6):
    """Variance-equalising weights under the two axes; returns {axis: {level: W}}."""
    out = {}
    for axis in ("linear", "sqrt"):
        u, x, _ = curve_path(tau, rate, axis=axis)
        words = Words(m)
        out[axis] = variance_equalising_weights(signature_of_samples(u, x, words), words)
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_inversion(r, name, label, path):
    fig, ax = plt.subplots(2, 1, figsize=(FIG_W, 8.6))
    ax[0].plot(r["tau"], r["rate"], "o-", color=COL_MKT, ms=4, lw=1.0, label="market points")
    ax[0].set_xlabel(r"maturity $\tau$ (years)"); ax[0].set_ylabel(f"{label} (% p.a.)")
    ax[0].set_title(rf"(a) {name}, {r['date']}: the $(\tau, x)$ view")
    ax[1].plot(r["u"], r["x0"] + r["x"], "o", color=COL_MKT, ms=4, label="market points")
    ax[1].plot(r["breakpoints"], r["x0"] + r["y_breakpoints"], "-", color=COL_FIT, lw=1.6,
               label=f"{r['n_seg']}-segment inversion, level {r['m']}")
    ax[1].plot(r["breakpoints"][1:-1], r["x0"] + r["y_breakpoints"][1:-1], "s", color=COL_FIT, ms=6,
               label="breakpoints")
    ax[1].set_xlabel(r"$u=(\sqrt{\tau}-\sqrt{\tau_0})/(\sqrt{\tau_{\max}}-\sqrt{\tau_0})$")
    ax[1].set_ylabel(f"{label} (% p.a.)")
    ax[1].set_title(rf"(b) the $(\sqrt{{\tau}}, x)$ path and its inversion "
                    rf"(max $|$error$|$ {r['max_abs_bp']:.1f} bp, mean {r['mean_abs_bp']:.1f} bp)")
    # maturity gridlines on the sqrt axis
    for t_mark in (0.25, 1, 2, 5, 10):
        if r["tau"][0] <= t_mark <= r["tau"][-1]:
            um = float(np.interp(t_mark, r["tau"], r["u"]))
            ax[1].axvline(um, color=COL_AXIS2, lw=0.5, ls=":")
            ax[1].text(um, ax[1].get_ylim()[0], f"{t_mark:g}y", fontsize=FONT - 2, color=COL_AXIS2,
                       ha="center", va="bottom")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=200, bbox_inches="tight"); fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)


def fig_sweep(sweep, path):
    fig, ax = plt.subplots(2, 1, figsize=(FIG_W, 7.2))
    for k, (name, rows) in enumerate(sweep.items()):
        lv = [r["m"] for r in rows]
        ax[k].plot(lv, [r["max_abs_bp"] for r in rows], "o-", color=COL_FIT, label="max |error|")
        ax[k].plot(lv, [r["mean_abs_bp"] for r in rows], "s--", color=COL_MKT, label="mean |error|")
        ax[k].set_xticks(lv); ax[k].set_xlabel("truncation level $m$"); ax[k].set_ylabel("fit error at market points (bp)")
        ax[k].set_title(f"({'ab'[k]}) {name}: {rows[0]['n_seg']} segments, error against level")
        ax[k].grid(alpha=0.3); ax[k].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=200, bbox_inches="tight"); fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)


def fig_weights(wr, path):
    fig, ax = plt.subplots(2, 1, figsize=(FIG_W, 7.2))
    for k, (name, w) in enumerate(wr.items()):
        for axis, col, mk in (("linear", COL_AXIS2, "o-"), ("sqrt", COL_FIT, "s-")):
            lv = sorted(w[axis]); ax[k].semilogy(lv, [w[axis][l] for l in lv], mk, color=col,
                                                 label=(r"$(\tau,x)$ axis" if axis == "linear" else r"$(\sqrt{\tau},x)$ axis"))
        rng_lin = max(w["linear"].values()) / min(w["linear"].values())
        rng_sq = max(w["sqrt"].values()) / min(w["sqrt"].values())
        ax[k].set_xticks(lv); ax[k].set_xlabel("level $l$"); ax[k].set_ylabel(r"$W_l / W_1$")
        ax[k].set_title(f"({'ab'[k]}) {name}: target-magnitude weights; range "
                        f"{rng_lin:.1e} vs {rng_sq:.1e} ({rng_lin / rng_sq:.0f}x tamer)")
        ax[k].grid(alpha=0.3, which="both"); ax[k].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=200, bbox_inches="tight"); fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)


def write_table(main, sweep, wr, path):
    lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r"Curve & Date & $n$ & $m$ & max $|$err$|$ (bp) & mean $|$err$|$ (bp) & breakpoints ($u$) & starts conv. \\", r"\midrule"]
    for name, r in main.items():
        bps = ", ".join(f"{b:.3f}" for b in r["breakpoints"][1:-1])
        lines.append(f"{name} & {r['date']} & {r['n_seg']} & {r['m']} & {r['max_abs_bp']:.1f} & {r['mean_abs_bp']:.1f} & {bps} & {r['converged']}/{r['n_starts']} \\\\")
    lines.append(r"\midrule")
    for name, rows in sweep.items():
        for r in rows:
            lines.append(f"{name} & {r['date']} & {r['n_seg']} & {r['m']} & {r['max_abs_bp']:.1f} & {r['mean_abs_bp']:.1f} & -- & {r['converged']}/{r['n_starts']} \\\\")
    lines.append(r"\midrule")
    for name, w in wr.items():
        rl = max(w['linear'].values()) / min(w['linear'].values()); rs = max(w['sqrt'].values()) / min(w['sqrt'].values())
        lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{{name}: weight range levels 1--6, $(\tau,x)$ {rl:.2e}, $(\sqrt\tau,x)$ {rs:.2e}, ratio {rl / rs:.0f}}} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
def main():
    quick = "--quick" in sys.argv
    n_starts = N_STARTS_QUICK if quick else N_STARTS_FULL
    
    t0 = time.time()
    os.makedirs(FIG_DIR, exist_ok=True)
    log(f"loading curves ({'quick' if quick else 'full'} mode, {n_starts} starts)", t0)
    di1, h15 = load_di1(), load_h15()
    curves = {"DI1": (curve_on(di1, DI1_DATE), DI1_DATE, N_SEG_DI1, M_DI1, "DI1 rate"),
              "H.15": (curve_on(h15, H15_DATE), H15_DATE, N_SEG_H15, M_H15, "Treasury yield")}
    main_res, sweep, wr = {}, {}, {}
    for name, ((tau, rate), date, n_seg, m, label) in curves.items():
        log(f"{name} {date}: {len(tau)} points, tau {tau[0]:.3f}-{tau[-1]:.1f}y, rate {rate.min():.2f}-{rate.max():.2f}%", t0)
        r = invert_curve(tau, rate, n_seg, m, n_starts); r["date"] = date
        main_res[name] = r
        log(f"  {n_seg} segments, level {m}: max |err| {r['max_abs_bp']:.1f} bp, mean {r['mean_abs_bp']:.1f} bp, "
            f"breakpoints u={np.round(r['breakpoints'][1:-1], 3)}, {r['converged']}/{n_starts} starts converged", t0)
        fig_inversion(r, name, label, os.path.join(FIG_DIR, f"fig5_{name.replace('.', '').lower()}_inversion"))
        rows = []
        for lv in (SWEEP_LEVELS[name][:2] if quick else SWEEP_LEVELS[name]):
            rr = invert_curve(tau, rate, n_seg, lv, n_starts); rr["date"] = date
            rows.append(rr)
            log(f"  sweep level {lv}: max |err| {rr['max_abs_bp']:.1f} bp, mean {rr['mean_abs_bp']:.1f} bp", t0)
        sweep[name] = rows
        wr[name] = weight_ranges(tau, rate)
        rl = max(wr[name]['linear'].values()) / min(wr[name]['linear'].values())
        rs = max(wr[name]['sqrt'].values()) / min(wr[name]['sqrt'].values())
        log(f"  weight range levels 1-6: (tau,x) {rl:.2e}, (sqrt tau,x) {rs:.2e}, ratio {rl / rs:.0f}", t0)
    fig_sweep(sweep, os.path.join(FIG_DIR, "fig5_level_sweep"))
    fig_weights(wr, os.path.join(FIG_DIR, "fig5_axis_weights"))
    write_table(main_res, sweep, wr, os.path.join(FIG_DIR, "results_curve_inversion.tex"))
    log(f"figures and table written to {FIG_DIR}", t0)


if __name__ == "__main__":
    main()
