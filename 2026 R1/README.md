# Core Signatures and Inversions — Wilmott revision 1 (September 2026)

Companion package for the revised manuscript *Core Signatures and Inversions*
(Wilmott, MS WP-1259, revision 1). Release `wilmott-r1-2026-09`.

## Contents

| Path | What it is |
|---|---|
| `CoreSignaturesAndInversions_Wilmott_R1cut_2026-09-20.pdf` | the manuscript (23 pages) and its source `.tex` |
| `CoreSignaturesAndInversions_Wilmott_R1_Supplement_2026-09-20.pdf` | the mathematical supplement (6 pages) and its source: S1 the 48 substitution rules through level 5, S2 the closed-form two-segment solutions and the three-segment matching system, S3 the proof of Proposition 1, S4 the full proofs of Theorems 7, 9 and 14 |
| `python/coresig.py` | the module: binary word index and Lyndon words (`Words`), segment signature, Chen product and piecewise-linear signature, two-segment inversion (`two_piece_inversion`), target-magnitude weights (`variance_equalising_weights`, legacy name), multistart weighted regression for time-ordered paths (`weighted_regression`) and for planar paths (`weighted_regression_2d`), curve path construction (`curve_path`) |
| `python/curves_load.py` | loaders for the two curve snapshots |
| `python/data/di1_source_snapshot.csv`, `h15_source_snapshot.csv` | the DI1 curve of 7 November 2023 (B3) and the US Treasury constant-maturity curve of 11 October 2023 (Federal Reserve H.15) used in Section 5 |
| `python/data/digit5_points.csv` | the eight knots (seven segments) of the digit-5 path of Figure 3(b) |
| `python/listings/listing1..5_*.py` | the five short scripts behind the displayed calls: words and Chen product, rule check, two-segment inversion, multistart regression, one DI1 inversion call |
| `python/fig_inversion_examples.py` | Figures 1–2 (two-segment inversions; four- and five-segment regressions) — about ten minutes |
| `python/fig_general_2d.py` | Figure 3 (semicircle and digit 5) — a few minutes |
| `python/fig_curve_inversion.py` | Figures 4–5 and the results table of Section 5 (it also writes the level-sweep and axis-weights figures, not printed) — about fifteen minutes at 24 starts |
| `python/sweep_config.py`, `python/weights_comparison.py` | the segment/level sweep (Table 5) and the weights comparison (Table 4); `_figs/*.csv` hold their saved outputs |
| `python/make_listings.py` | regenerates `listings/` from `coresig.py` |
| `_figs/` | the figures as printed and the LaTeX fragments the scripts write |

## Running

```
pip install -r python/requirements.txt
cd python
python listings/listing2_rules_check.py     # seconds
python fig_general_2d.py                    # minutes; writes ../_figs/fig4_general_2d.*
python fig_curve_inversion.py               # ~15 min; writes ../_figs/fig5_*.* and results_curve_inversion.tex
```

Every script runs without arguments; settings are constants at the top of each file. The
scripts print progress with elapsed times and write to `_figs/` next to `python/`.
Dependencies: NumPy and SciPy for the module; pandas and Matplotlib for the data and figure
scripts (`python/requirements.txt`).

## Notes

- `variance_equalising_weights` keeps its name for compatibility with earlier runs; it
  implements the target-magnitude schedule of Section 4.5.
- `weighted_regression_2d` matches the planar signature of the ordered vertices; segment
  durations do not enter its objective and it returns no traversal times (Section 4.3).
- Results in the paper are those of the runs recorded in `_figs/*.tex`; multistart searches are
  seeded (`SEED` in each script), and a start meeting the solver's stopping criteria is not
  thereby certified as a local minimum.
