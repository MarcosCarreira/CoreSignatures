# Notes and Intermediate Results — 2026 additions

This folder contains two kinds of material added during the 2026 revision
phase:

1. **Python verification scripts** (with findings reports) that port the
   Mathematica notebooks and let anyone reproduce the empirical claims of
   the paper from a standard Python environment.
2. **Intermediate-vintage Mathematica and Jupyter notebooks** preserved as
   version history between the 2023 baseline and the December 2024
   companion notebook now in `2026 version/`.

## Python verification scripts (2026 phase)

| File | Purpose |
|---|---|
| `verify_algorithm_n2.py` | Verifies Algorithm 1 of the paper at **n=2 segments** on three test targets: a synthetic in-class target (exact recovery), `sin(πt)` (out of class), and `sin(πt²) + 1 − t` (multi-solution case). Documents reproducibility of the inversion |
| `verify_algorithm_n2_REPORT.md` | Findings markdown for the n=2 verification |
| `wilmott_n4_comparison.py` | **PyCharm-runnable script** comparing the algebraic Chen-system inversion to the mean-L¹ evolutionary algorithm at **n=4 segments**, target `sin(2πt²)`. Both algorithms ported in a single file; reports breakpoint agreement and mean-L¹ at each algebraic solution |
| `wilmott_n4_comparison_README.md` | Usage doc, interpretation guide, and a recommended-defaults summary for the n=4 comparison script |

## Intermediate notebook versions (version history)

The companion notebook now in `2026 version/CoreSignatures202412.nb`
(December 2024) is the latest revision. The intermediate Mathematica and
Jupyter versions below are preserved here as a record of how the
computational artefact evolved between the 2023 baseline and that
December 2024 reference.

| File | Date | Format | Notes |
|---|---|---|---|
| `CoreSignatures_202312.ipynb` | Dec 2023 | Jupyter | First Jupyter port of the 2023 paper's computational content |
| `CoreSignatures_202404.nb` | Apr 2024 | Mathematica | Previous main revision of the inversion notebook; superseded by the December 2024 version in `2026 version/` |
| `CoreSignatures_202405.ipynb` | May 2024 | Jupyter | Updated Jupyter port reflecting the April-2024 Mathematica state |

The original 2023 notebook (`CoreSignatures.ipynb`, November 2023) lives
in `2023 version/` of this repository, not here. The current
December 2024 companion notebook lives in `2026 version/`.

## How to run the Python scripts

### Requirements

- Python 3.9+
- `numpy >= 1.20`
- `scipy >= 1.7`

```bash
pip install numpy scipy
```

### Verifying Algorithm 1 at n=2

```bash
python verify_algorithm_n2.py
```

Runs three tests in under a minute. Tests A (synthetic in-class) passes
to machine precision; Tests B and C illustrate that the regression
optimum differs from the closed-form algebraic solutions for out-of-class
smooth targets.

### Verifying the Wilmott n=4 claim

```bash
python wilmott_n4_comparison.py            # full Frechet.nb baseline (~5 min)
python wilmott_n4_comparison.py --quick    # fast smoke test (~30 s, under-converged)
python wilmott_n4_comparison.py --frechet-original  # exactly reproduce Frechet.nb settings
```

The default settings are the high-effort verification configuration
(population 250, generations 200, σ-schedule, multiple seeds, multi-start
algebraic solver). See the script's `--help` and the README for tuning.

### Headline empirical finding (relevant to Section 4.5 of the paper)

The signature-space regression and the mean-L¹ evolutionary algorithm
converge to **distinct nearby minima**, not to a common optimum. On the
target `sin(2πt²)` at n=4: the algebraic Chen-system root has residual
norm 8 × 10⁻¹⁶ but mean-L¹ ≈ 4.99 × 10⁻²; the ES converges to a
breakpoint configuration ~14% off in t₁ with mean-L¹ ≈ 4.51 × 10⁻²
(11% lower). The two algorithms minimise structurally different
objectives; the breakpoints agree only to within a few percent. This
is the multi-segment analogue of the single-segment 4.7% obstruction
(Remark 13 in the paper).

## Citation context

If you use these scripts to extend the verification (e.g. on different
targets, different segment counts, or under the GA weight-calibration
program flagged in Remark 12 of the paper's conjecture), please cite the
companion arXiv paper and consider opening a pull request to share your
findings.
