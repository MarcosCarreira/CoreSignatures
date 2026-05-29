# 2026 version — Core Signatures and Inversions, revised

This folder contains the 2026 revision of *Core Signatures and Inversions*,
in two parallel forms, together with the two Mathematica notebooks cited
as companion sources in the paper.

| File | Description | Size |
|---|---|---|
| `CoreSignaturesAndInversions_2026.tex` / `.pdf` | Full-version, arXiv-target preprint (28 pages) with complete proofs, all theorems, and the appendix of substitution rules | 1.5 MB |
| `CoreSignaturesAndInversions_Wilmott.tex` / `.pdf` | Compressed practitioner-facing version submitted to *Wilmott* magazine (13 pages); shares the same algorithmic content but omits the heavier algebra | 1.0 MB |
| `figures/` | The 11 PNG figures referenced by both `.tex` files (relative path) | — |
| `CoreSignatures202412.nb` | Companion Mathematica notebook (December 2024): Q₄, Q₅, Q₆ substitution rules; closed-form Chen-identity solutions for two- and three-piece inversion; `minim3lines` / `minim4lines` regression objectives; `chen4z4core` polynomial system for the four-piece inversion | 7 MB |
| `Frechet.nb` | Mean-L¹ evolutionary algorithm used as the comparison baseline in Section 4.5 (population 250, σ = 0.025, 50 generations). Despite the name, computes mean *L¹* distance — see Remark 10 in the paper | 0.9 MB |

The arXiv version is the **citable technical record** and is intended as
the companion preprint that the magazine article points to for full proofs.
The Wilmott version is the magazine-targeted exposition.

## Building from source

Either `.tex` file compiles to PDF with standard `pdflatex`:

```bash
pdflatex CoreSignaturesAndInversions_2026.tex
pdflatex CoreSignaturesAndInversions_2026.tex   # second pass for cross-refs
```

The same recipe works for the Wilmott version with the filename swapped.
Both files reference figures in `figures/` (relative path).

### Required LaTeX packages

All packages are standard in modern TeX Live distributions (verified on
TeX Live 2025 via arXiv's build pipeline):

`article`, `fontenc`, `inputenc`, `xcolor`, `colortbl`, `babel`, `cprotect`,
`float`, `url`, `amsmath`, `amssymb`, `amsthm`, `graphicx`, `geometry`,
`fancyhdr`, `hyperref`, `lastpage`, `breakcites`, `indentfirst`, `listings`,
`xurl`.

`breakcites` allows multi-citation line breaks; if your TeX Live doesn't
have it, comment out the corresponding `\usepackage{breakcites}` line and
the `\setlength{\emergencystretch}{3em}` directive will handle the worst
of the remaining justification issues.

## Companion notebooks

The two Mathematica notebooks cited as companion sources in the paper
live in this folder:

- **`CoreSignatures202412.nb`** is the algebraic-derivation notebook.
  Contains all the substitution-rule tables, the closed-form Chen system
  solutions for two and three pieces, the regression-objective definitions,
  and the four-piece `chen4z4core` polynomial system. Open in Mathematica
  14.1+ and evaluate top-to-bottom to reproduce the algebra.
- **`Frechet.nb`** is the mean-L¹ evolutionary algorithm. It is the
  *comparison baseline* against which Section 4.5 evaluates the
  signature-space regression — i.e. it is *not* the inversion algorithm
  itself, but the independent benchmark. Open in Mathematica and evaluate
  to reproduce the `meandist`, `genpts`, `mutate`, `next`, and
  `poplist = NestList[next, ..., generations]` skeleton of the evolutionary
  search.

Python ports of both algorithms are in the `Notes and Intermediate Results/`
folder of this repository, with documentation and dependency notes.

## Citation

If you cite this paper, the arXiv ID is the canonical reference. The BibTeX
entry will look like:

```bibtex
@misc{Carreira2026CoreSignatures,
  author       = {Carreira, Marcos Costa Santos},
  title        = {Core Signatures and Inversions},
  year         = {2026},
  eprint       = {XXXX.XXXXX},
  archivePrefix= {arXiv},
  primaryClass = {math.PR},
  url          = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

Replace `XXXX.XXXXX` with the actual arXiv identifier once available.
