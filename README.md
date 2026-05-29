# ToGit — Staging directory for GitHub upload

This folder mirrors the layout of the public GitHub repository for the
**Core Signatures and Inversions** paper. Two destinations:

- **`2026 version/`** — copy as a new top-level folder alongside the existing
  `2023 version/` in the repository. Contains the 2026 revision deliverables
  (arXiv-target full version and Wilmott magazine version) with sources,
  compiled PDFs, figures, and the two Mathematica notebooks cited as
  companion sources in the paper.
- **`Notes and Intermediate Results/`** — copy the contents into the existing
  `Notes and Intermediate Results/` folder in the repository. Contains the
  Python verification scripts and findings reports from the 2026 revision
  phase.

Each subfolder has its own `README.md` with build instructions, file
descriptions, and dependency notes. Nothing else outside these two
subfolders is intended for the public repository.

## Not included in this staging directory

The following 2026-phase artefacts are intentionally **excluded** as either
private correspondence or internal tracking:

- The Wilmott editor cover letter (private correspondence with the editor).
- The Reviewer #2 follow-up assessment reports (adversarial-LLM reviews
  used in revision; private to the author).
- The project card markdown (internal phase tracking).
- LyX intermediate files (`.lyx`) — anyone who wants LyX format can
  open the `.tex` files in LyX directly.
- The arXiv source tarball (`.tar.gz`) — redundant with the source files
  already inside `2026/`.
- The previous earlier-iteration verification scripts (`verify_algorithm.py`,
  `verify_algorithm_v2.py`, `verify_n4_wilmott.py`) — superseded by the
  current versions kept here.

If at some later point any of those exclusions should be reconsidered,
they live in `~/CoreSignaturesImproved/2026/` and
`~/CoreSignaturesImproved/Reviewer2/`.
