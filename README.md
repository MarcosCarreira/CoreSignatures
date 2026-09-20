# Core Signatures and Inversions

Marcos Costa Santos Carreira

Code, data and manuscripts for *Core Signatures and Inversions*: the core (Lyndon-word)
coordinates of a truncated path signature, the substitution rules that express every other
entry in them, and the inversion of core signatures into piecewise-linear paths, with
time-ordered term structures (interest-rate curves) as the application.

| Folder | Contents |
|---|---|
| `2023 version/` | the original 2023 paper and its companion material |
| `2026 version/` | the 2026 revision: the full preprint (28 pages, complete proofs and the substitution rules) and the version submitted to *Wilmott* in May 2026 (13 pages), with the two Mathematica notebooks they cite |
| `2026 R1/` | the *Wilmott* revision 1 (September 2026): the manuscript (23 pages), the mathematical supplement (6 pages: substitution rules, closed-form solutions, proofs), and the Python implementation with the data snapshots, figure scripts and printed figures |
| `Notes and Intermediate results/` | notebooks and verification scripts from the development of the method |

Each folder has its own `README.md` with file descriptions and build or run instructions.
The Python package in `2026 R1/python/` runs on NumPy and SciPy (pandas and Matplotlib for
the data and figure scripts); every script runs without arguments.
