"""
make_listings.py -- extract the functions printed in the paper from coresig.py
into python/listings/*.py, so that the printed code is the code that runs.
No arguments. Rerun after editing coresig.py; then recompile the paper.
"""
import ast, os, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "coresig.py")
OUT = os.path.join(HERE, "listings")

def extract(names):
    src = open(SRC).read(); tree = ast.parse(src); lines = src.splitlines()
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            start = node.lineno - 1
            if node.decorator_list:
                start = node.decorator_list[0].lineno - 1
            found[node.name] = "\n".join(lines[start:node.end_lineno])
    missing = [n for n in names if n not in found]
    assert not missing, missing
    return "\n\n\n".join(found[n] for n in names) + "\n"

HEADER = "# from coresig.py -- imports: itertools, math, numpy as np, dataclass/field, scipy.optimize\n\n"

LISTINGS = {
    "listing1_words_chen.py": ["is_lyndon", "lyndon_words", "Words", "segment_signature", "chen", "piecewise_signature"],
    "listing3_two_piece.py": ["_two_piece_given_t1", "two_piece_inversion"],
    "listing4_regression.py": ["variance_equalising_weights", "theta_to_path", "breakpoint_values", "weighted_regression"],
}

LISTING2 = '''# check two substitution rules of Appendix A on the signature of a random path
import numpy as np
from coresig import Words, piecewise_signature

words = Words(4)
rng = np.random.default_rng(0)
bp = np.sort(np.concatenate([[0.0, 1.0], rng.uniform(0, 1, 3)]))     # 4 segments
s = piecewise_signature(bp, rng.normal(size=4), words)               # all 30 entries, levels 1-4
s1, s2, s12, s112, s122, s1122 = (s[w] for w in [(1,), (2,), (1, 2), (1, 1, 2), (1, 2, 2), (1, 1, 2, 2)])
print(s[(2, 1)] - (s1 * s2 - s12))                                   # level 2 rule: 0 to rounding
print(s[(1, 2, 1, 2)] - (s12 ** 2 / 2 - 2 * s1122))                  # level 4 rule: 0 to rounding
print(len(words.core), "core entries out of", len(words.all) - 1)    # 8 of 30
'''

LISTING5 = '''# Section 5: invert one date of the DI1 curve (fig_curve_inversion.py, abridged)
import numpy as np
from coresig import Words, curve_path, signature_of_samples, variance_equalising_weights, weighted_regression
from curves_load import load_di1, curve_on

tau, rate = curve_on(load_di1(), "2023-11-07")            # maturities (years), rates (%)
u, x, x0 = curve_path(tau, rate, axis="sqrt")             # u in [0,1] on the sqrt(tau) axis; x = rate - x0
words = Words(5)                                          # level 5: 14 core entries
target = signature_of_samples(u, x, words)                # signature of the linear interpolant
W = variance_equalising_weights(target, words)            # W_l = 1 / max_{|w|=l} sigma_w^2
res = weighted_regression(target, n=5, words=words, weights=W, n_starts=24, dt_min=0.02)
print(np.round(res["breakpoints"][1:-1], 3), res["cost"], res["n_starts_converged"])
'''

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn, names in LISTINGS.items():
        open(os.path.join(OUT, fn), "w").write(HEADER + extract(names))
    open(os.path.join(OUT, "listing2_rules_check.py"), "w").write(LISTING2)
    open(os.path.join(OUT, "listing5_curve.py"), "w").write(LISTING5)
    for fn in sorted(os.listdir(OUT)):
        print(fn, sum(1 for _ in open(os.path.join(OUT, fn))), "lines")
