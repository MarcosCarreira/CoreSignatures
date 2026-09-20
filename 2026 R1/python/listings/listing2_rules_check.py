# check two substitution rules of Appendix A on the signature of a random path
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
