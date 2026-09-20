# from coresig.py -- imports: itertools, math, numpy as np, dataclass/field, scipy.optimize

def is_lyndon(w: tuple) -> bool:
    """w is Lyndon iff it is strictly smaller than every non-trivial rotation."""
    return all(w < w[k:] + w[:k] for k in range(1, len(w)))


def lyndon_words(m: int) -> list[tuple]:
    """Lyndon words on {1, 2} of length 1..m, level then lex order (the core index C_m)."""
    out = []
    for n in range(1, m + 1):
        out.extend(w for w in itertools.product((1, 2), repeat=n) if is_lyndon(w))
    return out


@dataclass
class Words:
    """All words of length 0..m, their Chen splits, and the core (Lyndon) index."""
    m: int
    all: list[tuple] = field(init=False)
    splits: dict = field(init=False)
    core: list[tuple] = field(init=False)

    def __post_init__(self):
        self.all = [()] + [w for n in range(1, self.m + 1)
                           for w in itertools.product((1, 2), repeat=n)]
        self.splits = {w: [(w[:j], w[j:]) for j in range(len(w) + 1)] for w in self.all}
        self.core = lyndon_words(self.m)

    def by_level(self, l: int) -> list[tuple]:
        return [w for w in self.core if len(w) == l]


def segment_signature(slope: float, dt: float, words: Words) -> dict:
    """sigma_w of the segment (1, slope) of duration dt: prod(slopes) dt^|w| / |w|!."""
    sig = {(): 1.0}
    for w in words.all[1:]:
        n2 = sum(1 for c in w if c == 2)
        sig[w] = slope ** n2 * dt ** len(w) / math.factorial(len(w))
    return sig


def chen(s1: dict, s2: dict, words: Words) -> dict:
    """Chen's identity, componentwise: z_w = sum_j x_{w[:j]} y_{w[j:]}."""
    return {w: sum(s1[u] * s2[v] for u, v in words.splits[w]) for w in words.all}


def piecewise_signature(breakpoints: Sequence[float], slopes: Sequence[float],
                        words: Words) -> dict:
    """Signature of the path with slope slopes[i] on (breakpoints[i], breakpoints[i+1])."""
    dts = np.diff(np.asarray(breakpoints, float))
    if len(dts) != len(slopes):
        raise ValueError("need one slope per interval")
    sig = segment_signature(float(slopes[0]), float(dts[0]), words)
    for b, dt in zip(slopes[1:], dts[1:]):
        sig = chen(sig, segment_signature(float(b), float(dt), words), words)
    return sig
