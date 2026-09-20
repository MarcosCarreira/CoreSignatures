# Section 5: invert one date of the DI1 curve (fig_curve_inversion.py, abridged)
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
