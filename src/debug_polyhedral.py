"""Why is the polyhedral step p-value anti-conservative in heterogeneous chains? Isolate plug-in covariance,
heterogeneity and discreteness. Prints P(p <= 0.05) among NULL record steps for several variants."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "honestboard", "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from honestboard import Board  # noqa: E402
from honestboard.chain import polyhedral_step_pvalue, selective_sign_pvalue  # noqa: E402
import validate_record_test as vrt  # noqa: E402

Lpool, base, sd_real, _ = vrt.fit_board(os.path.join(ROOT, "data", "boards", "swebench_verified.json"))
rho = vrt.calibrate_rho(Lpool, base, sd_real)
base = 0.60
K, T, R = 40, 500, 150


def offsets_late():
    return np.array([vrt.solve_a(Lpool, base if k < K // 2 else base + 0.06) for k in range(K)])


def offsets_null():
    return np.full(K, vrt.solve_a(Lpool, base))


def true_sigma(offsets, n=20000):
    """Covariance of the K score means under task sampling, from a large simulation."""
    Lt = vrt.rng.choice(Lpool, n)
    Y = np.vstack(vrt.draw(Lt, list(offsets), rho)).astype(float)
    return np.cov(Y) / T


def collect(offsets_fn, sigma_mode, pval="poly", lattice=0.0):
    hits = tot = 0
    for r in range(R):
        offsets = offsets_fn()
        theta = np.array([(1 / (1 + np.exp(-(a + Lpool)))).mean() for a in offsets])
        Lt = vrt.rng.choice(Lpool, T)
        Y = np.vstack(vrt.draw(Lt, list(offsets), rho))
        b = Board([f"s{k}" for k in range(K)], Y)
        sc = b.scores()
        sigma = b.sigma() if sigma_mode == "plugin" else true_sigma(offsets)
        holder = 0
        for pos in range(1, K):
            k = pos
            if sc[k] > sc[holder] + 1e-12:
                if theta[k] <= theta[holder] + 1e-12:         # null step
                    if pval == "poly":
                        p = polyhedral_step_pvalue(sc, sigma, k, holder, list(range(pos)), lattice=lattice)
                    else:
                        ya, yb = b.Y[holder], b.Y[k]
                        p = selective_sign_pvalue(int(((yb == 1) & (ya == 0)).sum()), int(((ya == 1) & (yb == 0)).sum()))
                    tot += 1
                    hits += p <= 0.05
                holder = k
    return hits / max(tot, 1), tot


for name, fn in (("null", offsets_null), ("late", offsets_late)):
    for mode in ("plugin", "true"):
        for lat in (0.0, 1.0 / T):
            rate, n = collect(fn, mode, "poly", lattice=lat)
            print(f"{name:5s} polyhedral, sigma={mode:6s}, lattice={'1/T' if lat else '0  '}: P(p<=0.05 | null step) = {rate:.3f}  (n={n})", flush=True)
    rate, n = collect(fn, "plugin", "exact")
    print(f"{name:5s} exact selective                      : P(p<=0.05 | null step) = {rate:.3f}  (n={n})", flush=True)
