"""Size of the record test among SELECTED steps (those where the challenger beat the holder), under a null where the
holder's score is not luck-inflated: holder and challenger have exactly the same true ability.

Reports the rejection rate among record steps for the exact one-sided and two-sided paired tests.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(1)
SD = 1.5


def solve_a(L, target):
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + L)))).mean() < target:
            lo = mid
        else:
            hi = mid
    return mid


for T in (300, 500, 2000):
    rec = rej1 = rej2 = 0
    R = 40000
    L = rng.normal(0, SD, (R, T))
    a = solve_a(L[0], 0.5)
    P = 1 / (1 + np.exp(-(a + L)))
    ya = rng.random((R, T)) < P
    yb = rng.random((R, T)) < P
    nb = (yb & ~ya).sum(1)
    na = (ya & ~yb).sum(1)
    is_rec = nb > na
    for x, m in zip(nb[is_rec], (nb + na)[is_rec]):
        rec += 1
        rej1 += stats.binomtest(int(x), int(m), 0.5, alternative="greater").pvalue <= 0.05
        rej2 += stats.binomtest(int(x), int(m), 0.5).pvalue <= 0.05
    print(f"T={T}: record steps {rec}; rejection among record steps: one-sided {rej1 / rec:.3f}, two-sided {rej2 / rec:.3f}")
