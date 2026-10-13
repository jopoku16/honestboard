"""Diagnostic: exact enumeration of Proposition 1 on the three-system, five-task board, printing the conditional
distribution of the selective p-value given the holder event and the record event, for several holder events."""
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "honestboard", "src"))
from honestboard import selective_sign_pvalue  # noqa: E402

T = 5
q = [0.3, 0.5, 0.7, 0.4, 0.6]
s = {1: [0.40, 0.30, 0.50, 0.35, 0.45], 0: [0.10, 0.15, 0.20, 0.05, 0.12]}
r = {1: [0.20, 0.25, 0.15, 0.22, 0.18], 0: [0.15, 0.20, 0.10, 0.25, 0.20]}
outcomes = [(yc, ya, yb) for yc in (0, 1) for ya in (0, 1) for yb in (0, 1)]


def prob(t, yc, ya, yb, b_factor):
    pc = q[t] if yc else 1 - q[t]
    ra, rb = r[yc][t], b_factor * r[yc][t]
    joint = {(1, 1): s[yc][t], (1, 0): ra, (0, 1): rb, (0, 0): 1 - s[yc][t] - ra - rb}
    return pc * joint[(ya, yb)]


def certified(yc, ya, level):
    n_a_only = sum(1 for c, a in zip(yc, ya) if a == 1 and c == 0)
    n_c_only = sum(1 for c, a in zip(yc, ya) if c == 1 and a == 0)
    return n_a_only > n_c_only and selective_sign_pvalue(n_a_only, n_c_only) <= level


EVENTS = {"record (total)": lambda yc, ya: sum(ya) > sum(yc),
          "certified at 0.3": lambda yc, ya: certified(yc, ya, 0.3),
          "certified at 0.5": lambda yc, ya: certified(yc, ya, 0.5),
          "certified at 1.0 (= beats c)": lambda yc, ya: certified(yc, ya, 1.0),
          "none": lambda yc, ya: True}

for name, ev in EVENTS.items():
    for b_factor in (1.0,):
        mass, by_n = {}, {}
        total = 0.0
        for config in itertools.product(outcomes, repeat=T):
            p_cfg = 1.0
            for t, o in enumerate(config):
                p_cfg *= prob(t, *o, b_factor)
            yc = [o[0] for o in config]; ya = [o[1] for o in config]
            n_b = sum(1 for o in config if o[2] == 1 and o[1] == 0)
            n_a = sum(1 for o in config if o[1] == 1 and o[2] == 0)
            if ev(yc, ya) and n_b > n_a:
                pv = selective_sign_pvalue(n_b, n_a)
                mass[pv] = mass.get(pv, 0.0) + p_cfg
                total += p_cfg
                key = (n_a + n_b, n_b)
                by_n[key] = by_n.get(key, 0.0) + p_cfg
        worst = max((sum(m for pv, m in mass.items() if pv <= u) / total - u, u) for u in np.linspace(0.01, 1.0, 100))
        print(f"{name:30s} mass {total:.4f}  worst excess {worst[0]:+.4f} at u={worst[1]:.2f}  p-values: " +
              ", ".join(f"{pv:.3f}:{m / total:.3f}" for pv, m in sorted(mass.items())))
        # conditional law of n_b given n, compared with the fair-coin law restricted to the record event
        for n in sorted({k[0] for k in by_n}):
            tot_n = sum(v for k, v in by_n.items() if k[0] == n)
            obs = {k[1]: v / tot_n for k, v in by_n.items() if k[0] == n}
            from scipy import stats
            fair = {k: stats.binom.pmf(k, n, 0.5) / stats.binom.sf(n // 2, n, 0.5) for k in range(n // 2 + 1, n + 1)}
            print(f"    n={n}: P(n_b | n, event, record) " + " ".join(f"{k}:{obs.get(k, 0):.3f}" for k in sorted(fair)) +
                  "   fair " + " ".join(f"{k}:{v:.3f}" for k, v in sorted(fair.items())))
