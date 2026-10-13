"""Tuning on the public tasks, in its simplest form: a team tries V variants against the benchmark and submits the one
with the highest score. Every variant is as able as the holder (selected-step null, fitted Verified model), the best of
the V is submitted, and the record test runs if it beats the holder. The selective p-value conditions on one
challenger having scored higher, not on the best of V, so its size grows with V. If V is declared, the Bonferroni
correction V * p_sel restores the level; the exact conditional law of the best of V would be sharper.

Reports, per V: share of submissions that set a record, size of the selective test among them, size of the
V-corrected test, and the same two under the global null of a 40-submission chain (every submission the best of V).
usage: python validate_private_tries.py [pairs]   -> results/validation_tries.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "honestboard", "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from honestboard import Board  # noqa: E402
import validate_record_test as vrt  # noqa: E402

PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
T = 500
path = os.path.join(ROOT, "data", "boards", "swebench_verified.json")
Lpool, base, sd_real, _ = vrt.fit_board(path)
rho = vrt.calibrate_rho(Lpool, base, sd_real)
a0 = vrt.solve_a(Lpool, base)


def p_sel(nb, n):
    nb, n = np.asarray(nb), np.asarray(n)
    nn = np.maximum(n, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where((n > 0) & (nb > n / 2), np.minimum(1.0, stats.binom.sf(nb - 1, nn, 0.5) / stats.binom.sf(nn // 2, nn, 0.5)), 1.0)
    return p


def selected_step(V, pairs, chunk=2000):
    """Holder not inflated; the challenger is the best of V equally able variants on the same tasks."""
    n_rec = rej = rej_c = n_tot = 0
    for start in range(0, pairs, chunk):
        r = min(chunk, pairs - start)
        Lt = vrt.rng.choice(Lpool, (r, T))
        ys = vrt.draw(Lt, [a0] * (V + 1), rho)
        ya = ys[0].astype(float)
        cand = np.stack([y.astype(float) for y in ys[1:]], 0)              # V x r x T
        best = np.argmax(cand.mean(2) + vrt.rng.random((V, r)) * 1e-9, axis=0)
        yb = cand[best, np.arange(r)]
        nb = ((yb == 1) & (ya == 0)).sum(1)
        na = ((ya == 1) & (yb == 0)).sum(1)
        sel = nb > na
        p = p_sel(nb[sel], (nb + na)[sel])
        n_tot += r
        n_rec += int(sel.sum())
        rej += int((p <= 0.05).sum())
        rej_c += int((np.minimum(1.0, V * p) <= 0.05).sum())
    return {"V": V, "submissions": n_tot, "record_share": n_rec / n_tot, "size_selective": rej / n_rec, "size_corrected": rej_c / n_rec}


def global_null(V, R, K=40):
    """A chain of K submissions, each the best of V equally able variants; false certifications per board."""
    fw = fw_c = steps = rej = rej_c = 0
    for _ in range(R):
        Lt = vrt.rng.choice(Lpool, T)
        rows = []
        for _k in range(K):
            ys = np.stack(vrt.draw(Lt, [a0] * V, rho), 0).astype(float)
            rows.append(ys[int(np.argmax(ys.mean(1) + vrt.rng.random(V) * 1e-9))])
        b = Board([f"s{k}" for k in range(K)], np.vstack(rows), [f"{20250101 + k}" for k in range(K)])
        a = b.record_audit()
        ps = np.array([s["p_selective"] for s in a["steps"]])
        steps += len(ps)
        rej += int((ps <= 0.05).sum())
        rej_c += int((np.minimum(1.0, V * ps) <= 0.05).sum())
        fw += bool((ps <= 0.05).any())
        fw_c += bool((np.minimum(1.0, V * ps) <= 0.05).any())
    return {"V": V, "boards": R, "steps_per_board": steps / R, "per_step_selective": rej / max(steps, 1),
            "per_step_corrected": rej_c / max(steps, 1), "fwer_selective": fw / R, "fwer_corrected": fw_c / R}


out = {"design": {"T": T, "rho": rho, "base": base, "pairs": PAIRS}, "selected": [], "global": []}
for V in (1, 2, 3, 5, 10):
    t0 = time.time()
    s = selected_step(V, PAIRS)
    g = global_null(V, 200)
    out["selected"].append(s)
    out["global"].append(g)
    print(f"V={V:2d}: records {100 * s['record_share']:.0f}% of submissions; size of the selective test {100 * s['size_selective']:.1f}%, "
          f"V-corrected {100 * s['size_corrected']:.1f}% | chain: per-step {100 * g['per_step_selective']:.1f}% / corrected {100 * g['per_step_corrected']:.1f}%, "
          f"FWER {100 * g['fwer_selective']:.1f}% / {100 * g['fwer_corrected']:.1f}% ({time.time() - t0:.0f}s)", flush=True)
with open(os.path.join(ROOT, "results", "validation_tries.json"), "w") as f:
    json.dump(out, f, indent=2)
