"""Does the sequential certification procedure hold its error rate, and what power does it have?

Simulated from the model fitted to SWE-bench Verified (see validate_record_test.py): K submissions in sequence,
true abilities by scenario, copula correlation rho between systems on a task.

Scenarios
  null        all K submissions equally able; every record is luck
  staircase   true score rises by `step` points every `period` submissions (within a level all equal), so records
              are a mixture of genuine jumps and lucky steps within a level
  late        equal for the first half, then one genuine jump and equal again (one true record late in the chain)
  variant     all equally able, but half of the submissions after the first are perturbations of a random earlier
              submission (its outcomes with passes flipped to failures at rate e1 and failures to passes at the rate
              that keeps the true score equal): the case in which the task-level null of Proposition 1 fails while
              the population-score null holds, so the copula simulation cannot produce it

Procedures (per-step p-value x level rule), all with the leaderboard holder as baseline unless noted:
  unadjusted (two-sided exact at alpha), holm (retrospective, realised chain), spending (alpha 6/(pi^2 i^2)),
  lord (LORD++), each with the exact selective p-value; lord and spending also with the polyhedral p-value;
  lord with the certified baseline.

Metrics: FWER (any false certification on a board), FDR, power (share of genuine record steps certified), mean
number of tests and certifications. A step is genuine if the true score of the new system exceeds the true score
of the system it was tested against.

usage: python validate_chain_test.py <R> [T]
Outputs results/validation_chain.json
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "honestboard", "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from honestboard import Board, certify_chain  # noqa: E402
import validate_record_test as vrt  # noqa: E402

RES = os.path.join(ROOT, "results")
K = 40
PROCS = [("unadjusted", "exact", "leader"), ("holm", "exact", "leader"), ("spending", "exact", "leader"), ("fallback", "exact", "leader"),
         ("lord", "exact", "leader"), ("spending", "polyhedral", "leader"), ("lord", "polyhedral", "leader"), ("lord", "exact", "certified")]


def true_scores(offsets, Lpool):
    return np.array([(1 / (1 + np.exp(-(a + Lpool)))).mean() for a in offsets])


VARIANT_SHARE, VARIANT_FLIP = 0.5, 0.10


def perturb(row, theta, rng, e1=VARIANT_FLIP):
    """Flip passes to failures with probability e1 and failures to passes with probability e1*theta/(1-theta), which
    keeps the expected score at theta while tying the new row to its parent."""
    e2 = e1 * theta / (1 - theta)
    u = rng.random(len(row))
    new = row.copy()
    new[(row > 0.5) & (u < e1)] = 0
    new[(row < 0.5) & (u < e2)] = 1
    return new


def scenario_offsets(name, Lpool, base, rng, step=0.03, period=5):
    if name in ("null", "variant"):
        return np.full(K, vrt.solve_a(Lpool, base))
    if name == "staircase":
        levels = [min(0.97, base + step * (k // period)) for k in range(K)]
        return np.array([vrt.solve_a(Lpool, s) for s in levels])
    if name == "late":
        levels = [base if k < K // 2 else min(0.97, base + 2 * step) for k in range(K)]
        return np.array([vrt.solve_a(Lpool, s) for s in levels])
    raise ValueError(name)


def run(R, T, seed=3):
    Lpool, base, sd_real, _ = vrt.fit_board(os.path.join(ROOT, "data", "boards", "swebench_verified.json"))
    rho = vrt.calibrate_rho(Lpool, base, sd_real)
    base = 0.60                                          # leave room for a staircase of improvements
    out = {"design": {"K": K, "T": T, "R": R, "rho": rho, "base": base, "step_points": 3, "period": 5,
                      "variant_share": VARIANT_SHARE, "variant_flip_pass_to_fail": VARIANT_FLIP}, "scenarios": {}}
    for scen in ("null", "staircase", "late", "variant"):
        t0 = time.time()
        acc = {p: {"fw": 0, "fdp": 0.0, "pow_num": 0, "pow_den": 0, "tests": 0, "cert": 0, "nan": 0} for p in PROCS}
        for r in range(R):
            offsets = scenario_offsets(scen, Lpool, base, vrt.rng)
            theta = true_scores(offsets, Lpool)
            Lt = vrt.rng.choice(Lpool, T)
            Y = np.vstack(vrt.draw(Lt, list(offsets), rho)).astype(float)
            if scen == "variant":
                for k in range(1, K):
                    if vrt.rng.random() < VARIANT_SHARE:
                        parent = int(vrt.rng.integers(k))
                        Y[k] = perturb(Y[parent], float(theta[parent]), vrt.rng)
            b = Board([f"s{k}" for k in range(K)], Y, [f"{20250101 + k}" for k in range(K)])
            for proc in PROCS:
                rule, pv, basel = proc
                res = certify_chain(b, rule=rule, pvalue=pv, baseline=basel, seed=r)
                idx = {n: i for i, n in enumerate(b.names)}
                false = total = genuine = genuine_cert = 0
                for s in res["steps"]:
                    truth = theta[idx[s["new"]]] > theta[idx[s["reference"]]] + 1e-12
                    if np.isnan(s["p"]):
                        acc[proc]["nan"] += 1
                    if s["certified"]:
                        total += 1
                        if not truth:
                            false += 1
                    if truth:
                        genuine += 1
                        genuine_cert += bool(s["certified"])
                a = acc[proc]
                a["fw"] += false > 0
                a["fdp"] += false / max(total, 1)
                a["pow_num"] += genuine_cert
                a["pow_den"] += genuine
                a["tests"] += res["n_tests"]
                a["cert"] += total
        out["scenarios"][scen] = {}
        for proc, a in acc.items():
            key = f"{proc[0]}_{proc[1]}_{proc[2]}"
            out["scenarios"][scen][key] = {"fwer": a["fw"] / R, "fdr": a["fdp"] / R,
                                            "power": (a["pow_num"] / a["pow_den"]) if a["pow_den"] else None,
                                            "mean_tests": a["tests"] / R, "mean_certified": a["cert"] / R,
                                            "undefined_pvalues": a["nan"]}
            v = out["scenarios"][scen][key]
            print(f"{scen:9s} {key:26s} FWER {v['fwer']:.3f}  FDR {v['fdr']:.3f}  power {v['power'] if v['power'] is None else round(v['power'], 3)}  "
                  f"tests {v['mean_tests']:.1f}  certified {v['mean_certified']:.2f}  nan {v['undefined_pvalues']}", flush=True)
        print(f"   ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(RES, "validation_chain.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    run(R, T)
