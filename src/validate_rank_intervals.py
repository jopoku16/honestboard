"""Do the 95% rank intervals cover the true ranks? Known-truth simulation.

The observed score matrix is the population (true score = row mean, true rank = rank of true scores, ties share the
better rank). A "benchmark" is n tasks drawn with replacement. For each system we ask whether its bootstrap rank
interval, computed from the benchmark alone, contains its true rank.

Reported: coverage averaged over all systems, over the ten truly best systems, and for the system that the benchmark
ranks first (the leaderboard winner, whose interval is selected on the same data).

usage: python validate_rank_intervals.py <verified|lite> <R>
Outputs results/validation_rankintervals_<tag>.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "honestboard", "src"))
from honestboard import Board  # noqa: E402

FILES = {"verified": "swebench_verified.json", "lite": "swebench_lite.json"}


def run(tag, R, B=400, seed=5):
    b0 = Board.from_json(os.path.join(ROOT, "data", "boards", FILES[tag]))
    K, T = b0.Y.shape
    theta = b0.Y.mean(1)
    true_rank = stats.rankdata(-theta, method="min")
    best10 = np.argsort(theta)[::-1][:10]
    rng = np.random.default_rng(seed)
    out = {"board": tag, "K": int(K), "T": int(T), "R": R, "by_n": {}}
    for n in (250, T):
        cov_all, cov_top10, cov_winner, width_top10, rank_err = [], [], [], [], []
        t0 = time.time()
        for _ in range(R):
            idx = rng.integers(0, T, n)
            b = Board(b0.names, b0.Y[:, idx])
            rs = b.rank_summary(B=B, seed=int(rng.integers(1 << 30)))
            name_to = {r["name"]: r for r in rs}
            lo = np.array([name_to[nm]["rank_lo"] for nm in b0.names])
            hi = np.array([name_to[nm]["rank_hi"] for nm in b0.names])
            covered = (lo <= true_rank) & (true_rank <= hi)
            cov_all.append(covered.mean())
            cov_top10.append(covered[best10].mean())
            width_top10.append((hi - lo + 1)[best10].mean())
            sc = b.scores()
            w = int(np.argmax(sc + rng.random(K) * 1e-12))
            cov_winner.append(bool(covered[w]))
        out["by_n"][str(n)] = {"coverage_all_systems": float(np.mean(cov_all)),
                               "coverage_true_top10": float(np.mean(cov_top10)),
                               "coverage_selected_winner": float(np.mean(cov_winner)),
                               "mean_width_true_top10": float(np.mean(width_top10))}
        print(f"{tag} n={n} ({time.time() - t0:.0f}s): all {np.mean(cov_all):.3f}, true top-10 {np.mean(cov_top10):.3f}, "
              f"selected winner {np.mean(cov_winner):.3f}, mean width top-10 {np.mean(width_top10):.1f}", flush=True)
    # hard case: the top m systems share one true ability, so their true rank is a tie for first
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from validate_winners_curse import bunched_model
    Pm, leaders = bunched_model(b0.Y.astype(float), m=min(20, K // 3))
    truth_b = Pm.mean(1)
    true_rank_b = stats.rankdata(-truth_b, method="min")          # tied leaders all have true rank 1
    true_rank_avg = stats.rankdata(-truth_b, method="average")    # tied leaders share the average rank (m + 1) / 2
    lead = np.array(sorted(leaders))
    m = len(lead)
    out["bunched"] = {"n_equal_leaders": m}
    for n in (250, T):
        cov_lead, cov_lead_avg, cov_lead_block, cov_all2, in_set, set_size, t0 = [], [], [], [], [], [], time.time()
        for _ in range(R):
            idx = rng.integers(0, T, n)
            b = Board(b0.names, rng.binomial(1, Pm[:, idx]).astype(np.uint8))
            rs = {r["name"]: r for r in b.rank_summary(B=B, seed=int(rng.integers(1 << 30)))}
            lo = np.array([rs[nm]["rank_lo"] for nm in b0.names])
            hi = np.array([rs[nm]["rank_hi"] for nm in b0.names])
            covered = (lo <= true_rank_b) & (true_rank_b <= hi)
            cov_lead.append(covered[lead].mean())
            cov_lead_avg.append(((lo <= true_rank_avg) & (true_rank_avg <= hi))[lead].mean())
            cov_lead_block.append(((lo <= m) & (hi >= 1))[lead].mean())      # interval meets the block of tied ranks 1..m
            cov_all2.append(covered.mean())
            # the confidence set of the best should contain each truly best system with probability about 95%
            cs = b.confidence_set_best(alpha=0.05, B=300, seed=int(rng.integers(1 << 30)))
            members = set(cs["members"])
            in_set.append(np.mean([b0.names[k] in members for k in lead]))
            set_size.append(cs["size"])
        out["bunched"][str(n)] = {"coverage_equal_leaders": float(np.mean(cov_lead)), "coverage_equal_leaders_avg": float(np.mean(cov_lead_avg)),
                                  "coverage_equal_leaders_block": float(np.mean(cov_lead_block)), "coverage_all_systems": float(np.mean(cov_all2)),
                                  "confidence_set_contains_each_equal_leader": float(np.mean(in_set)), "median_set_size": float(np.median(set_size))}
        print(f"{tag} bunched n={n} ({time.time() - t0:.0f}s): interval covers rank 1 (min convention) for equal leaders {np.mean(cov_lead):.3f}, "
              f"average rank {np.mean(cov_lead_avg):.3f}, meets the tied block {np.mean(cov_lead_block):.3f}; all systems {np.mean(cov_all2):.3f}; "
              f"confidence set contains each equal leader {np.mean(in_set):.3f} (median set size {np.median(set_size):.0f})", flush=True)
    with open(os.path.join(ROOT, "results", f"validation_rankintervals_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 400)
