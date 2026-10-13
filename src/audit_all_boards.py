"""Run the full honestboard audit on every board and write results/audit_summary.json.

Primary record test: the selective exact paired test (sign test conditional on the record event), best-first ordering
of same-date submissions. Sensitivity: two-sided exact, one-sided at 5%, worst-first ordering, cluster-level sign-flip
test. Resolution numbers carry a task-bootstrap 95% interval. Confidence set of the best: MCB-style
(primary) and the all-pairs simultaneous version. Resolution with iid and cluster-robust variance. Lag analysis,
upper bounds of non-significant margins, selection-corrected winner score.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "honestboard", "src"))
from honestboard import Board  # noqa: E402

# (label, board file, has cluster labels, is an agent leaderboard). LiveBench is a model leaderboard kept as a
# non-agent comparison; the totals cover the agent boards only.
BOARDS = [
    ("SWE-bench Verified", "swebench_verified.json", True, True),
    ("SWE-bench Lite", "swebench_lite.json", True, True),
    ("SWE-bench Test", "swebench_test.json", True, True),
    ("SWE-bench Multimodal", "swebench_multimodal.json", True, True),
    ("SWE-bench Multilingual", "swebench_multilingual.json", True, True),
    ("Terminal-Bench 2", "terminalbench.json", False, True),
    ("LiveBench", "livebench.json", True, False),
]


def counts(a):
    return {"steps": a["n_steps"], "significant": a["n_significant_raw"], "holm": a["n_significant_holm"]}


def audit(label, fname, has_clusters, B=4000):
    b = Board.from_json(os.path.join(ROOT, "data", "boards", fname))
    sc = b.scores()
    rs = b.rank_summary(B=B, top=3)
    lead = max(rs, key=lambda r: r["p_best"])          # the system with the highest top-rank frequency
    cs = b.confidence_set_best(B=B)                    # MCB-style
    cs_sim = b.confidence_set_best(B=B, method="simultaneous")
    out = {"board": label, "systems": b.K, "tasks": b.T,
           "leader": lead["name"], "leader_score": lead["score"], "leader_p_best": lead["p_best"],
           "leader_rank_interval": [lead["rank_lo"], lead["rank_hi"]],
           "leader_tied": int((sc >= sc.max() - 1e-12).sum()),
           "top3": rs, "confidence_set_size_95": cs["size"], "confidence_set_members": cs["members"],
           "confidence_set_critical": cs["critical_value"],
           "confidence_set_size_95_simultaneous": cs_sim["size"]}
    if has_clusters:
        csc = b.confidence_set_best(B=B, cluster=True)
        out["confidence_set_size_95_cluster"] = csc["size"]
        out["leader_p_best_cluster"] = max(r["p_best"] for r in b.rank_summary(B=B, top=3, cluster=True))
    if b.dates:
        a = b.record_audit()                                   # primary: selective p-value, best-first
        two = b.record_audit(sided="two")
        one = b.record_audit(sided="one")
        worst = b.record_audit(tie_order="worst_first")
        lags = b.record_lags()
        res = b.resolution()
        rng = np.random.default_rng(5)                         # task-bootstrap interval for the resolution numbers
        boot = [Board(b.names, b.Y[:, rng.integers(0, b.T, b.T)]).resolution() for _ in range(400)]
        res["mdd_ci95"] = [float(np.percentile([x["mdd"] for x in boot], 2.5)), float(np.percentile([x["mdd"] for x in boot], 97.5))]
        res["tasks_for_1pt_ci95"] = [float(np.percentile([x["tasks_for_1pt"] for x in boot], 2.5)), float(np.percentile([x["tasks_for_1pt"] for x in boot], 97.5))]
        margins = [s["margin"] for s in a["steps"]]
        Sb = b._boot_scores(3000, 1)
        uppers = []
        for s in a["steps"]:
            dd = Sb[b.names.index(s["new"])] - Sb[b.names.index(s["previous"])]
            s["upper975"] = float(np.percentile(dd, 97.5))
            if not s["significant_raw"]:
                uppers.append(s["upper975"])
        clus = None
        if has_clusters:
            ac = b.record_audit(cluster=True)
            clus = counts(ac)
            out["resolution_cluster"] = b.resolution(cluster=True)
        out["records"] = {
            **counts(a), "details": a["steps"],
            "two_sided": counts(two), "one_sided": counts(one), "worst_first": counts(worst), "cluster_robust": clus,
            "small_margin_steps(<2pts)": sum(1 for m in margins if m < 0.02),
            "small_margin_significant": sum(1 for s in a["steps"] if s["margin"] < 0.02 and s["significant_raw"]),
            "median_margin": float(np.median(margins)) if margins else None,
            "median_lag": lags["median_lag"], "share_adjacent": lags["share_adjacent"], "share_never": lags["share_never"],
            "lags": lags["lags"],
            "median_upper975_nonsig": float(np.median(uppers)) if uppers else None,
        }
        out["resolution"] = res
    w = b.winner_adjusted(B=B, w=b.names.index(lead["name"]))      # the same system as the leader column, also when the top is tied
    out["winner"] = {"name": w["name"], "observed": w["observed"], "adjusted_iterated": w["estimate"], "ci_iterated": list(w["ci"]),
                     "naive_ci": list(w["naive_ci"]), "conditional_estimate": w["conditional"]["estimate"],
                     "co_leaders": [b.names[i] for i in np.where(sc >= sc.max() - 1e-12)[0]]}
    wp = b.winner_bootstrap(B=B, w=w["winner"])
    out["winner"]["adjusted_plain"] = wp["estimate"]
    return out


if __name__ == "__main__":
    results = []
    for label, fname, hc, agent in BOARDS:
        if not os.path.exists(os.path.join(ROOT, "data", "boards", fname)):
            print("missing", fname)
            continue
        r = audit(label, fname, hc)
        r["agent"] = agent
        results.append(r)
        rec = r.get("records", {})
        print(f"{label:24s} K={r['systems']:3d} T={r['tasks']:4d} top-rank {r['leader_p_best']:.2f} (tied {r['leader_tied']}) "
              f"set {r['confidence_set_size_95']:3d} (all-pairs {r['confidence_set_size_95_simultaneous']:3d}) "
              f"records {rec.get('significant', '-')}/{rec.get('steps', '-')} selective, {rec.get('two_sided', {}).get('significant', '-')} two-sided, {rec.get('holm', '-')} Holm, "
              f"one-sided {rec.get('one_sided', {}).get('significant', '-')}, worst-first {rec.get('worst_first', {}).get('significant', '-')}/{rec.get('worst_first', {}).get('steps', '-')}; "
              f"MDD {100 * r['resolution']['mdd'] if 'resolution' in r else float('nan'):.1f} pts; "
              f"winner {r['winner']['observed'] * 100:.1f} -> {r['winner']['adjusted_iterated'] * 100:.1f}")
    keys = ("steps", "significant", "holm")
    agent_boards = [r for r in results if r["agent"] and "records" in r]
    tot = {k: sum(r["records"][k] for r in agent_boards) for k in keys}
    tot1 = {k: sum(r["records"]["one_sided"][k] for r in agent_boards) for k in keys}
    tot2 = {k: sum(r["records"]["two_sided"][k] for r in agent_boards) for k in keys}
    totw = {k: sum(r["records"]["worst_first"][k] for r in agent_boards) for k in keys}
    cl = [r["records"]["cluster_robust"] for r in agent_boards if r["records"].get("cluster_robust")]
    cl_sig, cl_steps = sum(c["significant"] for c in cl), sum(c["steps"] for c in cl)
    print(f"\nAgent boards (primary, selective): {tot['steps']} record steps, {tot['significant']} significant ({100 * tot['significant'] / tot['steps']:.0f}%), "
          f"{tot['holm']} after Holm. Two-sided: {tot2['significant']}. One-sided: {tot1['significant']}. Worst-first: {totw['significant']}/{totw['steps']}. "
          f"Cluster-robust (SWE-bench boards): {cl_sig}/{cl_steps}.")
    json.dump({"boards": results, "totals": tot, "totals_one_sided": tot1, "totals_two_sided": tot2, "total_two_sided_significant": tot2["significant"],
               "totals_worst_first": totw, "cluster_robust_total": {"significant": cl_sig, "steps": cl_steps},
               "totals_note": "totals cover the agent leaderboards only; LiveBench is reported separately"},
              open(os.path.join(ROOT, "results", "audit_summary.json"), "w"), indent=2, default=float)
