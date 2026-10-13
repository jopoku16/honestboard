"""Run-split replication of the record test on Terminal-Bench 2.

Terminal-Bench runs each task several times (five for most submissions). For each record step (holder a,
challenger b) and each of R random splits, every task's trials for a and for b are divided at random into a first
part (floor(n/2) trials, two of five) and a second part (the rest). Each part gives a per-task score (the share of
its trials that passed); the record test (the selective sign-flip test, with the per-date factor) is run on the
first part, and the second part records whether b still beats a (the step replicates) and whether the second-part
test is itself significant.

Trials are independent runs of the same agent on the same task, so this is a replication across runs rather than
across tasks: it asks whether a record survives the agent's own run-to-run randomness, which the task split of
split_half_replication.py cannot ask.

Null reference: if a and b are equally good on every task, the trials of a and b on a task are exchangeable, so the
reference redistributes the pooled trials of each task between a and b at random (keeping each system's trial
count), keeps the draws in which b beats a on the full data (the record event), and applies the same split.

Needs data/raw_terminalbench.json (the raw download). Writes results/tb_run_split.json.
usage: python tb_run_split.py [R] [null draws]
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "honestboard", "src"))
from honestboard import Board  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NULL = int(sys.argv[2]) if len(sys.argv) > 2 else 500
FLIPS = 4000
rng = np.random.default_rng(11)

raw = json.load(open(os.path.join(ROOT, "data", "raw_terminalbench.json"), encoding="utf-8"))
board_json = json.load(open(os.path.join(ROOT, "data", "boards", "terminalbench.json"), encoding="utf-8"))
tasks = board_json["tasks"]
T = len(tasks)
b0 = Board.from_json(os.path.join(ROOT, "data", "boards", "terminalbench.json"))
audit = b0.record_audit()


def trial_task(trial_name):
    return trial_name.rsplit("__", 1)[0]


def trials_of(name):
    """Per-task arrays of trial outcomes (1 pass, 0 fail), counted exactly as the board was built."""
    per_task = {}
    for res in raw[name]["jobs"].values():
        evals = (res.get("stats") or {}).get("evals") or {}
        for ev in evals.values():
            for rew, names in ((ev.get("reward_stats") or {}).get("reward") or {}).items():
                for t in names:
                    per_task.setdefault(trial_task(t), {})[t] = 1.0 if float(rew) >= 1.0 else 0.0
            for names in (ev.get("exception_stats") or {}).values():
                for t in names:
                    per_task.setdefault(trial_task(t), {}).setdefault(t, 0.0)
    return [np.array([per_task[t][k] for k in sorted(per_task[t])]) for t in tasks]


def selective_p(d, signs):
    """The record test for rates: P(S >= s_obs | S > 0) for the sum of randomly signed per-task differences."""
    null = (signs * d).mean(1)
    obs = d.mean()
    pos = int((null > 1e-15).sum())
    if obs <= 1e-15 or pos == 0:
        return 1.0
    return min(1.0, (1 + int((null >= obs - 1e-15).sum())) / (1 + pos))


def split_scores(trials):
    """Per-task scores on part 1 (floor(n/2) trials, at least one) and part 2 (the rest), trials permuted at random."""
    s1, s2 = np.empty(len(trials)), np.empty(len(trials))
    for k, tr in enumerate(trials):
        n = len(tr)
        m = max(1, n // 2)
        perm = rng.permutation(n)
        s1[k], s2[k] = tr[perm[:m]].mean(), tr[perm[m:]].mean()
    return s1, s2


def one_split(ta, tb, g, signs):
    a1, a2 = split_scores(ta)
    b1, b2 = split_scores(tb)
    d1, d2 = b1 - a1, b2 - a2
    sig1 = min(1.0, g * selective_p(d1, signs)) <= 0.05
    rep = d2.sum() > 1e-12
    repsig = bool(rep and min(1.0, g * selective_p(d2, signs)) <= 0.05)
    return sig1, rep, repsig


def pool(entries, keyn, keyr, keyrs):
    n = sum(e[keyn] for e in entries)
    return {"n": int(round(n)), "rep": (sum(e[keyr] for e in entries) / n) if n else None, "repsig": (sum(e[keyrs] for e in entries) / n) if n else None}


steps, skipped = [], []
for s in audit["steps"]:
    t0 = time.time()
    ta, tb = trials_of(s["previous"]), trials_of(s["new"])
    min_trials = min(min(len(x) for x in ta), min(len(x) for x in tb))
    if min_trials < 2:
        skipped.append({"new": s["new"], "previous": s["previous"], "reason": f"a task with {min_trials} trial"})
        continue
    g = s.get("date_group", 1)
    signs = rng.choice([-1.0, 1.0], size=(FLIPS, T))
    full = float(np.mean([x.mean() for x in tb]) - np.mean([x.mean() for x in ta]))
    sig1, rep, repsig = np.zeros(R, bool), np.zeros(R, bool), np.zeros(R, bool)
    for r in range(R):
        sig1[r], rep[r], repsig[r] = one_split(ta, tb, g, signs)
    # null reference: exchangeable trials within each task, conditioned on the record event on the full data
    na = [len(x) for x in ta]
    nsig1, nrep, nrepsig = [], [], []
    tries = 0
    while len(nsig1) < NULL:
        tries += 1
        pa, pb = [], []
        for k in range(T):
            pooled = np.concatenate([ta[k], tb[k]])
            perm = rng.permutation(len(pooled))
            pa.append(pooled[perm[:na[k]]])
            pb.append(pooled[perm[na[k]:]])
        if np.mean([x.mean() for x in pb]) - np.mean([x.mean() for x in pa]) <= 1e-12:
            continue
        x1, x2, x3 = one_split(pa, pb, g, signs)
        nsig1.append(x1); nrep.append(x2); nrepsig.append(x3)
    nsig1, nrep, nrepsig = np.array(nsig1), np.array(nrep), np.array(nrepsig)
    nr = {"p_sig1": float(nsig1.mean()),
          "rep_given_sig1": float(nrep[nsig1].mean()) if nsig1.any() else None,
          "repsig_given_sig1": float(nrepsig[nsig1].mean()) if nsig1.any() else None,
          "rep_given_nonsig1": float(nrep[~nsig1].mean()) if (~nsig1).any() else None,
          "repsig_given_nonsig1": float(nrepsig[~nsig1].mean()) if (~nsig1).any() else None,
          "draws": int(len(nsig1)), "record_event_rate": float(len(nsig1) / tries)}
    e = {"new": s["new"], "previous": s["previous"], "date": s["date"], "margin": s["margin"], "margin_from_trials": full,
         "p_full": s["p"], "sig_full": bool(s["significant_raw"]), "date_group": g, "min_trials": int(min_trials),
         "n_sig1": int(sig1.sum()), "rep_sig1": int((rep & sig1).sum()), "repsig_sig1": int((repsig & sig1).sum()),
         "n_nonsig1": int((~sig1).sum()), "rep_nonsig1": int((rep & ~sig1).sum()), "repsig_nonsig1": int((repsig & ~sig1).sum()),
         "null": nr,
         "null_counts": {"n_sig1": R * nr["p_sig1"], "rep_sig1": R * nr["p_sig1"] * (nr["rep_given_sig1"] or 0),
                         "repsig_sig1": R * nr["p_sig1"] * (nr["repsig_given_sig1"] or 0),
                         "n_nonsig1": R * (1 - nr["p_sig1"]), "rep_nonsig1": R * (1 - nr["p_sig1"]) * (nr["rep_given_nonsig1"] or 0),
                         "repsig_nonsig1": R * (1 - nr["p_sig1"]) * (nr["repsig_given_nonsig1"] or 0)}}
    steps.append(e)
    print(f"{s['date']} {s['new'][:40]:40s} margin {100 * s['margin']:5.1f} p {s['p']:.3f} | part-1 significant {sig1.mean():.2f} "
          f"(null {nr['p_sig1']:.2f}); replicates given sig {((rep & sig1).sum() / max(1, sig1.sum())):.2f} (null {nr['rep_given_sig1'] if nr['rep_given_sig1'] is not None else float('nan'):.2f}), "
          f"given not sig {((rep & ~sig1).sum() / max(1, (~sig1).sum())):.2f} (null {nr['rep_given_nonsig1'] if nr['rep_given_nonsig1'] is not None else float('nan'):.2f}) ({time.time() - t0:.0f}s)", flush=True)

out = {"board": "Terminal-Bench 2", "T": T, "R": R, "null_draws": NULL, "flips": FLIPS, "n_steps": len(audit["steps"]), "n_steps_used": len(steps),
       "skipped": skipped, "steps": steps,
       "pooled": {"real": {"sig1": pool(steps, "n_sig1", "rep_sig1", "repsig_sig1"), "nonsig1": pool(steps, "n_nonsig1", "rep_nonsig1", "repsig_nonsig1")},
                  "null": {"sig1": pool([e["null_counts"] for e in steps], "n_sig1", "rep_sig1", "repsig_sig1"),
                           "nonsig1": pool([e["null_counts"] for e in steps], "n_nonsig1", "rep_nonsig1", "repsig_nonsig1")}}}
for key, cond in (("full_sig", lambda e: e["sig_full"]), ("full_nonsig", lambda e: not e["sig_full"])):
    sub = [e for e in steps if cond(e)]
    tot = sum(e["n_sig1"] + e["n_nonsig1"] for e in sub)
    rep = sum(e["rep_sig1"] + e["rep_nonsig1"] for e in sub)
    out["pooled"][key] = {"steps": len(sub), "rep_any_half": (rep / tot) if tot else None}
p = out["pooled"]
fmt = lambda x: "--" if x is None else f"{x:.2f}"
print(f"\nPooled over {len(steps)} steps: part-1 significant splits replicate {fmt(p['real']['sig1']['rep'])} (significant on part 2 {fmt(p['real']['sig1']['repsig'])}) "
      f"against {fmt(p['real']['nonsig1']['rep'])} for the rest; null reference {fmt(p['null']['sig1']['rep'])} and {fmt(p['null']['nonsig1']['rep'])}.")
print(f"Records significant on the full data replicate across runs {fmt(p['full_sig']['rep_any_half'])} of the time ({p['full_sig']['steps']} steps); the others {fmt(p['full_nonsig']['rep_any_half'])} ({p['full_nonsig']['steps']} steps).")
with open(os.path.join(ROOT, "results", "tb_run_split.json"), "w") as f:
    json.dump(out, f, indent=2)
