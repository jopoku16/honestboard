"""Split-half replication of the record test on the real boards.

For each record step (holder a, challenger b) and each of R random halves of the tasks, the record test (the selective
exact test) is run on half 1, and half 2 records whether b's margin over a is positive (the step replicates) and whether the half-2
test is itself significant. Replication rates are compared between steps that were significant on half 1 and the rest.

Caveat: every step is a record on the full data, so given the full-data margin the two halves are negatively
dependent, and even a lucky record replicates on half 2 more often than a coin flip. A null reference is therefore
simulated for each step with the same number of discordant tasks: fair-coin signs, conditioned on the full-data
record event, each discordant task sent to half 1 with probability one half. The reference says what the
replication rates would look like if no record were genuine.

Writes results/split_half.json. usage: python split_half_replication.py [R] [null draws]
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

BOARDS = [("SWE-bench Verified", "swebench_verified.json"), ("SWE-bench Lite", "swebench_lite.json"),
          ("SWE-bench Test", "swebench_test.json"), ("SWE-bench Multimodal", "swebench_multimodal.json"),
          ("Terminal-Bench 2", "terminalbench.json")]
R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NULL = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
rng = np.random.default_rng(7)


def p_two_sided(nb, n):
    """The record test: selective p-value P(X >= nb | X > n/2) on n discordant tasks of which nb favour the
    challenger (p = 1 when nb <= n/2 or n = 0). The function keeps its old name; it was the two-sided test before
    the selective p-value became the primary test."""
    nb, n = np.asarray(nb), np.asarray(n)
    nn = np.maximum(n, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tail = stats.binom.sf(nb - 1, nn, 0.5)
        cond = stats.binom.sf(nn // 2, nn, 0.5)
        p = np.where((n > 0) & (nb > n / 2), np.minimum(1.0, tail / cond), 1.0)
    return p


def null_reference(n, draws):
    """Expected (sig1, rep, repsig) pattern per split under fair-coin signs on n discordant tasks, conditioned on the
    full-data record event n_b > n/2."""
    need, rows = draws, []
    while need > 0:
        S = rng.random((2 * need + 50, n)) < 0.5                 # True = task favours the challenger
        S = S[S.sum(1) > n / 2]
        rows.append(S[:need])
        need -= len(S[:need])
    S = np.vstack(rows)
    H = rng.random(S.shape) < 0.5                                 # True = task falls in half 1
    n1 = H.sum(1); nb1 = (S & H).sum(1)
    n2 = n - n1; nb2 = S.sum(1) - nb1
    sig1 = p_two_sided(nb1, n1) <= 0.05
    rep = nb2 > n2 - nb2
    repsig = rep & (p_two_sided(nb2, n2) <= 0.05)
    return {"p_sig1": float(sig1.mean()),
            "rep_given_sig1": float(rep[sig1].mean()) if sig1.any() else None,
            "repsig_given_sig1": float(repsig[sig1].mean()) if sig1.any() else None,
            "rep_given_nonsig1": float(rep[~sig1].mean()) if (~sig1).any() else None,
            "repsig_given_nonsig1": float(repsig[~sig1].mean()) if (~sig1).any() else None}


def pool(entries, keyn, keyr, keyrs):
    n = sum(e[keyn] for e in entries)
    return {"n": int(n), "rep": (sum(e[keyr] for e in entries) / n) if n else None, "repsig": (sum(e[keyrs] for e in entries) / n) if n else None}


out = {"R": R, "null_draws": NULL, "boards": []}
grand = []
for label, fname in BOARDS:
    t0 = time.time()
    b0 = Board.from_json(os.path.join(ROOT, "data", "boards", fname))
    a = b0.record_audit()
    T = b0.T
    steps = []
    for s in a["steps"]:
        i, j = b0.names.index(s["previous"]), b0.names.index(s["new"])
        d = b0.Y[j].astype(float) - b0.Y[i].astype(float)
        if b0.is_binary:
            disc = np.where(d != 0)[0]
            sign = d[disc] > 0
            n = len(disc)
            H = rng.random((R, T)) < 0.5                          # half 1 membership of every task, per split
            Hd = H[:, disc]
            n1 = Hd.sum(1); nb1 = (Hd & sign).sum(1)
            n2 = n - n1; nb2 = int(sign.sum()) - nb1
            sig1 = p_two_sided(nb1, n1) <= 0.05
            rep = nb2 > n2 - nb2
            repsig = rep & (p_two_sided(nb2, n2) <= 0.05)
        else:                                                     # rates: paired sign-flip test on each half
            n = int((d != 0).sum())
            sig1, rep, repsig = np.zeros(R, bool), np.zeros(R, bool), np.zeros(R, bool)
            for r in range(R):
                perm = rng.permutation(T)
                h1, h2 = perm[: T // 2], perm[T // 2:]
                c1 = Board(b0.names, b0.Y[:, h1]).compare(i, j, B=2000, seed=r)
                c2 = Board(b0.names, b0.Y[:, h2]).compare(i, j, B=2000, seed=r + 1)
                sig1[r] = c1["p_selective"] <= 0.05
                rep[r] = c2["diff"] > 0
                repsig[r] = rep[r] and c2["p_selective"] <= 0.05
        e = {"new": s["new"], "previous": s["previous"], "margin": s["margin"], "p_full": s["p"], "sig_full": bool(s["significant_raw"]),
             "n_discordant": n, "n_sig1": int(sig1.sum()), "rep_sig1": int((rep & sig1).sum()), "repsig_sig1": int((repsig & sig1).sum()),
             "n_nonsig1": int((~sig1).sum()), "rep_nonsig1": int((rep & ~sig1).sum()), "repsig_nonsig1": int((repsig & ~sig1).sum())}
        if n > 0:
            nr = null_reference(n, NULL)
            e["null"] = nr
            # expected counts per R splits under the null, for pooling on the same footing as the real counts
            e["null_counts"] = {"n_sig1": R * nr["p_sig1"], "rep_sig1": R * nr["p_sig1"] * (nr["rep_given_sig1"] or 0),
                                "repsig_sig1": R * nr["p_sig1"] * (nr["repsig_given_sig1"] or 0),
                                "n_nonsig1": R * (1 - nr["p_sig1"]), "rep_nonsig1": R * (1 - nr["p_sig1"]) * (nr["rep_given_nonsig1"] or 0),
                                "repsig_nonsig1": R * (1 - nr["p_sig1"]) * (nr["repsig_given_nonsig1"] or 0)}
        steps.append(e)
    grand.extend(steps)
    board = {"board": label, "T": T, "n_steps": len(steps), "steps": steps,
             "real": {"sig1": pool(steps, "n_sig1", "rep_sig1", "repsig_sig1"), "nonsig1": pool(steps, "n_nonsig1", "rep_nonsig1", "repsig_nonsig1")},
             "null": {"sig1": pool([e["null_counts"] for e in steps if "null_counts" in e], "n_sig1", "rep_sig1", "repsig_sig1"),
                      "nonsig1": pool([e["null_counts"] for e in steps if "null_counts" in e], "n_nonsig1", "rep_nonsig1", "repsig_nonsig1")}}
    out["boards"].append(board)
    rl, nl = board["real"], board["null"]
    fmt = lambda x: "--" if x is None else f"{x:.2f}"
    print(f"{label:22s} steps {len(steps):2d}  half-1 significant: real {rl['sig1']['n']:5d} splits, replicates {fmt(rl['sig1']['rep'])} (sig {fmt(rl['sig1']['repsig'])}); "
          f"not significant: {rl['nonsig1']['n']:5d}, replicates {fmt(rl['nonsig1']['rep'])} (sig {fmt(rl['nonsig1']['repsig'])}) | "
          f"null: {fmt(nl['sig1']['rep'])} / {fmt(nl['nonsig1']['rep'])} ({time.time() - t0:.0f}s)", flush=True)

out["pooled"] = {"real": {"sig1": pool(grand, "n_sig1", "rep_sig1", "repsig_sig1"), "nonsig1": pool(grand, "n_nonsig1", "rep_nonsig1", "repsig_nonsig1")},
                 "null": {"sig1": pool([e["null_counts"] for e in grand if "null_counts" in e], "n_sig1", "rep_sig1", "repsig_sig1"),
                          "nonsig1": pool([e["null_counts"] for e in grand if "null_counts" in e], "n_nonsig1", "rep_nonsig1", "repsig_nonsig1")}}
# the same split by full-data significance: do the 20 significant records replicate on random halves more than the others?
for key, cond in (("full_sig", lambda e: e["sig_full"]), ("full_nonsig", lambda e: not e["sig_full"])):
    sub = [e for e in grand if cond(e)]
    tot = sum(e["n_sig1"] + e["n_nonsig1"] for e in sub)
    rep = sum(e["rep_sig1"] + e["rep_nonsig1"] for e in sub)
    out["pooled"][key] = {"steps": len(sub), "rep_any_half": (rep / tot) if tot else None}
p = out["pooled"]
print(f"\nPooled: half-1 significant splits replicate {p['real']['sig1']['rep']:.2f} (significant on half 2: {p['real']['sig1']['repsig']:.2f}) against "
      f"{p['real']['nonsig1']['rep']:.2f} ({p['real']['nonsig1']['repsig']:.2f}) for the rest; null reference {p['null']['sig1']['rep']:.2f} and {p['null']['nonsig1']['rep']:.2f}.")
print(f"Records significant on the full data replicate on a random half {p['full_sig']['rep_any_half']:.2f} of the time; the others {p['full_nonsig']['rep_any_half']:.2f}.")
with open(os.path.join(ROOT, "results", "split_half.json"), "w") as f:
    json.dump(out, f, indent=2)
