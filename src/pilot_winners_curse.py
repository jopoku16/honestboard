"""Pilot: how large is the winner's curse / rank fragility on the SWE-bench Verified leaderboard?

Inputs : data/raw_verified.json (results.json + metadata.yaml per public submission)
Outputs: results/pilot_results.json and a printed summary.

Three measurements, all from per-task resolved lists (no API spend):
  1. OOB-bootstrap optimism of the selected winner (select on in-bag tasks, score on out-of-bag).
  2. Paired-bootstrap rank intervals for the top contenders.
  3. Share of record-setting submissions whose margin over the previous record is not significant.
"""
import json, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
rng = np.random.default_rng(20261012)
N_TASKS = 500
B = 4000

raw = json.load(open(os.path.join(D, "raw_verified.json"), encoding="utf-8"))

subs = []
for name, d in sorted(raw.items()):
    if "results" not in d:
        continue
    try:
        r = json.loads(d["results"])
    except Exception:
        continue
    if not isinstance(r, dict) or "resolved" not in r or not isinstance(r["resolved"], list):
        continue
    m = re.match(r"(\d{8})_(.*)", name)
    date = m.group(1) if m else "00000000"
    subs.append({"name": name, "date": date, "resolved": set(r["resolved"])})

# task universe: every id that appears anywhere, padded with never-solved tasks up to 500
ids = sorted(set().union(*[s["resolved"] for s in subs]))
n_seen = len(ids)
pad = max(0, N_TASKS - n_seen)
ids = ids + [f"__never_solved_{i}" for i in range(pad)]
idx = {t: i for i, t in enumerate(ids)}
T = len(ids)
K = len(subs)
X = np.zeros((K, T), dtype=np.int8)
for k, s in enumerate(subs):
    for t in s["resolved"]:
        X[k, idx[t]] = 1
score = X.mean(1)
order = np.argsort(-score)
print(f"submissions={K}  tasks_seen={n_seen}  padded_to={T}  score range {score.min():.3f}..{score.max():.3f}")
print("top 10:")
for k in order[:10]:
    print(f"  {subs[k]['name'][:60]:60s} {score[k]*100:5.1f}")

# restrict to contenders: within 5 points of the best score
best = score.max()
cont = np.where(score >= best - 0.05)[0]
print(f"contenders within 5 pts of best: {len(cont)}")

res = {"K": K, "T": T, "tasks_seen": n_seen, "best": float(best), "n_contenders_5pt": int(len(cont))}

# ---- 1. OOB-bootstrap optimism of the selected winner ----
def oob_optimism(sub_idx, topn):
    """Select top-n by in-bag score; compare in-bag vs out-of-bag score of the selected, net of the field."""
    Xs = X[sub_idx].astype(np.float32)
    gaps = []
    for _ in range(B):
        draw = rng.integers(0, T, T)
        inbag = np.bincount(draw, minlength=T).astype(np.float32)
        oob = (inbag == 0).astype(np.float32)
        if oob.sum() < 10:
            continue
        s_in = Xs @ inbag / inbag.sum()
        s_out = Xs @ oob / oob.sum()
        sel = np.argsort(-s_in)[:topn]
        # net of common difficulty shift: subtract the field-average in/out difference
        gap = (s_in[sel] - s_out[sel]).mean() - (s_in - s_out).mean()
        gaps.append(gap)
    gaps = np.array(gaps)
    return float(gaps.mean()), float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))

res["optimism"] = {}
for label, sidx in [("all_submissions", np.arange(K)), ("contenders_5pt", cont)]:
    for topn in (1, 3, 5):
        if topn > len(sidx):
            continue
        m, lo, hi = oob_optimism(sidx, topn)
        res["optimism"][f"{label}_top{topn}"] = {"mean_pts": m * 100, "lo": lo * 100, "hi": hi * 100}
        print(f"optimism {label:16s} top{topn}: {m*100:+.2f} pts  [{lo*100:+.2f}, {hi*100:+.2f}]")

# ---- 2. Rank intervals for the contenders under task resampling ----
Xc = X[cont].astype(np.float32)
ranks = np.zeros((B, len(cont)))
for b in range(B):
    draw = rng.integers(0, T, T)
    w = np.bincount(draw, minlength=T).astype(np.float32)
    s = Xc @ w
    ranks[b] = (-s).argsort().argsort() + 1
corder = np.argsort(-score[cont])
res["rank_intervals"] = []
print("rank intervals (95%) for the leaders:")
for j in corder[:10]:
    lo, hi = np.percentile(ranks[:, j], [2.5, 97.5])
    p_top = float((ranks[:, j] == 1).mean())
    res["rank_intervals"].append({"name": subs[cont[j]]["name"], "score": float(score[cont[j]]), "rank_lo": float(lo), "rank_hi": float(hi), "p_rank1": p_top})
    print(f"  {subs[cont[j]]['name'][:55]:55s} {score[cont[j]]*100:5.1f}  rank [{lo:.0f}, {hi:.0f}]  P(rank1)={p_top:.2f}")
top1 = corder[0]
p_win = float((ranks[:, top1] == 1).mean())
res["p_leader_is_leader"] = p_win
print(f"P(the observed #1 is #1 on resampled tasks) = {p_win:.2f}")

# ---- 3. Record-setters: is each record margin distinguishable from the previous record? ----
chron = sorted(range(K), key=lambda k: (subs[k]["date"], -score[k]))
records = []
cur = None
for k in chron:
    if cur is None or score[k] > score[cur]:
        if cur is not None:
            records.append((cur, k))
        cur = k
def paired_p(a, b):
    """Two-sided bootstrap p-value that submission b exceeds a on resampled tasks."""
    d = (X[b] - X[a]).astype(np.float32)
    draws = rng.integers(0, T, (B, T))
    means = d[draws].mean(1)
    obs = d.mean()
    return float(2 * min((means <= 0).mean(), (means >= 0).mean())), float(obs)
sig = 0
rows = []
for a, b in records:
    p, obs = paired_p(a, b)
    rows.append({"prev": subs[a]["name"], "new": subs[b]["name"], "margin_pts": obs * 100, "p": p})
    sig += p < 0.05
res["records"] = {"n": len(records), "significant_at_5pct": int(sig), "rows": rows}
print(f"record-setting steps: {len(records)}; margin significant at 5%: {sig} ({100*sig/max(1,len(records)):.0f}%)")
small = [r for r in rows if r["margin_pts"] < 2.0]
print(f"  steps with margin < 2 pts: {len(small)}; of those significant: {sum(r['p']<0.05 for r in small)}")

json.dump(res, open(os.path.join(RES, "pilot_results.json"), "w"), indent=2)
