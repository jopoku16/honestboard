import json, os, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw = json.load(open(os.path.join(ROOT, "data", "raw_terminalbench.json"), encoding="utf-8"))
board = json.load(open(os.path.join(ROOT, "data", "boards", "terminalbench.json"), encoding="utf-8"))
keep = {s["name"] for s in board["systems"]}
rows = []
for name, d in raw.items():
    if name not in keep:
        continue
    ntot = [int(r.get("n_total_trials") or 0) for r in d["jobs"].values()]
    leaderboard_mean = []
    for r in d["jobs"].values():
        for ek, ev in (r["stats"].get("evals") or {}).items():
            m = ev.get("metrics") or []
            leaderboard_mean.append(round(m[0]["mean"], 3) if m else None)
    s = next(x for x in board["systems"] if x["name"] == name)
    mine = sum(s["rates"].values()) / len(s["rates"])
    rows.append((name, len(d["jobs"]), ntot, leaderboard_mean, round(mine, 3)))
print("total trials per submission (sum over jobs), distribution of sum/89:")
print(collections.Counter(round(sum(r[2]) / 89, 2) for r in rows))
print("multi-job examples (name, jobs, n_total per job, job-level mean, my mean):")
for r in [r for r in rows if r[1] > 1][:10]:
    print("  ", r)
