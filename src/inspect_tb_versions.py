"""Where does the benchmark version live in the Terminal-Bench job results, and are there five trials per task?"""
import collections
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw = json.load(open(os.path.join(ROOT, "data", "raw_terminalbench.json"), encoding="utf-8"))


def walk(o, path="", hits=None):
    hits = hits if hits is not None else []
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{path}.{k}" if path else k
            if isinstance(v, (dict, list)):
                walk(v, p, hits)
            elif isinstance(v, str) and (re.search(r"version|dataset|registry|tbench|terminal-bench", k, re.I) or re.search(r"terminal-bench-?2|tbench", v, re.I)):
                hits.append((p, v[:90]))
    elif isinstance(o, list):
        for i, v in enumerate(o[:3]):
            walk(v, f"{path}[{i}]", hits)
    return hits


names = sorted(raw)
for name in names[:4]:
    for job, res in raw[name]["jobs"].items():
        print(f"== {name} / {job}: top-level keys {sorted(res)[:12]}")
        for p, v in walk(res)[:12]:
            print("   ", p, "=", v)

# version candidates across all jobs: job folder name, eval key, any 'version' strings
jobname = collections.Counter()
evalkey = collections.Counter()
for name in names:
    for job, res in raw[name]["jobs"].items():
        jobname[re.sub(r"-k\d.*|-v\d+.*", "", job)] += 1
        for ek in (res.get("stats", {}).get("evals") or {}):
            evalkey[ek.split("__")[-1]] += 1
print("\njob folder prefixes:", jobname.most_common(12))
print("eval key tails:", evalkey.most_common(12))

# per-task trial counts for a multi-job submission
multi = [n for n in names if len(raw[n]["jobs"]) > 1][:3]
for name in multi:
    per_task = collections.Counter()
    for job, res in raw[name]["jobs"].items():
        for ek, ev in (res.get("stats", {}).get("evals") or {}).items():
            for rew, lst in (ev.get("reward_stats") or {}).get("reward", {}).items():
                for t in lst:
                    per_task[t.rsplit("__", 1)[0]] += 1
            for exc, lst in (ev.get("exception_stats") or {}).items():
                for t in lst:
                    per_task[t.rsplit("__", 1)[0]] += 1
    print(f"\n{name}: jobs {len(raw[name]['jobs'])}, tasks {len(per_task)}, trials per task {collections.Counter(per_task.values())}")
