"""Build a honestboard board from the public Terminal-Bench leaderboard dataset on Hugging Face.

Source : https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard
Reads  : submissions/terminal-bench/2.0/<Agent>__<Model>/<job>/result.json (job-level file only, ~30 KB)
Writes : data/raw_terminalbench.json (raw job files), data/boards/terminalbench.json, results/tb_build.json

Per task and system, every trial named in the job files is counted once: trials listed under reward 1.0 pass,
trials listed under reward 0.0 fail, and trials that appear only in the exception lists (errored, timed out) count
as failures. The per-task score is the share of that task's trials that passed, with the trial count the file
actually reports (pooled multi-job submissions can have more than five trials on a task). Submissions with a job
labelled as benchmark version 2.1 are dropped; the rest are the 2.0 leaderboard's submissions (most job files do
not name a version, and the dataset path is the 2.0 leaderboard). A submission must have at least one trial on
every task to be kept.
"""
import json
import os
import re
import time
from collections import Counter

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
os.makedirs(os.path.join(DATA, "boards"), exist_ok=True)
os.makedirs(RES, exist_ok=True)
DS = "harborframework/terminal-bench-2-leaderboard"
API = f"https://huggingface.co/api/datasets/{DS}/tree/main"
RAW = f"https://huggingface.co/datasets/{DS}/resolve/main/"
UA = {"User-Agent": "honestboard-audit (jeffery.opoku01@utrgv.edu)"}


def get_json(url, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60, headers=UA)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(3 * (i + 1))
        except Exception:
            time.sleep(2)
    return None


def fetch_raw():
    path = os.path.join(DATA, "raw_terminalbench.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    subs = get_json(f"{API}/submissions/terminal-bench/2.0") or []
    dirs = [e["path"] for e in subs if e["type"] == "directory"]
    raw = {}
    for i, d in enumerate(dirs):
        name = d.split("/")[-1]
        ents = get_json(f"{API}/{d}") or []
        jobs = [e["path"] for e in ents if e["type"] == "directory"]
        raw[name] = {"jobs": {}}
        for j in jobs:
            res = get_json(RAW + j + "/result.json")
            if res is not None:
                raw[name]["jobs"][j.split("/")[-1]] = res
        if i % 10 == 0:
            print("fetched", i + 1, "/", len(dirs), flush=True)
        time.sleep(0.1)
    json.dump(raw, open(path, "w"))
    return raw


def trial_task(trial_name):
    return trial_name.rsplit("__", 1)[0]


def job_version(job_name, eval_keys):
    tag = (job_name + " " + " ".join(eval_keys)).lower()
    if re.search(r"2[-.]1(\b|$)|tbench-2\.1", tag):
        return "2.1"
    if re.search(r"2[-.]0(\b|$)|terminal-bench-2(\b|$)|tb2", tag):
        return "2.0"
    return "unlabelled"


def build(raw):
    parsed, dropped_v21, dropped_missing, dropped_empty = [], [], [], []
    for name, d in raw.items():
        per_task, versions, dates = {}, set(), []
        for job, res in d["jobs"].items():
            evals = (res.get("stats") or {}).get("evals") or {}
            versions.add(job_version(job, evals.keys()))
            for key in ("started_at", "finished_at", "updated_at"):
                if res.get(key):
                    dates.append(res[key][:10].replace("-", ""))
                    break
            for ev in evals.values():
                for rew, names in ((ev.get("reward_stats") or {}).get("reward") or {}).items():
                    for t in names:
                        per_task.setdefault(trial_task(t), {})[t] = 1.0 if float(rew) >= 1.0 else 0.0
                for names in (ev.get("exception_stats") or {}).values():
                    for t in names:
                        per_task.setdefault(trial_task(t), {}).setdefault(t, 0.0)      # errored: failure unless it also got a reward
        if not per_task:                                   # job files without per-task results
            dropped_empty.append(name)
            continue
        if "2.1" in versions:
            dropped_v21.append(name)
            continue
        parsed.append({"name": name, "per_task": per_task, "date": min(dates) if dates else "", "n_jobs": len(d["jobs"]), "versions": sorted(versions)})
    # the benchmark's task set: tasks that at least half of the submissions ran (a few files name extra tasks)
    cover = Counter(t for p in parsed for t in p["per_task"])
    tasks = sorted(t for t, c in cover.items() if c >= len(parsed) / 2)
    extra = sorted(t for t in cover if t not in tasks)
    keep = []
    for p in parsed:
        if any(t not in p["per_task"] for t in tasks) or not p["date"]:
            dropped_missing.append(p["name"])
            continue
        keep.append(p)
    if not keep:
        raise RuntimeError(f"no submission covers all {len(tasks)} tasks; task counts per submission: {sorted(len(p['per_task']) for p in parsed)}")
    print(f"task set: {len(tasks)} tasks; {len(extra)} extra task names ignored: {extra[:5]}")
    systems, counts_all, exactly5 = [], [], 0
    for p in keep:
        rates = {t: float(np.mean(list(p["per_task"][t].values()))) for t in tasks}
        counts = [len(p["per_task"][t]) for t in tasks]
        counts_all += counts
        exactly5 += all(c == 5 for c in counts)
        all_trials = [v for t in tasks for v in p["per_task"][t].values()]
        systems.append({"name": p["name"], "date": p["date"], "rates": rates, "n_jobs": p["n_jobs"],
                        "trials": {t: len(p["per_task"][t]) for t in tasks},
                        "trials_per_task_min": min(counts), "trials_per_task_max": max(counts),
                        "mean_over_trials": float(np.mean(all_trials))})
    summary = {"n_submissions_found": len(raw), "dropped_no_results": dropped_empty, "dropped_version_21": dropped_v21,
               "dropped_missing_tasks": dropped_missing, "kept": len(keep), "n_tasks": len(tasks), "trials_per_task_min": int(min(counts_all)), "trials_per_task_max": int(max(counts_all)),
               "n_exactly_five_everywhere": int(exactly5), "n_multi_job": sum(1 for s in systems if s["n_jobs"] > 1),
               "version_labels": dict(Counter(v for p in parsed for v in p["versions"]))}
    json.dump(summary, open(os.path.join(RES, "tb_build.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("dropped")}, indent=1))
    print("dropped (no per-task results):", dropped_empty, "| dropped (2.1):", dropped_v21, "| dropped (missing tasks or date):", dropped_missing)
    return {"benchmark": "Terminal-Bench 2.0 leaderboard", "tasks": tasks, "systems": systems}


if __name__ == "__main__":
    raw = fetch_raw()
    board = build(raw)
    json.dump(board, open(os.path.join(DATA, "boards", "terminalbench.json"), "w"))
    sc = sorted(((sum(s["rates"].values()) / len(board["tasks"]), s["mean_over_trials"], s["name"], s["n_jobs"], s["trials_per_task_min"], s["trials_per_task_max"])
                 for s in board["systems"]), reverse=True)
    print("top 8 (mean of per-task rates, mean over trials, name, jobs, trials/task min-max):")
    for row in sc[:8]:
        print(f"  {row[0] * 100:5.1f}  {row[1] * 100:5.1f}  {row[2]}  jobs={row[3]}  trials {row[4]}-{row[5]}")
