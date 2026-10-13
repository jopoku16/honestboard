"""Development selection, measured where the data allow it: Terminal-Bench submissions that ran the whole benchmark
several times (several jobs). A team that reported its best job instead of the mean would inflate its score by
(best job - mean job). We report that inflation, and a model of "try V variants, submit the best" calibrated to
the observed between-job spread: expected inflation = E[max of V normals] * sd_between_jobs.

Writes results/tb_job_selection.json.
"""
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw = json.load(open(os.path.join(ROOT, "data", "raw_terminalbench.json"), encoding="utf-8"))
build = json.load(open(os.path.join(ROOT, "results", "tb_build.json")))
dropped = set(build["dropped_version_21"]) | set(build["dropped_missing_tasks"])


def trial_task(t):
    return t.rsplit("__", 1)[0]


rows = []
for name, d in raw.items():
    if name in dropped or len(d["jobs"]) < 2:
        continue
    job_scores = []
    for job, res in d["jobs"].items():
        per_task = defaultdict(dict)
        for ev in ((res.get("stats") or {}).get("evals") or {}).values():
            for rew, names in ((ev.get("reward_stats") or {}).get("reward") or {}).items():
                for t in names:
                    per_task[trial_task(t)][t] = 1.0 if float(rew) >= 1.0 else 0.0
            for names in (ev.get("exception_stats") or {}).values():
                for t in names:
                    per_task[trial_task(t)].setdefault(t, 0.0)
        if len(per_task) < 80:                     # a partial job (one submission has a 29-trial job)
            continue
        job_scores.append(float(np.mean([np.mean(list(v.values())) for v in per_task.values()])))
    if len(job_scores) >= 2:
        rows.append({"name": name, "n_jobs": len(job_scores), "jobs": job_scores, "mean": float(np.mean(job_scores)),
                     "best": float(np.max(job_scores)), "sd": float(np.std(job_scores, ddof=1))})

inflation = [r["best"] - r["mean"] for r in rows]
sds = [r["sd"] for r in rows]
k_vals = [r["n_jobs"] for r in rows]
# expected maximum of V standard normals (V = 1..10), by numerical integration
V = list(range(1, 11))
emax = []
for v in V:
    x = np.linspace(-8, 8, 20001)
    pdf = v * stats.norm.pdf(x) * stats.norm.cdf(x) ** (v - 1)
    emax.append(float(np.trapezoid(x * pdf, x)))
sd_med = float(np.median(sds))
out = {"n_submissions_with_full_jobs": len(rows), "jobs_per_submission_median": float(np.median(k_vals)),
       "best_minus_mean_median_pts": 100 * float(np.median(inflation)), "best_minus_mean_mean_pts": 100 * float(np.mean(inflation)),
       "best_minus_mean_max_pts": 100 * float(np.max(inflation)),
       "between_job_sd_median_pts": 100 * sd_med, "between_job_sd_iqr_pts": [100 * float(np.percentile(sds, 25)), 100 * float(np.percentile(sds, 75))],
       "expected_inflation_pts_by_V": {str(v): 100 * sd_med * e for v, e in zip(V, emax)},
       "rows": rows}
with open(os.path.join(ROOT, "results", "tb_job_selection.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
