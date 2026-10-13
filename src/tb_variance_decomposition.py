"""Terminal-Bench 2 scores each task with repeated trials, which lets us split the variance of a score into task
sampling and run-to-run randomness of the agent.

Per task t and system k, k_t of n_t trials passed. k_t(n_t - k_t)/(n_t(n_t - 1)) is unbiased for p_t(1 - p_t), the
within-task (run) variance of one trial; Var_t(r_t) = Var_t(p_t) + E_t[p_t(1 - p_t)/n_t] gives the between-task
variance of the true pass probability by moments. For a SINGLE-RUN score the per-task variance is
Var_t(p_t) + E_t[p_t(1 - p_t)], so the run share is E[p(1-p)] / (Var(p) + E[p(1-p)]). For the score the leaderboard
actually reports (n_t trials per task) the run term is E[p(1-p)/n_t]. The same decomposition is applied to the
paired difference between neighbouring systems in the top ten. Systems with fewer than two trials on any task are
skipped. Writes results/tb_variance.json.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
board = json.load(open(os.path.join(ROOT, "data", "boards", "terminalbench.json"), encoding="utf-8"))
tasks = board["tasks"]
systems = [s for s in board["systems"] if min(s["trials"].values()) >= 2]
systems.sort(key=lambda s: -np.mean([s["rates"][t] for t in tasks]))
R = np.array([[s["rates"][t] for t in tasks] for s in systems])
N = np.array([[s["trials"][t] for t in tasks] for s in systems], dtype=float)
Kc = np.rint(R * N)
pq = Kc * (N - Kc) / (N * (N - 1))                     # unbiased for p(1-p) per task and system

single, reported, var_p_all = [], [], []
top = list(range(min(20, len(systems))))
for i in top:
    E_pq = pq[i].mean()
    E_pq_over_n = (pq[i] / N[i]).mean()
    var_p = max(R[i].var(ddof=1) - E_pq_over_n, 0.0)
    single.append(E_pq / (var_p + E_pq))
    reported.append(E_pq_over_n / (var_p + E_pq_over_n))
    var_p_all.append(var_p)

pair_single, pair_reported = [], []
for a, c in zip(range(9), range(1, 10)):
    if c >= len(systems):
        break
    d = R[c] - R[a]
    E_pq_pair = (pq[a] + pq[c]).mean()
    E_pq_pair_n = (pq[a] / N[a] + pq[c] / N[c]).mean()
    var_true = max(d.var(ddof=1) - E_pq_pair_n, 0.0)
    pair_single.append(E_pq_pair / (var_true + E_pq_pair))
    pair_reported.append(E_pq_pair_n / (var_true + E_pq_pair_n))

out = {"n_systems_single": len(top), "n_pairs": len(pair_single), "n_systems_with_repeats": len(systems),
       "run_share_single_median": float(np.median(single)), "run_share_single_min": float(np.min(single)), "run_share_single_max": float(np.max(single)),
       "run_share_five_median": float(np.median(reported)),
       "paired_run_share_single_median": float(np.median(pair_single)), "paired_run_share_five_median": float(np.median(pair_reported)),
       "mean_within_task_var_top20": float(np.mean([pq[i].mean() for i in top])),
       "mean_between_task_var_top20": float(np.mean(var_p_all)),
       "mean_trials_per_task_top20": float(np.mean([N[i].mean() for i in top]))}
with open(os.path.join(ROOT, "results", "tb_variance.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=1))
