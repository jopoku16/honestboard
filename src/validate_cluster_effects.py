"""Size of the record test when tasks within a repository move together.

Selected-step null from the fitted Verified model (holder and challenger equally able, challenger scored higher), with
system-by-repository random effects: each system gets an offset u_kg ~ N(0, tau^2) on the logit scale in each
repository g, on top of the shared task difficulty L_t and the copula, so its outcomes are correlated within a
repository beyond what shared difficulty gives. Repositories and their sizes are Verified's own.

tau is set from the data: for the 30 strongest systems, the between-repository variance of a system's residual pass
rate (observed minus fitted) beyond its binomial sampling variance, by the method of moments, converted to the logit
scale by the mean of p(1 - p); it is reported beside the results. The simulation runs at tau = 0, 0.25, 0.5 and 1 on the
logit scale, and the real board's ratio of cluster-robust to task-level standard error (neighbouring top-10 pairs) says
which level the data resemble.

Reported per tau: the rejection rate of the task-level selective test (which treats tasks as independent), of the
repository-level selective sign-flip test, and the mean ratio of the cluster-robust to the task-level standard error
of the paired difference. Writes results/validation_cluster.json.   usage: python validate_cluster_effects.py [pairs]
"""
import json
import os
import sys
import time

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "honestboard", "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from honestboard import Board  # noqa: E402
import validate_record_test as vrt  # noqa: E402

PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
path = os.path.join(ROOT, "data", "boards", "swebench_verified.json")
b0 = Board.from_json(path)
Y = b0.Y.astype(float)
K, T = Y.shape
clusters = b0.clusters
G = int(clusters.max()) + 1
ng = np.array([(clusters == g).sum() for g in range(G)])
L, base, sd_real, _ = vrt.fit_board(path)                 # L is the fitted logit per task, in task order
rho = vrt.calibrate_rho(L, base, sd_real)

# ---- tau from the board: residual pass rate per (system, repository) beyond binomial noise ----
order = np.argsort(-Y.mean(1), kind="stable")[:30]
r2, v, slopes = [], [], []
for k in order:
    a_k = vrt.solve_a(L, Y[k].mean())
    p = 1 / (1 + np.exp(-(a_k + L)))
    slopes.append(float((p * (1 - p)).mean()))
    for g in range(G):
        m = clusters == g
        r2.append(float((Y[k, m] - p[m]).mean() ** 2))
        v.append(float((p[m] * (1 - p[m])).sum() / m.sum() ** 2))
tau_p = float(np.sqrt(max(0.0, np.mean(r2) - np.mean(v))))
tau_logit = tau_p / float(np.mean(slopes))
print(f"repositories {G}, sizes {ng.tolist()}; residual spread per (system, repository) {100 * tau_p:.1f} points beyond noise, "
      f"tau = {tau_logit:.2f} on the logit scale; rho = {rho:.2f}", flush=True)

signs = ((np.arange(2 ** G)[:, None] >> np.arange(G)) & 1) * 2.0 - 1.0          # all sign patterns, 2^G x G
a0 = vrt.solve_a(L, base)


def simulate(tau, pairs, chunk=2000):
    n_sel = rej_task = rej_clu = 0
    ratios = []
    for start in range(0, pairs, chunk):
        r = min(chunk, pairs - start)
        u = vrt.rng.normal(size=(r, T))
        ys = []
        for _ in range(2):
            ukg = vrt.rng.normal(scale=tau, size=(r, G)) if tau > 0 else np.zeros((r, G))
            p = 1 / (1 + np.exp(-(a0 + L[None, :] + ukg[:, clusters])))
            z = np.sqrt(rho) * u + np.sqrt(1 - rho) * vrt.rng.normal(size=(r, T))
            ys.append((stats.norm.cdf(z) < p).astype(float))
        ya, yb = ys
        nb = ((yb == 1) & (ya == 0)).sum(1)
        na = ((ya == 1) & (yb == 0)).sum(1)
        sel = nb > na
        if not sel.any():
            continue
        nb_s, n_s = nb[sel], (nb + na)[sel]
        p_task = stats.binom.sf(nb_s - 1, n_s, 0.5) / stats.binom.sf(n_s // 2, n_s, 0.5)
        d = (yb - ya)[sel]
        Dg = np.stack([d[:, clusters == g].sum(1) for g in range(G)], 1)          # m x G repository sums
        null = Dg @ signs.T                                                       # m x 2^G
        obs = Dg.sum(1)
        c1 = (null >= obs[:, None] - 1e-9).sum(1)
        cpos = (null > 1e-9).sum(1)
        p_clu = np.where(cpos > 0, c1 / np.maximum(cpos, 1), 1.0)
        dbar = d.mean(1)
        se_cl2 = G / (G - 1) * ((Dg - ng[None, :] * dbar[:, None]) ** 2).sum(1) / T ** 2
        se_iid2 = d.var(1, ddof=1) / T
        n_sel += int(sel.sum())
        rej_task += int((p_task <= 0.05).sum())
        rej_clu += int((p_clu <= 0.05).sum())
        ratios.append(np.sqrt(se_cl2 / se_iid2))
    return {"tau_logit": float(tau), "selected_steps": n_sel, "reject_task_selective": rej_task / n_sel,
            "reject_cluster_selective": rej_clu / n_sel, "se_ratio_cluster_over_task": float(np.concatenate(ratios).mean())}


# real-data reference: ratio of the cluster-robust to the task-level standard error for neighbouring top-10 pairs
sig_i, sig_c = b0.sigma(False), b0.sigma(True)
top = np.argsort(-Y.mean(1), kind="stable")[:10]
real_ratios = [np.sqrt((sig_c[a, a] + sig_c[c, c] - 2 * sig_c[a, c]) / (sig_i[a, a] + sig_i[c, c] - 2 * sig_i[a, c])) for a, c in zip(top[:-1], top[1:])]
real_ratio = float(np.median(real_ratios))
print(f"real board: median SE ratio cluster/task for neighbouring top-10 pairs {real_ratio:.2f}", flush=True)

out = {"design": {"board": "swebench_verified", "G": G, "T": T, "rho": rho, "base": base, "pairs": PAIRS,
                  "tau_est_pts": 100 * tau_p, "tau_est_logit": tau_logit, "repository_sizes": ng.tolist(),
                  "real_se_ratio_top10": real_ratio}, "levels": []}
for tau in (0.0, 0.25, 0.5, 1.0):                       # absolute levels on the logit scale (the data estimate is reported beside them)
    t0 = time.time()
    res = simulate(tau, PAIRS)
    out["levels"].append(res)
    print(f"tau {tau:<4}: {res['selected_steps']} null record steps; task-level test rejects {100 * res['reject_task_selective']:.1f}%, "
          f"repository-level {100 * res['reject_cluster_selective']:.1f}%; SE ratio cluster/task {res['se_ratio_cluster_over_task']:.2f} ({time.time() - t0:.0f}s)", flush=True)
with open(os.path.join(ROOT, "results", "validation_cluster.json"), "w") as f:
    json.dump(out, f, indent=2)
