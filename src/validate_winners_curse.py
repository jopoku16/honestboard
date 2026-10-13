"""Does the winner's-curse correction work? Validation on one board.

V1   Known truth, observed matrix as population. A "benchmark" is n tasks drawn with replacement. (Favourable for a
     bootstrap method: true gaps equal the observed gaps.)
V1b  Known truth, equal leaders. The top m systems are given EQUAL true ability (the hardest case for a winner's
     curse); outcomes are simulated from a task-difficulty model fitted to the board.
V2   Real held-out tasks: random halves of the real tasks (select on one, score on the other).
V3   Real held-out repositories (boards with cluster labels): halves made of different repositories. The selection
     board keeps its repository labels and the corrections use cluster resampling. The validation score is adjusted
     by the field-wide difficulty difference between the halves (mean over all systems); V3_raw reports the same
     estimates against the unadjusted score, and V3_iid the task-resampling (iid) corrections for comparison.
V4   Is the observed winner a true best, and does the confidence set cover a true best when the winner is not?

Estimators for the winner's true score: naive (reported), bootstrap (plug-in), iterated bootstrap, conditional
(polyhedral), the hybrid of Andrews, Kitagawa and McCloskey (2024) and the zoom interval of Zrnic and Fithian (2025)
(whose point estimate is the reported score, so its error columns repeat the naive ones). Every summary gives the
Monte Carlo standard error of the mean error and the mean and median error on the common subset of repetitions in
which all estimators are defined.
Outputs results/validation_<tag>.json (or results/validation_<tag>__<stage>.json when a stage is given).

usage: python validate_winners_curse.py <verified|lite|terminalbench|livebench> <R> [all|V1|V1b|V4|V23]
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "honestboard", "src"))
from honestboard import Board  # noqa: E402

RES = os.path.join(ROOT, "results")
FILES = {"verified": "swebench_verified.json", "lite": "swebench_lite.json", "terminalbench": "terminalbench.json",
         "livebench": "livebench.json"}
B_BOOT, B_OUT, B_IN = 300, 100, 100
METHODS = ("naive", "bootstrap", "iterated", "conditional", "hybrid", "zoom")


def summarize(rows, label, coverage_meaningful=True):
    E = {m: np.array([r[m]["err"] for r in rows], dtype=float) for m in METHODS}
    common = np.all([np.isfinite(E[m]) for m in METHODS], axis=0)
    out = {"_n_common": int(common.sum()), "_coverage_meaningful": bool(coverage_meaningful)}
    for m in METHODS:
        err = E[m]
        cov = np.array([r[m]["cov"] for r in rows], dtype=float)
        ln = np.array([r[m]["len"] for r in rows], dtype=float)
        fin = np.isfinite(err)
        nf = int(fin.sum())
        out[m] = {
            "n": int(len(rows)), "finite": nf,
            "bias_pts": float(err[fin].mean() * 100) if nf else None,
            "se_bias_pts": float(err[fin].std(ddof=1) / np.sqrt(nf) * 100) if nf > 1 else None,
            "median_err_pts": float(np.median(err[fin]) * 100) if nf else None,
            "rmse_pts": float(np.sqrt((err[fin] ** 2).mean()) * 100) if nf else None,
            "coverage": float(np.nanmean(cov)) if (coverage_meaningful and np.isfinite(cov).any()) else None,
            "median_len_pts": float(np.median(ln[np.isfinite(ln)]) * 100) if np.isfinite(ln).any() else None,
            "bias_common_pts": float(err[common].mean() * 100) if common.any() else None,
            "median_common_pts": float(np.median(err[common]) * 100) if common.any() else None,
        }
    print(f"--- {label}  (common subset {int(common.sum())}/{len(rows)})", flush=True)
    for m in METHODS:
        v = out[m]
        cov = "n/a" if v["coverage"] is None else f"{v['coverage']:.3f}"
        print(f"  {m:12s} bias {v['bias_pts']:+.2f} (se {v['se_bias_pts']:.2f})  median {v['median_err_pts']:+.2f}  "
              f"rmse {v['rmse_pts']:.2f}  cov {cov}  common bias {v['bias_common_pts']:+.2f} median {v['median_common_pts']:+.2f}  "
              f"finite {v['finite']}/{v['n']}", flush=True)
    return out


def estimates(b, rng, cluster=False):
    """Winner and the four estimates (value, ci) from a Board."""
    sc = b.scores()
    w = int(np.argmax(sc + rng.random(b.K) * 1e-12))
    sd = float(np.sqrt(b.sigma(cluster)[w, w]))
    seed = int(rng.integers(1 << 30))
    boot = b.winner_bootstrap(B=B_BOOT, seed=seed, w=w, cluster=cluster)
    boot2 = b.winner_bootstrap(B=B_BOOT, seed=seed, w=w, cluster=cluster, iterate=True, B_outer=B_OUT, B_inner=B_IN)
    try:
        cond = b.winner_conditional(w=w, cluster=cluster)
        cond_est, cond_ci = cond["estimate"], cond["ci"]
    except ValueError:
        cond_est, cond_ci = np.nan, (np.nan, np.nan)
    try:
        hyb = b.winner_hybrid(w=w, cluster=cluster, seed=seed)
        hyb_est, hyb_ci = hyb["estimate"], hyb["ci"]
    except ValueError:
        hyb_est, hyb_ci = np.nan, (np.nan, np.nan)
    zoom = b.winner_zoom(w=w, cluster=cluster, seed=seed)
    return w, {"naive": (sc[w], (sc[w] - 1.96 * sd, sc[w] + 1.96 * sd)), "bootstrap": (boot["estimate"], boot["ci"]),
               "iterated": (boot2["estimate"], boot2["ci"]), "conditional": (cond_est, cond_ci),
               "hybrid": (hyb_est, hyb_ci), "zoom": (zoom["estimate"], zoom["ci"])}


def score_row(ests, truth):
    """Per-method error, coverage and length. A method with a non-finite estimate is recorded as undefined (NaN)."""
    row = {}
    for m, (est, ci) in ests.items():
        lo, hi = ci
        if not np.isfinite(est) or np.isnan(lo) or np.isnan(hi):
            row[m] = {"err": np.nan, "cov": np.nan, "len": np.nan}
            continue
        row[m] = {"err": est - truth, "cov": float(lo <= truth <= hi),
                  "len": (hi - lo) if (np.isfinite(lo) and np.isfinite(hi)) else np.inf}
    return row


def bunched_model(Y, m=20, ref=30):
    """Probability matrix P (K x T) in which the top m systems share one true ability; difficulty from the board."""
    K, T = Y.shape
    sc = Y.mean(1)
    order = np.argsort(-sc, kind="stable")
    mt = np.clip(Y[order[:ref]].mean(0), 0.03, 0.97)
    L = np.log(mt / (1 - mt))
    theta = sc.copy()
    theta[order[:m]] = sc[order[:m]].mean()
    P = np.empty((K, T))
    for k in range(K):
        lo, hi = -15.0, 15.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if (1 / (1 + np.exp(-(mid + L)))).mean() < theta[k]:
                lo = mid
            else:
                hi = mid
        P[k] = 1 / (1 + np.exp(-(mid + L)))
    return P, set(order[:m].tolist())


def run_board(tag, R, stage="all", seed=11):
    b0 = Board.from_json(os.path.join(ROOT, "data", "boards", FILES[tag]))
    Y = b0.Y.astype(float)
    K, T = Y.shape
    trials = 5 if tag == "terminalbench" else 1
    theta = Y.mean(1)
    rng = np.random.default_rng(seed + (0 if stage == "all" else sum(map(ord, stage))))
    result = {"board": tag, "K": int(K), "T": int(T), "R": R}
    ns = [100, 250, T] if T >= 250 else [T]
    on = lambda s: stage in ("all", s)

    if on("V1"):
        result["V1"] = {}
        for n in ns:
            rows, t0 = [], time.time()
            for _ in range(R):
                idx = rng.integers(0, T, n)
                b = Board(b0.names, b0.Y[:, idx])
                w, ests = estimates(b, rng)
                rows.append(score_row(ests, theta[w]))
            result["V1"][str(n)] = summarize(rows, f"{tag} V1 known truth n={n} ({time.time() - t0:.0f}s)")

    P, best_set = bunched_model(Y, m=min(20, K // 3))
    truth_b = P.mean(1)
    result["V1b"] = {"n_equal_top": len(best_set)}
    for n in ([x for x in ns if x >= 250] or ns[-1:]) if on("V1b") else []:
        rows, t0 = [], time.time()
        for _ in range(R):
            idx = rng.integers(0, T, n)
            Ys = rng.binomial(trials, P[:, idx]) / trials
            b = Board(b0.names, Ys)
            w, ests = estimates(b, rng)
            rows.append(score_row(ests, truth_b[w]))
        result["V1b"][str(n)] = summarize(rows, f"{tag} V1b equal leaders n={n} ({time.time() - t0:.0f}s)")

    if trials == 1 and on("V4"):
        best = set(np.where(theta >= theta.max() - 1e-12)[0].tolist())
        result["V4"] = {}
        for n in ns:
            win_true, cover_when_wrong, size = [], [], []
            t0 = time.time()
            for _ in range(max(300, R // 2)):
                idx = rng.integers(0, T, n)
                b = Board(b0.names, b0.Y[:, idx])
                sc = b.scores()
                w = int(np.argmax(sc + rng.random(b.K) * 1e-12))
                is_true = w in best
                cs = b.confidence_set_best(alpha=0.05, B=300, seed=int(rng.integers(1 << 30)))
                members = {b.names.index(m) for m in cs["members"]}
                win_true.append(is_true)
                if not is_true:
                    cover_when_wrong.append(bool(members & best))
                size.append(cs["size"])
            result["V4"][str(n)] = {"p_winner_is_true_best": float(np.mean(win_true)),
                                    "set_covers_true_best_when_winner_wrong": float(np.mean(cover_when_wrong)) if cover_when_wrong else None,
                                    "n_winner_wrong": len(cover_when_wrong), "median_set_size": float(np.median(size)),
                                    "n_true_best": len(best)}
            print(f"--- {tag} V4 n={n} ({time.time() - t0:.0f}s): P(winner is a true best)={np.mean(win_true):.2f}, "
                  f"set covers a true best when winner wrong={np.mean(cover_when_wrong) if cover_when_wrong else float('nan'):.2f} "
                  f"(n={len(cover_when_wrong)}), median set size={np.median(size):.0f}", flush=True)

    if on("V23"):
        rows, t0 = [], time.time()
        for _ in range(R):
            perm = rng.permutation(T)
            s_idx, v_idx = perm[: T // 2], perm[T // 2:]
            bs = Board(b0.names, b0.Y[:, s_idx])
            w, ests = estimates(bs, rng)
            rows.append(score_row(ests, Y[w, v_idx].mean()))
        result["V2"] = summarize(rows, f"{tag} V2 held-out tasks ({time.time() - t0:.0f}s)", coverage_meaningful=False)

        if b0.clusters is not None:
            import itertools
            rows_c, rows_raw, rows_iid, t0 = [], [], [], time.time()
            G = int(b0.clusters.max()) + 1
            sizes = np.bincount(b0.clusters, minlength=G)
            # balanced splits: repository subsets whose selection half holds 40 to 60 percent of the tasks
            cands = []
            if G <= 16:
                for r in range(2, G - 1):                # at least two repositories on each side
                    for comb in itertools.combinations(range(G), r):
                        s = int(sizes[list(comb)].sum())
                        if 0.4 * T <= s <= 0.6 * T:
                            cands.append(comb)
            sel_sizes = []
            for _ in range(R):
                if cands:
                    pick = np.array(cands[rng.integers(len(cands))])
                else:                                     # too many repositories to enumerate: random half
                    pick = rng.permutation(G)[: G // 2]
                mask = np.isin(b0.clusters, pick)
                s_idx, v_idx = np.where(mask)[0], np.where(~mask)[0]
                sel_sizes.append(int(mask.sum()))
                sub = np.unique(b0.clusters[s_idx], return_inverse=True)[1]
                bs = Board(b0.names, b0.Y[:, s_idx], None, sub)
                w, ests = estimates(bs, rng, cluster=True)
                shift = Y[:, v_idx].mean() - Y[:, s_idx].mean()            # field-wide difficulty difference between halves
                rows_c.append(score_row(ests, Y[w, v_idx].mean() - shift))
                rows_raw.append(score_row(ests, Y[w, v_idx].mean()))
                bi = Board(b0.names, b0.Y[:, s_idx])
                w2, ests2 = estimates(bi, rng)
                rows_iid.append(score_row(ests2, Y[w2, v_idx].mean() - shift))
            result["V3_split"] = {"n_candidate_splits": len(cands), "min": int(min(sel_sizes)), "max": int(max(sel_sizes)), "G": int(G)}
            print(f"V3 splits: {len(cands)} balanced repository subsets; selection half {min(sel_sizes)}-{max(sel_sizes)} tasks", flush=True)
            result["V3"] = summarize(rows_c, f"{tag} V3 held-out repositories, cluster corrections, shift-adjusted ({time.time() - t0:.0f}s)", coverage_meaningful=False)
            result["V3_raw"] = summarize(rows_raw, f"{tag} V3_raw same estimates, unadjusted target", coverage_meaningful=False)
            result["V3_iid"] = summarize(rows_iid, f"{tag} V3_iid task-resampling corrections, shift-adjusted", coverage_meaningful=False)

    name = f"validation_{tag}.json" if stage == "all" else f"validation_{tag}__{stage}.json"
    with open(os.path.join(RES, name), "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    tags = sys.argv[1].split(",")
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    stage = sys.argv[3] if len(sys.argv) > 3 else "all"
    for tag in tags:
        run_board(tag, R, stage)
