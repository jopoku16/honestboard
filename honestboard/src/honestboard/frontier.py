"""Who is really on the cost-accuracy frontier?

A leaderboard that reports cost beside accuracy is read through its Pareto frontier: the systems that no cheaper system
beats. Accuracy is estimated from a sample of tasks, so a frontier drawn from point estimates is an estimate too. From
per-task outcomes and a cost per system (or per task) this module gives:

1. The observed frontier: system i is on it when no system with cost at most cost_i has accuracy at least acc_i, with
   at least one of the two strict.
2. A confidence set for the frontier at level 1 - alpha: system i stays in unless some cheaper-or-equal system beats it
   by more than a simultaneous paired margin c * se_ij, where se_ij is the standard error of the paired accuracy
   difference and c is the (1 - alpha) bootstrap quantile of the largest studentised deviation of any cheaper rival
   from any system on the board. This is the multiple-comparisons-with-the-best construction behind
   `Board.confidence_set_best`, with the rival set restricted to cheaper systems and the margin taken jointly over
   systems: every truly undominated system is kept at once with probability about 1 - alpha, so the set contains the
   whole true frontier. Two smaller sets with weaker guarantees are returned beside it: `frontier_set_boardwide` (the
   largest per-system margin) and `frontier_set_persystem` (each system's own margin), each keeping a given truly
   undominated system with probability about 1 - alpha. The set always contains the observed frontier.
3. The undominated frequency: the share of task resamples in which system i is on the observed frontier. A resampling
   frequency, not a probability of being Pareto-optimal.
4. The cheapest plausible best: the cheapest member of the confidence set of the best (no cost), what a buyer who wants
   "as good as the best, as cheap as possible" is asking for.

Costs given per system are treated as fixed. Costs given per task are resampled with the outcomes, so the cost ordering
itself is uncertain and a rival counts as cheaper in a resample only if it is cheaper there. Tasks are treated as
exchangeable draws; dependence between tasks (repositories) makes the standard errors optimistic, as in any task-level
analysis.
"""
from __future__ import annotations

import numpy as np


def _bootstrap_means(Y, B, rng, chunk=200):
    K, T = Y.shape
    out = np.empty((K, B))
    done = 0
    while done < B:
        m = min(chunk, B - done)
        ix = rng.integers(0, T, size=(m, T))
        out[:, done:done + m] = Y[:, ix].mean(axis=2)
        done += m
    return out


def _bootstrap_clusters(Y, C, B, rng, clusters):
    """Bootstrap accuracies (and per-task-cost means) by resampling task CLUSTERS with replacement: (K x B, K x B or None)."""
    labels = np.asarray(clusters)
    groups = [np.where(labels == g)[0] for g in np.unique(labels)]
    G = len(groups)
    acc = np.empty((Y.shape[0], B))
    cost = np.empty((Y.shape[0], B)) if C is not None else None
    for b in range(B):
        ix = np.concatenate([groups[g] for g in rng.integers(0, G, size=G)])
        acc[:, b] = Y[:, ix].mean(axis=1)
        if C is not None:
            cost[:, b] = C[:, ix].mean(axis=1)
    return acc, cost


def _bootstrap_pairs(Y, C, B, rng, chunk=200):
    """Bootstrap accuracies and per-task-cost totals on the same resamples: (K x B, K x B)."""
    K, T = Y.shape
    acc = np.empty((K, B))
    cost = np.empty((K, B))
    done = 0
    while done < B:
        m = min(chunk, B - done)
        ix = rng.integers(0, T, size=(m, T))
        acc[:, done:done + m] = Y[:, ix].mean(axis=2)
        cost[:, done:done + m] = C[:, ix].mean(axis=2)
        done += m
    return acc, cost


def observed_frontier(acc, cost):
    """Boolean mask: system i is undominated (no cheaper-or-equal system with at least its accuracy, one strict)."""
    acc = np.asarray(acc, float)
    cost = np.asarray(cost, float)
    K = len(acc)
    cheaper_eq = cost[None, :] <= cost[:, None]
    better_eq = acc[None, :] >= acc[:, None]
    strict = (cost[None, :] < cost[:, None]) | (acc[None, :] > acc[:, None])
    dom = cheaper_eq & better_eq & strict & ~np.eye(K, dtype=bool)
    return ~dom.any(axis=1)


def pareto_frontier(board, costs, alpha: float = 0.05, B: int = 4000, seed: int = 0, names=None, clusters=None) -> dict:
    """Confidence set for the Pareto frontier of a board.

    Args:
        board: a `Board` (systems-by-tasks outcomes in [0, 1]).
        costs: {system: cost} with one number per system, or {system: array of per-task costs} (length T).
        alpha: one minus the confidence level.
        B: bootstrap resamples over tasks.
        names: restrict to these systems (default: every system that has a cost).
    """
    names = [n for n in (names or board.names) if n in costs]
    if len(names) < 2:
        raise ValueError("the frontier needs costs for at least two systems on the board")
    idx = [board.index(n) for n in names]
    Y = np.asarray(board.Y, dtype=float)[idx]
    K, T = Y.shape
    per_task = any(np.ndim(costs[n]) > 0 for n in names)
    if per_task:
        C = np.array([np.asarray(costs[n], float) if np.ndim(costs[n]) > 0 else np.full(T, float(costs[n]) / T) for n in names])
        if C.shape != (K, T):
            raise ValueError(f"per-task costs must have one entry per task ({T}); got {C.shape}")
        cost = C.mean(axis=1)                                 # mean cost per task; the ordering is the same as the total
    else:
        C = None
        cost = np.array([float(costs[n]) for n in names])
    acc = Y.mean(axis=1)
    rng = np.random.default_rng(seed)
    if clusters is not None:                                      # resample task groups (repositories, capsules) instead of tasks
        S, SC = _bootstrap_clusters(Y, C if per_task else None, B, rng, clusters)
    elif per_task:
        S, SC = _bootstrap_pairs(Y, C, B, rng)
    else:
        S = _bootstrap_means(Y, B, rng)
        SC = None

    Sigma = np.cov(Y) / T if K > 1 else np.zeros((1, 1))
    var_d = np.diag(Sigma)[:, None] + np.diag(Sigma)[None, :] - 2 * Sigma
    se = np.sqrt(np.maximum(var_d, 0.0))
    se[se == 0] = 1e-12
    eye = np.eye(K, dtype=bool)
    cheaper = (cost[None, :] <= cost[:, None]) & ~eye            # cheaper[i, j]: j is a cheaper-or-equal rival of i

    dev = ((S[None, :, :] - S[:, None, :]) - (acc[None, :] - acc[:, None])[:, :, None]) / se[:, :, None]   # [i, j, b]
    if per_task:                                                  # the rival set varies with the resample
        cheaper_b = (SC[None, :, :] <= SC[:, None, :]) & ~eye[:, :, None]
        masked = np.where(cheaper_b, dev, -np.inf)
    else:
        masked = np.where(cheaper[:, :, None], dev, -np.inf)
    M = masked.max(axis=1)                                        # K x B
    has_rival = np.isfinite(M).any(axis=1)
    q = np.array([np.quantile(M[i][np.isfinite(M[i])], 1 - alpha) if has_rival[i] else -np.inf for i in range(K)])
    c_board = float(max(0.0, q[has_rival].max())) if has_rival.any() else 0.0     # the largest per-system margin
    M_all = np.where(has_rival[:, None], M, -np.inf).max(axis=0)                 # B: the largest deviation on the board
    fin = np.isfinite(M_all)
    c = float(max(0.0, np.quantile(M_all[fin], 1 - alpha))) if fin.any() else 0.0   # joint margin: all truly undominated kept together

    gap = (acc[None, :] - acc[:, None]) / se                      # [i, j]: how far j sits above i, in se units
    beaten_by = np.where(cheaper, gap, -np.inf)
    worst = beaten_by.max(axis=1)
    in_set = ~(worst > c)                                         # the confidence set for the frontier (joint)
    in_set_board = ~(worst > c_board)
    q_eff = np.where(has_rival, np.maximum(np.where(np.isfinite(q), q, 0.0), 0.0), 0.0)
    in_set_ps = ~(worst > q_eff)                                  # per-system margins: valid per system, smaller sets
    observed = observed_frontier(acc, cost)
    if per_task:
        freq = np.array([observed_frontier(S[:, b], SC[:, b]) for b in range(B)]).mean(axis=0)
    else:
        freq = np.array([observed_frontier(S[:, b], cost) for b in range(B)]).mean(axis=0)

    dev_all = np.where(~eye[:, :, None], dev, -np.inf).max(axis=1)
    c_all = float(max(0.0, np.quantile(dev_all, 1 - alpha, axis=1).max())) if K > 1 else 0.0
    best_set = ~(np.where(~eye, gap, -np.inf).max(axis=1) > c_all)
    cheapest_best = min((k for k in range(K) if best_set[k]), key=lambda k: (cost[k], -acc[k]))
    top = int(np.argmax(acc))

    systems = []
    for k in sorted(range(K), key=lambda k: cost[k]):
        rival = int(np.argmax(beaten_by[k])) if np.isfinite(worst[k]) else None
        systems.append({
            "system": names[k], "cost": float(cost[k]), "accuracy": float(acc[k]),
            "on_observed_frontier": bool(observed[k]), "in_frontier_set": bool(in_set[k]),
            "undominated_frequency": float(freq[k]), "in_best_set": bool(best_set[k]),
            "strongest_cheaper_rival": names[rival] if rival is not None else None,
            "rival_lead": float(acc[rival] - acc[k]) if rival is not None else None,
            "rival_lead_se": float(worst[k]) if rival is not None else None,
        })
    return {
        "alpha": alpha, "systems": K, "tasks": T, "B": B, "per_task_costs": per_task, "clustered": clusters is not None,
        "margin_se": c, "margin_se_boardwide": c_board, "margin_se_best": c_all,
        "observed_frontier": [names[k] for k in range(K) if observed[k]],
        "frontier_set": [names[k] for k in range(K) if in_set[k]],
        "frontier_set_boardwide": [names[k] for k in range(K) if in_set_board[k]],
        "frontier_set_persystem": [names[k] for k in range(K) if in_set_ps[k]],
        "margins_persystem": {names[k]: float(q_eff[k]) for k in range(K)},
        "best_set": [names[k] for k in range(K) if best_set[k]],
        "top": {"system": names[top], "accuracy": float(acc[top]), "cost": float(cost[top])},
        "cheapest_plausible_best": {"system": names[cheapest_best], "accuracy": float(acc[cheapest_best]), "cost": float(cost[cheapest_best])},
        "table": systems,
    }


# ---- the frontier winner's curse -------------------------------------------------------------------------------------
def frontier_winner(board, costs, system, alpha: float = 0.05, B: int = 4000, seed: int = 0, iterate: bool = True,
                    B_outer: int = 150, B_inner: int = 150) -> dict:
    """Selection-adjusted accuracy of a system that sits on the observed frontier.

    A system reported as "on the frontier" was selected for having the highest accuracy among the systems that cost
    no more than it (its pool), so its reported accuracy is inflated, as a leaderboard winner's is. This is the
    winner's-curse problem of `Board.winner_adjusted` with the pool in place of the whole board:

    * conditional: the polyhedral (truncated-normal) estimate and interval given that the system is the maximum of
      its pool (selective.winner_adjusted with the pool as the candidate set);
    * bootstrap: in each task resample the winner of the pool is found again and its excess over its own observed
      accuracy measures the selection bias; the reported accuracy is corrected by the mean excess, with a basic
      bootstrap interval; iterate=True removes the bias left by the first correction with a second bootstrap level.

    Costs are fixed numbers per system (per-task costs enter by their mean), so the pool is fixed. A system with no
    cheaper-or-equal rival (the cheapest on the board) has a pool of one and nothing to correct.
    """
    from . import selective
    names = [n for n in board.names if n in costs]
    if len(names) < 2:
        raise ValueError("the frontier needs costs for at least two systems on the board")
    idx = np.array([board.index(n) for n in names])
    Y = board.Y[idx].astype(float)
    K, T = Y.shape
    cost = np.array([float(np.mean(costs[n])) if np.ndim(costs[n]) > 0 else float(costs[n]) for n in names])
    k = names.index(system) if isinstance(system, str) else int(system)
    acc = Y.mean(axis=1)
    pool = np.where(cost <= cost[k])[0]
    if acc[k] < acc[pool].max() - 1e-12:
        raise ValueError(f"{names[k]} is not on the observed frontier: a cheaper-or-equal system is more accurate")
    Sigma = np.cov(Y) / T if K > 1 else np.zeros((1, 1))
    sd = float(np.sqrt(max(Sigma[k, k], 0.0)))
    zq = float(_norm_ppf(1 - alpha / 2))
    out = {"system": names[k], "observed": float(acc[k]), "se": sd, "cost": float(cost[k]), "pool_size": int(len(pool)),
           "pool": [names[j] for j in pool], "naive_ci": (float(acc[k] - zq * sd), float(acc[k] + zq * sd)),
           "strongest_cheaper_rival": None, "rival_accuracy": None}
    rivals = [j for j in pool if j != k]
    if rivals:
        r = max(rivals, key=lambda j: acc[j])
        out["strongest_cheaper_rival"] = names[r]
        out["rival_accuracy"] = float(acc[r])
    if len(pool) == 1 or sd <= 0:                                     # nothing was selected against, or no variability
        out.update({"conditional": {"estimate": float(acc[k]), "ci": out["naive_ci"], "degenerate": False},
                    "bootstrap": {"estimate": float(acc[k]), "bias": 0.0, "ci": out["naive_ci"], "iterated": bool(iterate)}})
        return out
    cond = selective.winner_adjusted(acc, Sigma, w=k, alpha=alpha, candidates=pool)
    out["conditional"] = {"estimate": float(cond["estimate"]), "ci": (float(cond["ci"][0]), float(cond["ci"][1])),
                          "degenerate": bool(cond.get("degenerate", False)), "lower_gap": float(cond["lower_gap"])}
    rng = np.random.default_rng(seed)
    S = _bootstrap_means(Y, B, rng)                                   # K x B
    Sp = S[pool]
    wstar = pool[np.argmax(Sp + rng.random(Sp.shape) * 1e-12, axis=0)]
    delta = S[wstar, np.arange(B)] - acc[wstar]
    bias = float(delta.mean())
    est = float(acc[k] - bias)
    if iterate:
        c_out = rng.multinomial(T, np.full(T, 1.0 / T), size=B_outer).astype(float)
        b1 = np.empty(B_outer)
        for b in range(B_outer):
            w_out = c_out[b]
            p_b = (Y @ w_out) / w_out.sum()
            c_in = rng.multinomial(T, w_out / w_out.sum(), size=B_inner).astype(float)
            S_in = (c_in @ Y.T) / c_in.sum(axis=1, keepdims=True)         # B_inner x K
            S_in_pool = S_in[:, pool]
            ws = pool[np.argmax(S_in_pool + rng.random(S_in_pool.shape) * 1e-12, axis=1)]
            b1[b] = float((S_in[np.arange(B_inner), ws] - p_b[ws]).mean())
        est = float(acc[k] - 2 * bias + b1.mean())
    centred = delta - bias
    lo = float(est - np.quantile(centred, 1 - alpha / 2))
    hi = float(est - np.quantile(centred, alpha / 2))
    out["bootstrap"] = {"estimate": est, "bias": float(acc[k] - est), "ci": (lo, hi), "iterated": bool(iterate)}
    return out


def _norm_ppf(q):
    from scipy import stats
    return stats.norm.ppf(q)


def frontier_curse(board, costs, alpha: float = 0.05, B: int = 4000, seed: int = 0, iterate: bool = True,
                   B_outer: int = 150, B_inner: int = 150) -> list:
    """`frontier_winner` for every system on the observed frontier, cheapest first."""
    names = [n for n in board.names if n in costs]
    idx = np.array([board.index(n) for n in names])
    acc = board.Y[idx].astype(float).mean(axis=1)
    cost = np.array([float(np.mean(costs[n])) if np.ndim(costs[n]) > 0 else float(costs[n]) for n in names])
    members = [k for k in np.argsort(cost, kind="stable") if observed_frontier(acc, cost)[k]]
    return [frontier_winner(board, costs, names[k], alpha=alpha, B=B, seed=seed + k, iterate=iterate, B_outer=B_outer, B_inner=B_inner)
            for k in members]


# ---- certifying a frontier entrant against the whole cheaper pool ----------------------------------------------------
def entrant_pvalues_arrays(acc, Sigma, b, pool, T):
    """Polyhedral selective p-values (Lee et al., 2016) for H_k: theta_b <= theta_k, one per cheaper-or-equal system k
    in `pool`, each conditional on the WHOLE frontier event {acc_b >= acc_k for every k in pool}, under the normal
    approximation acc ~ N(theta, Sigma) with a lattice correction of 1/T. The intersection-union p-value is their
    maximum: it is super-uniform under the composite null 'some system in the pool truly dominates b', because under
    H_k the k-th p-value alone is super-uniform. Returns (p_iu, {k: p_k})."""
    from . import selective
    K = len(acc)
    rows, rhs = [], []
    for k in pool:
        r = np.zeros(K); r[k] = 1.0; r[b] = -1.0
        rows.append(r); rhs.append(0.0)                           # acc_k - acc_b <= 0
    A, rhs = np.array(rows), np.array(rhs)
    out = {}
    for k in pool:
        eta = np.zeros(K); eta[b] = 1.0; eta[k] = -1.0
        pk = selective.polyhedral_pvalue(np.asarray(acc, float), Sigma, eta, A, rhs, lattice=1.0 / T)
        out[int(k)] = 1.0 if not np.isfinite(pk) else float(pk)   # no variance in the contrast: nothing can be shown
    p_iu = max(out.values()) if out else float("nan")
    return float(p_iu), out


def entrant_pvalues(board, costs, system, earlier=None):
    """Board-level wrapper of entrant_pvalues_arrays: the pool is every system in `earlier` (names; default all other
    systems with a cost) whose mean cost per task is at most the entrant's. Returns a dict with p_iu, the per-rival
    p-values by name, and the pool."""
    names = [n for n in board.names if n in costs]
    idx = np.array([board.index(n) for n in names])
    Y = board.Y[idx].astype(float)
    K, T = Y.shape
    cost = np.array([float(np.mean(costs[n])) if np.ndim(costs[n]) > 0 else float(costs[n]) for n in names])
    b = names.index(system) if isinstance(system, str) else int(system)
    cand = range(K) if earlier is None else [names.index(n) for n in earlier if n in names]
    pool = [k for k in cand if k != b and cost[k] <= cost[b]]
    acc = Y.mean(axis=1)
    Sigma = np.cov(Y) / T if K > 1 else np.zeros((1, 1))
    if not pool:
        return {"system": names[b], "p_iu": float("nan"), "pairs": {}, "pool": []}
    if acc[b] < acc[pool].max() - 1e-12:
        raise ValueError(f"{names[b]} is not on the observed frontier of its pool")
    p_iu, pairs = entrant_pvalues_arrays(acc, Sigma, b, pool, T)
    return {"system": names[b], "p_iu": p_iu, "pairs": {names[k]: v for k, v in pairs.items()}, "pool": [names[k] for k in pool]}


# ---- an exact finite-sample frontier set -----------------------------------------------------------------------------
def exact_frontier_set(board, costs, alpha=0.05, draws=50000, seed=0):
    """Keep system i unless some cheaper-or-equal system j beats it in a one-sided exact paired test at level
    alpha / (number of i's rivals): the sign test on discordant tasks for pass/fail outcomes, a sign-flip test for
    rates. A truly undominated system has no rival that is truly better, so each rival's test rejects with
    probability at most alpha/m under the task-wise null and the system is kept with probability at least 1 - alpha,
    for any number of tasks (Bonferroni over its rivals). Costs are fixed at their means. Conservative but exact."""
    from scipy import stats
    from .chain import signflip_pvalues
    names = [n for n in board.names if n in costs]
    idx = np.array([board.index(n) for n in names])
    Y = board.Y[idx].astype(float)
    K, T = Y.shape
    cost = np.array([float(np.mean(costs[n])) if np.ndim(costs[n]) > 0 else float(costs[n]) for n in names])
    binary = bool(np.all((Y == 0) | (Y == 1)))
    rng = np.random.default_rng(seed)
    kept, minp, threshold = [], {}, {}
    for i in range(K):
        rivals = [j for j in range(K) if j != i and cost[j] <= cost[i]]
        if not rivals:
            kept.append(names[i]); minp[names[i]] = 1.0; threshold[names[i]] = float(alpha); continue
        pm = 1.0
        for j in rivals:
            if binary:
                n_j = int(((Y[j] == 1) & (Y[i] == 0)).sum()); n_i = int(((Y[i] == 1) & (Y[j] == 0)).sum())
                p = 1.0 if n_j + n_i == 0 else float(stats.binom.sf(n_j - 1, n_j + n_i, 0.5))   # P(X >= n_j)
            else:
                p = signflip_pvalues(Y[j] - Y[i], rng, draws=draws)[0]
            pm = min(pm, p)
        thr = alpha / len(rivals)
        minp[names[i]] = float(pm); threshold[names[i]] = float(thr)
        if pm > thr:
            kept.append(names[i])
    n_pairs = sum(1 for i in range(K) for j in range(K) if j != i and cost[j] <= cost[i])
    kept_joint = [n for n in names if minp[n] > alpha / max(1, n_pairs)]       # Bonferroni over every rival pair: joint
    return {"alpha": alpha, "frontier_set_exact": kept, "frontier_set_exact_joint": kept_joint, "min_rival_p": minp,
            "threshold": threshold, "n_rival_pairs": n_pairs}


# ---- a simultaneous confidence band for the frontier curve ------------------------------------------------------------
def frontier_band(board, costs, alpha: float = 0.05, B: int = 4000, seed: int = 0) -> dict:
    """Simultaneous confidence band for the frontier curve A(c) = the best true accuracy among the systems that cost at
    most c. With c_band the (1 - alpha) bootstrap quantile of max_j |acc*_j - acc_j| / se_j (a max-t band for all K
    accuracies at once), L(c) = max over systems with cost <= c of (acc_j - c_band se_j) and U(c) likewise with a plus.
    On the event that every |acc_j - mu_j| <= c_band se_j, which has probability about 1 - alpha, L(c) <= A(c) <= U(c)
    for every c at once (fixed costs). The band is evaluated at every system's cost; between those costs the curves
    are flat (the frontier is a staircase)."""
    names = [n for n in board.names if n in costs]
    idx = np.array([board.index(n) for n in names])
    Y = board.Y[idx].astype(float)
    K, T = Y.shape
    cost = np.array([float(np.mean(costs[n])) if np.ndim(costs[n]) > 0 else float(costs[n]) for n in names])
    acc = Y.mean(axis=1)
    se = np.sqrt(np.maximum(Y.var(axis=1, ddof=1), 0.0) / T)
    se_safe = np.where(se > 0, se, 1e-12)
    rng = np.random.default_rng(seed)
    S = _bootstrap_means(Y, B, rng)
    dev = (np.abs(S - acc[:, None]) / se_safe[:, None])[se > 0]
    c_band = float(np.quantile(dev.max(axis=0), 1 - alpha)) if dev.size else 0.0
    lo_j, hi_j = acc - c_band * se, acc + c_band * se
    order = np.argsort(cost, kind="stable")
    grid, lower, upper, observed = [], [], [], []
    run_lo = run_hi = run_obs = -np.inf
    for k in order:
        run_lo, run_hi, run_obs = max(run_lo, lo_j[k]), max(run_hi, hi_j[k]), max(run_obs, acc[k])
        grid.append(float(cost[k])); lower.append(float(run_lo)); upper.append(float(min(run_hi, 1.0))); observed.append(float(run_obs))
    width = np.array(upper) - np.array(lower)
    return {"alpha": alpha, "B": B, "c_band": c_band, "grid": grid, "lower": lower, "upper": upper, "observed": observed,
            "width_points_median": float(100 * np.median(width)), "width_points_at_top": float(100 * width[-1]),
            "per_system": {names[k]: {"cost": float(cost[k]), "accuracy": float(acc[k]), "se": float(se[k]),
                                      "lower": float(lo_j[k]), "upper": float(hi_j[k])} for k in range(K)}}


# ---- an exact randomization certificate against the whole cheaper pool -----------------------------------------------
def entrant_pvalues_exact_arrays(Y, b, pool, draws=20000, seed=0):
    """Exact selective p-values for H_k: k is at least as good as b, one per cheaper-or-equal k in `pool`, conditional
    on the WHOLE frontier event, by randomization. Under the sharp task-wise null for the pair (b, k) the two systems'
    outcomes on each task are exchangeable, so swapping them on a random subset of tasks leaves the law unchanged;
    conditioning on the frontier event keeps only the swaps under which b still beats every system in the pool. The
    p-value is the share of kept swaps on which b's lead over k is at least the observed one (with the observed
    configuration counted once). The intersection-union maximum over k is valid under the composite null that some
    system in the pool truly dominates b, for any number of tasks. Returns (p_iu, {k: p_k}, {k: kept swaps})."""
    Y = np.asarray(Y, float)
    T = Y.shape[1]
    ybar = Y.mean(axis=1)
    rng = np.random.default_rng(seed)
    out, kept_n = {}, {}
    for k in pool:
        d = Y[b] - Y[k]
        others = [j for j in pool if j != k]
        thr = max(ybar[j] for j in others) if others else -np.inf
        neg = rng.random((draws, T)) < 0.5                                   # True: outcomes of b and k swapped on that task
        shift = (neg * d[None, :]).sum(axis=1) / T
        yb_s, yk_s = ybar[b] - shift, ybar[k] + shift
        keep = (yb_s >= thr - 1e-12) & (yb_s >= yk_s - 1e-12)                # the frontier event still holds under the swap
        stat_obs = ybar[b] - ybar[k]
        stat_s = yb_s - yk_s
        n_keep = int(keep.sum())
        p = 1.0 if n_keep == 0 else float((1 + int((stat_s[keep] >= stat_obs - 1e-12).sum())) / (1 + n_keep))
        out[int(k)] = min(1.0, p)
        kept_n[int(k)] = n_keep
    p_iu = max(out.values()) if out else float("nan")
    return float(p_iu), out, kept_n


def entrant_pvalues_exact(board, costs, system, earlier=None, draws=20000, seed=0):
    """Board-level wrapper of entrant_pvalues_exact_arrays (same pool rule as entrant_pvalues)."""
    names = [n for n in board.names if n in costs]
    idx = np.array([board.index(n) for n in names])
    Y = board.Y[idx].astype(float)
    K, T = Y.shape
    cost = np.array([float(np.mean(costs[n])) if np.ndim(costs[n]) > 0 else float(costs[n]) for n in names])
    b = names.index(system) if isinstance(system, str) else int(system)
    cand = range(K) if earlier is None else [names.index(n) for n in earlier if n in names]
    pool = [k for k in cand if k != b and cost[k] <= cost[b]]
    acc = Y.mean(axis=1)
    if not pool:
        return {"system": names[b], "p_iu": float("nan"), "pairs": {}, "pool": []}
    if acc[b] < acc[pool].max() - 1e-12:
        raise ValueError(f"{names[b]} is not on the observed frontier of its pool")
    p_iu, pairs, kept = entrant_pvalues_exact_arrays(Y, b, pool, draws=draws, seed=seed)
    return {"system": names[b], "p_iu": p_iu, "pairs": {names[k]: v for k, v in pairs.items()},
            "kept_swaps": {names[k]: v for k, v in kept.items()}, "pool": [names[k] for k in pool]}
