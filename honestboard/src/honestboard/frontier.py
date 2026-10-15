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


def _gated_tie_robust(Y, C, S, SC, alpha, gate=True):
    """Gated step-down (Romano-Wolf) frontier set that stays valid when true mean costs tie.

    Two families of one-sided hypotheses, one pair for every ordered pair of systems (i, j):
      C_ij  "j is not strictly cheaper than i", rejected when (cbar_i - cbar_j) / se_c > c, with se_c the PAIRED standard error
            of the mean cost difference (costs move together across tasks, so it is much smaller than the sum of the two);
      A_ij  "j is not strictly more accurate than i", tested only once C_ij has been rejected, rejected when G_ij > c.
    c is the (1 - alpha) bootstrap quantile of the largest centred deviation over the hypotheses still standing; it is
    recomputed after every step, and the rejection sets grow until nothing changes. (With the gate the standing family
    also grows as accuracy hypotheses are exposed, so c is not monotone in the steps.) System i is excluded when some j has both C_ij and A_ij
    rejected, that is, when j is shown to be strictly cheaper and strictly more accurate than i. With probability at least
    1 - alpha - o(1) no true hypothesis is rejected, so every system with no strictly better cheaper-or-equal rival is kept.
    With gate=False every A_ij stands from the start: the plain Romano-Wolf step-down over both families, whose coverage is
    proved without any condition on how the cost and accuracy estimation errors depend on each other. gate=True shrinks the
    family further and gives slightly smaller sets, but its coverage is supported by simulation only.
    Returns (in_set, info).
    """
    K, T = Y.shape
    B = S.shape[1]
    acc, cost = Y.mean(axis=1), C.mean(axis=1)
    eye = np.eye(K, dtype=bool)

    def paired_se(M):
        Sig = np.cov(M) / T
        v = np.diag(Sig)[:, None] + np.diag(Sig)[None, :] - 2 * Sig
        return np.sqrt(np.maximum(v, 0.0))

    se_a = paired_se(Y)
    se_a[se_a == 0] = 1e-12
    se_c = paired_se(C)
    scale = np.abs(cost[:, None]) + np.abs(cost[None, :]) + 1e-300
    se_c = np.where(se_c <= 1e-9 * scale, 1e-12, se_c)
    dev_a = ((S[None, :, :] - S[:, None, :]) - (acc[None, :] - acc[:, None])[:, :, None]) / se_a[:, :, None]       # [i, j, b]: j above i
    dev_c = ((SC[:, None, :] - SC[None, :, :]) - (cost[:, None] - cost[None, :])[:, :, None]) / se_c[:, :, None]   # [i, j, b]: j cheaper than i
    stat_a = (acc[None, :] - acc[:, None]) / se_a
    stat_c = (cost[:, None] - cost[None, :]) / se_c
    rc = np.zeros((K, K), dtype=bool)
    ra = np.zeros((K, K), dtype=bool)
    steps, c = 0, 0.0
    while True:
        steps += 1
        remc = ~eye & ~rc
        rema = (rc if gate else ~eye) & ~ra
        m = np.full(B, -np.inf)
        if remc.any():
            m = np.maximum(m, np.where(remc[:, :, None], dev_c, -np.inf).max(axis=(0, 1)))
        if rema.any():
            m = np.maximum(m, np.where(rema[:, :, None], dev_a, -np.inf).max(axis=(0, 1)))
        c = float(max(0.0, np.quantile(m, 1 - alpha))) if np.isfinite(m).any() else 0.0
        new_c = rc | (remc & (stat_c > c))
        new_a = ra | (rema & (stat_a > c))                       # only hypotheses that stood when c was computed are tested
        if (new_c == rc).all() and (new_a == ra).all():
            break
        rc, ra = new_c, new_a
    in_set = ~(rc & ra).any(axis=1)
    iu = np.triu_indices(K, 1)
    return in_set, {"c": c, "steps": steps, "cost_ordered_share": float((rc[iu] | rc.T[iu]).mean()) if K > 1 else 1.0}


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


def pareto_frontier(board, costs, alpha: float = 0.05, B: int = 4000, seed: int = 0, names=None, clusters=None, tie_robust: bool = False) -> dict:
    """Confidence set for the Pareto frontier of a board.

    Args:
        board: a `Board` (systems-by-tasks outcomes in [0, 1]).
        costs: {system: cost} with one number per system, or {system: array of per-task costs} (length T).
        alpha: one minus the confidence level.
        B: bootstrap resamples over tasks.
        names: restrict to these systems (default: every system that has a cost).
        clusters: one label per task; resample task groups instead of tasks.
        tie_robust: with per-task costs, a set that stays valid when true mean costs tie. True (or "stepdown") runs a
            Romano-Wolf step-down on two one-sided families: a system is excluded only when some other system is shown to be
            strictly cheaper (a test on the PAIRED mean cost difference) and strictly more accurate, with one critical value
            shared by both kinds of test and recomputed as hypotheses are rejected. Its coverage is proved with no condition
            on ties or on how cost and accuracy errors depend on each other: the joint set contains every system that has no
            strictly better cheaper-or-equal rival with probability at least 1 - alpha - o(1). "gated" tests an accuracy
            hypothesis only once its cost hypothesis has been rejected, which gives slightly smaller sets but whose coverage
            is supported by simulation only. "intervals" is the more conservative 0.5.1 construction: a rival counts only
            when its cost interval lies at or below the system's, with the margin taken over every ordered pair at level
            alpha / 2. All three contain the observed frontier, and only the joint set carries the guarantee: the boardwide
            and per-system variants returned beside it equal it in this mode.
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
    alpha_set, c_cost, cheaper_fixed, gated = alpha, 0.0, None, None
    if tie_robust and per_task and tie_robust not in ("intervals",):
        gated = _gated_tie_robust(Y, C, S, SC, alpha, gate=(tie_robust == "gated"))
    elif tie_robust and per_task:
        sg = C.std(axis=1, ddof=1) / np.sqrt(T)
        sg = np.where(sg <= 1e-9 * np.maximum(np.abs(cost), 1e-300), 0.0, sg)
        if (sg > 0).any():
            dc = np.abs(SC[sg > 0] - cost[sg > 0, None]) / sg[sg > 0, None]
            c_cost = float(np.quantile(dc.max(axis=0), 1 - alpha / 2))
        lo_c, hi_c = cost - c_cost * sg, cost + c_cost * sg
        cheaper_fixed = (hi_c[None, :] <= lo_c[:, None]) & ~eye   # j is surely no costlier than i
        cheaper = cheaper_fixed
        alpha_set = alpha / 2

    dev = ((S[None, :, :] - S[:, None, :]) - (acc[None, :] - acc[:, None])[:, :, None]) / se[:, :, None]   # [i, j, b]
    if cheaper_fixed is not None:                                 # tie-robust: the margin covers EVERY ordered pair (a deterministic
        masked = np.where(~eye[:, :, None], dev, -np.inf)         # index set), whichever pairs end up as rivals of each other
    elif per_task:                                                # the rival set varies with the resample
        cheaper_b = (SC[None, :, :] <= SC[:, None, :]) & ~eye[:, :, None]
        masked = np.where(cheaper_b, dev, -np.inf)
    else:
        masked = np.where(cheaper[:, :, None], dev, -np.inf)
    M = masked.max(axis=1)                                        # K x B
    has_rival = np.isfinite(M).any(axis=1)
    q = np.array([np.quantile(M[i][np.isfinite(M[i])], 1 - alpha_set) if has_rival[i] else -np.inf for i in range(K)])
    c_board = float(max(0.0, q[has_rival].max())) if has_rival.any() else 0.0     # the largest per-system margin
    M_all = np.where(has_rival[:, None], M, -np.inf).max(axis=0)                 # B: the largest deviation on the board
    fin = np.isfinite(M_all)
    c = float(max(0.0, np.quantile(M_all[fin], 1 - alpha_set))) if fin.any() else 0.0   # joint margin: all truly undominated kept together

    gap = (acc[None, :] - acc[:, None]) / se                      # [i, j]: how far j sits above i, in se units
    beaten_by = np.where(cheaper, gap, -np.inf)
    worst = beaten_by.max(axis=1)
    in_set = ~(worst > c)                                         # the confidence set for the frontier (joint)
    in_set_board = ~(worst > c_board)
    q_eff = np.where(has_rival, np.maximum(np.where(np.isfinite(q), q, 0.0), 0.0), 0.0)
    in_set_ps = ~(worst > q_eff)                                  # per-system margins: valid per system, smaller sets
    observed = observed_frontier(acc, cost)
    if cheaper_fixed is not None:                                 # the observed frontier is always kept
        in_set, in_set_board, in_set_ps = in_set | observed, in_set_board | observed, in_set_ps | observed
    if gated is not None:                                         # gated step-down: one joint set, shared by the three variants
        in_set = in_set_board = in_set_ps = gated[0] | observed
        c = c_board = gated[1]["c"]
        c_cost = gated[1]["c"]
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
            "undominated_frequency": float(freq[k]), "frontier_stability": float(freq[k]), "in_best_set": bool(best_set[k]),
            "strongest_cheaper_rival": names[rival] if rival is not None else None,
            "rival_lead": float(acc[rival] - acc[k]) if rival is not None else None,
            "rival_lead_se": float(worst[k]) if rival is not None else None,
        })
    return {
        "alpha": alpha, "systems": K, "tasks": T, "B": B, "per_task_costs": per_task, "clustered": clusters is not None, "tie_robust": bool(cheaper_fixed is not None or gated is not None),
        "tie_robust_method": (("gated" if tie_robust == "gated" else "stepdown") if gated is not None else ("intervals" if cheaper_fixed is not None else None)),
        "cost_ordered_share": (gated[1]["cost_ordered_share"] if gated is not None else None), "cost_interval_c": c_cost,
        "margin_se": c, "margin_se_boardwide": c_board, "margin_se_best": c_all,
        "observed_frontier": [names[k] for k in range(K) if observed[k]],
        "frontier_set": [names[k] for k in range(K) if in_set[k]],
        "frontier_set_boardwide": [names[k] for k in range(K) if in_set_board[k]],
        "frontier_set_persystem": [names[k] for k in range(K) if in_set_ps[k]],
        "margins_persystem": {names[k]: float(q_eff[k]) for k in range(K)},
        "best_set": [names[k] for k in range(K) if best_set[k]],
        "top": {"system": names[top], "accuracy": float(acc[top]), "cost": float(cost[top])},
        "cheapest_plausible_best": {"system": names[cheapest_best], "accuracy": float(acc[cheapest_best]), "cost": float(cost[cheapest_best])},
        "cheapest_unresolved_competitor": {"system": names[cheapest_best], "accuracy": float(acc[cheapest_best]), "cost": float(cost[cheapest_best])},
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
def frontier_band(board, costs, alpha: float = 0.05, B: int = 4000, seed: int = 0, small: int = 10) -> dict:
    """Simultaneous confidence band for the frontier curve A(x) = the best true accuracy among the systems whose true
    mean cost is at most x (defined for x at or above the cheapest true cost), valid whether costs are fixed numbers or
    measured per task.

    With c the quantile of the largest studentised deviation over the regular accuracies and ALL mean costs (tasks
    resampled jointly), each system has an accuracy interval and a cost interval [g_j -/+ c sg_j]. On the event that
    every true accuracy and true mean cost lies in its interval (probability at least 1 - alpha, asymptotically),
        L(x) = max of (lower accuracy end) over the systems whose cost interval lies at or below x     <= A(x),
        U(x) = max of (upper accuracy end) over the systems whose cost interval reaches x or below      >= A(x),
    for every x at once: a system enters the lower edge only when its cost is surely within budget and the upper edge
    when it possibly is, so the cost order need not be right. With fixed costs the cost intervals are points.

    A bootstrap-t accuracy interval undercovers, and collapses to a point when a system passes or fails every task,
    for systems with few passes or few failures. For pass/fail outcomes a system with fewer than `small` passes or
    fewer than `small` failures therefore gets a Clopper-Pearson interval at level alpha / (2K) (alpha / K when no
    max-t part remains), and the max-t quantile for everything else is taken at alpha / 2; the union bound gives the
    total. The lower edge is 0 where no system is surely within budget."""
    from scipy import stats as _st
    names = [n for n in board.names if n in costs]
    idx = np.array([board.index(n) for n in names])
    Y = board.Y[idx].astype(float)
    K, T = Y.shape
    per_task = any(np.ndim(costs[n]) > 0 for n in names)
    if per_task:
        C = np.array([np.asarray(costs[n], float) if np.ndim(costs[n]) > 0 else np.full(T, float(costs[n]) / T) for n in names])
        g = C.mean(axis=1)
        sg = C.std(axis=1, ddof=1) / np.sqrt(T)
        sg = np.where(sg <= 1e-9 * np.maximum(np.abs(g), 1e-300), 0.0, sg)    # a constant cost has no uncertainty, not floating-point noise
    else:
        C = None
        g = np.array([float(costs[n]) for n in names])
        sg = np.zeros(K)
    acc = Y.mean(axis=1)
    se = np.sqrt(np.maximum(Y.var(axis=1, ddof=1), 0.0) / T)
    binary = bool(np.all((Y == 0) | (Y == 1)))
    passes = Y.sum(axis=1)
    small_mask = (np.minimum(passes, T - passes) < small) if binary else np.zeros(K, bool)
    regular = ~small_mask & (se > 0)
    has_maxt = bool(regular.any() or (sg > 0).any())
    if small_mask.any() and has_maxt:
        alpha_t, delta = alpha / 2, alpha / (2 * K)
    elif small_mask.any():
        alpha_t, delta = alpha, alpha / K
    else:
        alpha_t, delta = alpha, 0.0
    rng = np.random.default_rng(seed)
    if per_task:
        S, SC = _bootstrap_pairs(Y, C, B, rng)
    else:
        S, SC = _bootstrap_means(Y, B, rng), None
    dev = []
    if regular.any():
        dev.append(np.abs(S[regular] - acc[regular, None]) / se[regular, None])
    if per_task and (sg > 0).any():
        dev.append(np.abs(SC[sg > 0] - g[sg > 0, None]) / sg[sg > 0, None])
    c_band = float(np.quantile(np.vstack(dev).max(axis=0), 1 - alpha_t)) if dev else 0.0
    a_lo, a_hi = acc.copy(), acc.copy()
    a_lo[regular] = acc[regular] - c_band * se[regular]
    a_hi[regular] = np.minimum(acc[regular] + c_band * se[regular], 1.0)
    for j in np.where(small_mask)[0]:                                          # exact interval where the bootstrap-t is unreliable
        x = int(round(passes[j]))
        a_lo[j] = 0.0 if x == 0 else float(_st.beta.ppf(delta / 2, x, T - x + 1))
        a_hi[j] = 1.0 if x == T else float(_st.beta.ppf(1 - delta / 2, x + 1, T - x))
    g_lo, g_hi = g - c_band * sg, g + c_band * sg
    grid = np.unique(np.concatenate([g_lo, g_hi, g]))
    lower, upper, observed = [], [], []
    for x in grid:
        sure = a_lo[g_hi <= x + 1e-12]
        poss = a_hi[g_lo <= x + 1e-12]
        obs = acc[g <= x + 1e-12]
        lower.append(float(max(sure.max(), 0.0)) if sure.size else 0.0)
        upper.append(float(poss.max()) if poss.size else float("nan"))
        observed.append(float(obs.max()) if obs.size else float("nan"))
    lower, upper, observed = np.array(lower), np.array(upper), np.array(observed)
    ok = np.isfinite(upper)
    width = (upper - lower)[ok]
    return {"alpha": alpha, "B": B, "c_band": c_band, "per_task_costs": bool(per_task), "n_small": int(small_mask.sum()),
            "grid": [float(x) for x in grid[ok]], "lower": [float(v) for v in lower[ok]], "upper": [float(v) for v in upper[ok]],
            "observed": [float(v) if np.isfinite(v) else None for v in observed[ok]],
            "width_points_median": float(100 * np.median(width)), "width_points_at_top": float(100 * width[-1]),
            "per_system": {names[k]: {"cost": float(g[k]), "cost_se": float(sg[k]), "accuracy": float(acc[k]), "se": float(se[k]),
                                      "small": bool(small_mask[k]), "lower": float(a_lo[k]), "upper": float(a_hi[k])} for k in range(K)}}


def band_at(band, x):
    """(lower, upper) of a band returned by frontier_band at cost x: the step functions' values at the largest grid
    point at or below x (None when x lies below the first grid point, where no system can be within budget)."""
    grid = np.asarray(band["grid"])
    i = int(np.searchsorted(grid, x + 1e-12, side="right")) - 1
    if i < 0:
        return None
    return band["lower"][i], band["upper"][i]


# ---- an exact randomization certificate against the whole cheaper pool -----------------------------------------------
def entrant_pvalues_exact_arrays(Y, b, pool, draws=20000, seed=0):
    """Exact selective p-values for H_k: k is at least as good as b, one per cheaper-or-equal k in `pool`, conditional
    on the whole frontier event E = {ybar_b >= ybar_j for every j in pool}.

    Pass/fail outcomes (closed form, no Monte Carlo): write D for the number of tasks on which b and k differ, N for the
    number b wins, n11 for the number both pass, and T for the number of tasks. Swapping b's and k's outcomes on a
    random subset of the discordant tasks turns N into N' ~ Bin(D, 1/2) under the sharp null, and the swaps that keep
    E are those with N' >= n0 = max(ceil(D / 2), ceil(T * max_{j in pool, j != k} ybar_j - n11)). Hence
        p_k = P(X >= N) / P(X >= n0),  X ~ Bin(D, 1/2):
    the companion paper's selective sign test with a higher truncation point. Under H_k (on every discordant task b
    wins with probability at most 1/2 given the pairs and the other systems' outcomes) the p-value is super-uniform
    given E, for any number of tasks. The intersection-union maximum over k is then valid under the composite null
    that H_k holds for SOME k in the pool. Graded outcomes in [0, 1] (no closed form) fall back to `draws` random
    swaps with the +1 correction, valid under the sharp null only and floored at 1 / (1 + kept swaps).
    Returns (p_iu, {k: p_k}, {k: kept mass P(X >= n0) (closed form) or kept swaps (Monte Carlo)})."""
    from scipy import stats as _st
    Y = np.asarray(Y, float)
    T = Y.shape[1]
    ybar = Y.mean(axis=1)
    binary = bool(np.all((Y == 0) | (Y == 1)))
    out, kept_n = {}, {}
    if binary:
        for k in pool:
            yb, yk = Y[b], Y[k]
            N = int(((yb == 1) & (yk == 0)).sum())
            D = N + int(((yk == 1) & (yb == 0)).sum())
            n11 = int(((yb == 1) & (yk == 1)).sum())
            others = [j for j in pool if j != k]
            need = T * max(ybar[j] for j in others) - n11 if others else -np.inf
            n0 = max(int(np.ceil(D / 2 - 1e-12)), int(np.ceil(need - 1e-9)) if np.isfinite(need) else 0, 0)
            if D == 0 or N < n0:                                              # nothing to swap, or the observed event fails
                out[int(k)], kept_n[int(k)] = 1.0, 1.0
                continue
            denom = float(_st.binom.sf(n0 - 1, D, 0.5))                       # P(X >= n0)
            out[int(k)] = float(min(1.0, _st.binom.sf(N - 1, D, 0.5) / denom))
            kept_n[int(k)] = denom
    else:
        rng = np.random.default_rng(seed)
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
            "kept_mass": {names[k]: v for k, v in kept.items()}, "pool": [names[k] for k in pool]}
