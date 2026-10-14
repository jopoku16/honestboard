"""Who is really on the cost-accuracy frontier?

A leaderboard that reports cost beside accuracy is read through its Pareto frontier: the systems that no cheaper system
beats. Accuracy is estimated from a sample of tasks, so a frontier drawn from point estimates is an estimate too. From
per-task outcomes and a cost per system (or per task) this module gives:

1. The observed frontier: system i is on it when no system with cost at most cost_i has accuracy at least acc_i, with
   at least one of the two strict.
2. A confidence set for the frontier at level 1 - alpha: system i stays in unless some cheaper-or-equal system beats it
   by more than a simultaneous paired margin c * se_ij, where se_ij is the standard error of the paired accuracy
   difference and c is a bootstrap quantile of the largest studentised deviation of any cheaper rival from i, taken over
   all systems. This is the multiple-comparisons-with-the-best construction behind `Board.confidence_set_best`, with the
   rival set restricted to cheaper systems: a truly undominated system is kept with probability about 1 - alpha. The
   set always contains the observed frontier.
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


def pareto_frontier(board, costs, alpha: float = 0.05, B: int = 4000, seed: int = 0, names=None) -> dict:
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
    if per_task:
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
    c = float(max(0.0, q[has_rival].max())) if has_rival.any() else 0.0

    gap = (acc[None, :] - acc[:, None]) / se                      # [i, j]: how far j sits above i, in se units
    beaten_by = np.where(cheaper, gap, -np.inf)
    worst = beaten_by.max(axis=1)
    in_set = ~(worst > c)
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
        "alpha": alpha, "systems": K, "tasks": T, "B": B, "per_task_costs": per_task,
        "margin_se": c, "margin_se_best": c_all,
        "observed_frontier": [names[k] for k in range(K) if observed[k]],
        "frontier_set": [names[k] for k in range(K) if in_set[k]],
        "best_set": [names[k] for k in range(K) if best_set[k]],
        "top": {"system": names[top], "accuracy": float(acc[top]), "cost": float(cost[top])},
        "cheapest_plausible_best": {"system": names[cheapest_best], "accuracy": float(acc[cheapest_best]), "cost": float(cost[cheapest_best])},
        "table": systems,
    }
