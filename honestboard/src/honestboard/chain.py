"""Sequential certification of record claims on a leaderboard.

A leaderboard tests a submission only when it scores higher than the current record. Three things make a naive
test wrong: the record event itself (selection on the sign of the difference), the holder being a running
maximum, and the unknown, data-dependent number of records that will be tested. This module provides:

* `selective_sign_pvalue`: the exact sign-test p-value conditional on the record event,
  P(X >= n_b | X > n/2) with X ~ Bin(n, 1/2), n = n_a + n_b discordant tasks. Under equal ability it is
  super-uniform given that the step is a record (for a fixed holder); a holder that is a running maximum makes
  it conservative.
* `selective_signflip_pvalue`: the same idea for per-task rates, by Monte Carlo sign flips.
* `alpha_spending_level`: alpha_i = alpha * 6 / (pi^2 i^2) for the i-th record. The family-wise error over the
  whole chain is at most alpha for ANY number of records and ANY dependence, by the union bound, provided each
  step's p-value is valid given that the step occurred.
* `lord_level`: LORD++ (Ramdas, Yang, Wainwright and Jordan, 2017), an online false-discovery-rate rule in which
  each certified record earns back testing budget, so late records remain testable. Its guarantee is proved
  for independent p-values; with shared tasks the p-values are dependent, so it is checked by simulation.
* `certify_chain`: run a rule along a board's submission history, either against the leaderboard holder or
  against the last certified record (a baseline that moves only when a record is certified).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from . import selective


# ---- per-step p-values ----------------------------------------------
def selective_sign_pvalue(n_b: int, n_a: int) -> float:
    """P(X >= n_b | X > n/2), X ~ Bin(n, 1/2). Equals 1 unless n_b > n_a."""
    n = int(n_a + n_b)
    if n == 0 or n_b <= n_a:
        return 1.0
    tail = stats.binom.sf(n_b - 1, n, 0.5)          # P(X >= n_b)
    cond = stats.binom.sf(n // 2, n, 0.5)           # P(X > n/2)
    return float(min(1.0, tail / cond))


def selective_signflip_pvalue(d, rng=None, draws=20000) -> float:
    """Rates version: P(S >= s_obs | S > 0) under random sign flips of the per-task differences d."""
    d = np.asarray(d, dtype=float)
    obs = d.sum()
    if obs <= 1e-12:
        return 1.0
    rng = np.random.default_rng(0) if rng is None else rng
    null = rng.choice([-1.0, 1.0], size=(draws, d.size)) @ d
    pos = int((null > 1e-12).sum())
    if pos == 0:
        return 1.0
    return float(min(1.0, (1 + int((null >= obs - 1e-12).sum())) / (1 + pos)))


def polyhedral_step_pvalue(p_hat, sigma, b, a, earlier, lattice=0.0) -> float:
    """Selective p-value under p_hat ~ N(theta, sigma) for H0: theta_b <= theta_a, conditional on the record
    event {p_b > p_a} and on a having been the holder {p_a >= p_j for every earlier j != a}. `lattice` (1/T for
    pass counts) applies a continuity correction; without it, ties with the closest competitor put the statistic
    on the truncation boundary and the p-value collapses."""
    K = len(p_hat)
    rows, rhs = [], []
    r = np.zeros(K); r[a] = 1.0; r[b] = -1.0
    rows.append(r); rhs.append(0.0)                 # p_a - p_b <= 0
    for j in earlier:
        if j == a:
            continue
        r = np.zeros(K); r[j] = 1.0; r[a] = -1.0
        rows.append(r); rhs.append(0.0)             # p_j - p_a <= 0
    eta = np.zeros(K); eta[b] = 1.0; eta[a] = -1.0
    return selective.polyhedral_pvalue(np.asarray(p_hat, float), sigma, eta, np.array(rows), np.array(rhs), lattice=lattice)


# ---- test levels ----------------------------------------------------
def alpha_spending_level(i: int, alpha: float) -> float:
    """Level for the i-th record (i = 1, 2, ...); the levels sum to alpha."""
    return alpha * 6.0 / (np.pi ** 2 * i ** 2)


def lord_gamma(j: int, kind: str = "quadratic") -> float:
    """A spending sequence with sum at most 1. 'quadratic': 6 / (pi^2 j^2), which sums to 1 and suits chains of
    tens of records. 'log': the sequence published with LORD++, 0.0722 log(max(j, 2)) / (j e^{sqrt(log j)}), which
    sums to 1 only in the limit and is built for very long streams."""
    if j < 1:
        return 0.0
    if kind == "quadratic":
        return float(6.0 / (np.pi ** 2 * j ** 2))
    if j == 1:
        return float(0.0722 * np.log(2))
    return float(0.0722 * np.log(max(j, 2)) / (j * np.exp(np.sqrt(np.log(j)))))


def lord_level(t: int, alpha: float, w0: float, rejection_times, kind: str = "quadratic") -> float:
    """LORD++ level for the t-th test given the times of earlier rejections (1-indexed test counts)."""
    lvl = lord_gamma(t, kind) * w0
    if rejection_times:
        lvl += (alpha - w0) * lord_gamma(t - rejection_times[0], kind)
        for tau in rejection_times[1:]:
            lvl += alpha * lord_gamma(t - tau, kind)
    return float(lvl)


# ---- the procedure --------------------------------------------------
def _submission_order(board, tie_order="best_first"):
    if board.dates is None:
        raise ValueError("certification needs dates")
    undated = [n for n, d in zip(board.names, board.dates) if not d]
    if undated:
        raise ValueError(f"{len(undated)} systems have no date (first: {undated[0]})")
    sc = board.scores()
    sign = -1.0 if tie_order == "best_first" else 1.0
    return sorted(range(board.K), key=lambda k: (board.dates[k], sign * sc[k]))


def _step_pvalue(board, a, b, pvalue, sigma, earlier, seed):
    if pvalue == "polyhedral":
        return polyhedral_step_pvalue(board.scores(), sigma, b, a, earlier, lattice=1.0 / board.T)
    ya, yb = board.Y[a], board.Y[b]
    if board.is_binary:
        n_b = int(((yb == 1) & (ya == 0)).sum())
        n_a = int(((ya == 1) & (yb == 0)).sum())
        return selective_sign_pvalue(n_b, n_a)
    return selective_signflip_pvalue(yb.astype(float) - ya.astype(float), np.random.default_rng(seed + 7 * b + a))


def fallback_level(t: int, alpha: float, prev_level: float, prev_certified: bool) -> float:
    """Online fallback (Wiens 2003; Tian and Ramdas 2021): the t-th record is tested at its spending level plus the
    level of the previous record if that record was certified, so a run of certified records accumulates budget.
    Family-wise error over the chain is at most alpha for any dependence, by the same argument as for spending:
    on the event that step t is the first null step certified, every step in the run that feeds it is non-null, and
    the runs feeding different null steps are disjoint, so the levels available to null steps sum to at most alpha."""
    return float(alpha_spending_level(t, alpha) + (prev_level if prev_certified else 0.0))


def certify_chain(board, alpha=0.05, rule="lord", pvalue="exact", baseline="leader", tie_order="best_first", w0=None, seed=0):
    """Walk the submissions in date order and certify records.

    rule: 'unadjusted' (level alpha every step), 'holm' (retrospective, over the realised chain),
          'spending' (alpha * 6 / (pi^2 i^2) for the i-th tested record; family-wise control for any number of
          records), 'fallback' (spending plus the level of the previous record when it was certified; the same
          family-wise guarantee with more power along runs of genuine records), 'lord' (LORD++ online FDR).
    pvalue: 'exact' (sign test conditional on the record event; sign flips for rates) or 'polyhedral'
            (normal approximation, conditional on the record event and on the holder having been the maximum).
    baseline: 'leader' tests each leaderboard record against the record it replaced; 'certified' tests every
              submission that beats the last certified record against that record (the baseline moves only when
              a record is certified)."""
    if rule not in ("unadjusted", "holm", "spending", "fallback", "lord"):
        raise ValueError("rule must be 'unadjusted', 'holm', 'spending', 'fallback' or 'lord'")
    if pvalue not in ("exact", "polyhedral"):
        raise ValueError("pvalue must be 'exact' or 'polyhedral'")
    if baseline not in ("leader", "certified"):
        raise ValueError("baseline must be 'leader' or 'certified'")
    w0 = alpha / 2 if w0 is None else w0
    order = _submission_order(board, tie_order)
    sc = board.scores()
    sigma = board.sigma() if pvalue == "polyhedral" else None
    holder = order[0]
    certified_holder = order[0]
    steps, rejections, t = [], [], 0
    prev_level, prev_cert = 0.0, False
    for pos, k in enumerate(order[1:], start=1):
        ref = holder if baseline == "leader" else certified_holder
        if sc[k] > sc[ref] + 1e-12:
            t += 1
            p = _step_pvalue(board, ref, k, pvalue, sigma, order[:pos], seed)
            if rule == "unadjusted":
                level = alpha
            elif rule == "spending":
                level = alpha_spending_level(t, alpha)
            elif rule == "fallback":
                level = fallback_level(t, alpha, prev_level, prev_cert)
            elif rule == "lord":
                level = lord_level(t, alpha, w0, rejections)
            else:
                level = None
            cert = (p <= level) if level is not None else None
            prev_level, prev_cert = (level or 0.0), bool(cert)
            if cert:
                rejections.append(t)
                certified_holder = k
            steps.append({"test": t, "position": pos, "new": board.names[k], "reference": board.names[ref],
                          "margin": float(sc[k] - sc[ref]), "p": float(p), "level": level, "certified": cert})
        if sc[k] > sc[holder] + 1e-12:
            holder = k
    if rule == "holm":
        m = len(steps)
        ps = np.array([s["p"] for s in steps]) if m else np.array([])
        rej = np.zeros(m, dtype=bool)
        for rank, idx in enumerate(np.argsort(ps)):
            if ps[idx] <= alpha / (m - rank):
                rej[idx] = True
            else:
                break
        for s, r in zip(steps, rej):
            s["certified"] = bool(r)
            s["level"] = None
    return {"rule": rule, "pvalue": pvalue, "baseline": baseline, "alpha": alpha, "n_tests": len(steps),
            "n_certified": int(sum(1 for s in steps if s["certified"])),
            "certified": [s["new"] for s in steps if s["certified"]], "steps": steps,
            "final_certified_holder": board.names[certified_holder]}
