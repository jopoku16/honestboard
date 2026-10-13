"""Conditional (selective) inference for the winner of a leaderboard.

Model: the vector of observed scores p ~ N(theta, Sigma) with Sigma known (estimated from the
per-task data). We condition on the event that system `w` has the highest score among the
considered systems. By the polyhedral lemma (Lee, Sun, Sun and Taylor, 2016) the winner's score is
then a truncated normal on [V_minus, V_plus] with mean theta_w, which gives a selection-adjusted
median-unbiased estimate and confidence interval for theta_w.

This is the plain (non-randomised) conditional method. It is valid but its intervals can be long
when the margin of victory is small. Randomised variants (for example Bakshi, Gao, Gao and
Panigrahi, 2026) give shorter intervals and can be swapped in.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats


def _log_ndtr_diff(a, b):
    """log(Phi(b) - Phi(a)) for a < b, stable in both tails."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    flip = a > 0
    lo = np.where(flip, -b, a)
    hi = np.where(flip, -a, b)
    lh = special.log_ndtr(hi)
    ll = special.log_ndtr(lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        return lh + np.log1p(-np.exp(ll - lh))


def truncation_limits(p, sigma, w, candidates=None):
    """Return (V_minus, V_plus, ok) for the event {w has the maximum score among candidates}.

    ok is False if the observed data do not satisfy the event (w is not the maximum).
    """
    p = np.asarray(p, dtype=float)
    K = len(p)
    cand = np.arange(K) if candidates is None else np.asarray(candidates)
    if sigma[w, w] <= 0:
        return -np.inf, np.inf, False
    c = sigma[:, w] / sigma[w, w]
    z = p - c * p[w]
    v_minus, v_plus = -np.inf, np.inf
    for j in cand:
        if j == w:
            continue
        slope = 1.0 - c[j]
        if abs(slope) < 1e-12:
            if z[j] > 1e-12:
                return v_minus, v_plus, False
            continue
        bound = z[j] / slope
        if slope > 0:
            v_minus = max(v_minus, bound)
        else:
            v_plus = min(v_plus, bound)
    ok = (p[w] >= v_minus - 1e-12) and (p[w] <= v_plus + 1e-12)
    return v_minus, v_plus, ok


def _tn_cdf(x, theta, sd, lo, hi):
    u = (x - theta) / sd
    a = (lo - theta) / sd
    b = (hi - theta) / sd
    with np.errstate(divide="ignore", invalid="ignore"):
        num = _log_ndtr_diff(a, np.maximum(u, a))
        den = _log_ndtr_diff(a, b)
        val = np.exp(num - den)
    return float(np.clip(np.nan_to_num(val, nan=0.0), 0.0, 1.0))


def _solve_theta(x, sd, lo, hi, q):
    """theta such that the truncated-normal CDF at x equals q (CDF decreases in theta)."""
    f = lambda t: _tn_cdf(x, t, sd, lo, hi) - q
    left, right = x - 40 * sd, x + 40 * sd
    fl, fr = f(left), f(right)
    if fl < 0:
        return -np.inf
    if fr > 0:
        return np.inf
    return float(optimize.brentq(f, left, right, xtol=1e-10, maxiter=200))


def polyhedral_pvalue(p, sigma, eta, A, b, null_value=0.0, lattice=0.0):
    """One-sided selective p-value for H0: eta'theta <= null_value given the polyhedron {A p <= b}, under
    p ~ N(theta, sigma) with sigma known (Lee, Sun, Sun and Taylor, 2016). Returns nan if the observed point
    is not inside the polyhedron or the contrast has no variance.

    `lattice` is a continuity correction for scores that live on a grid (pass counts / T): each constraint is
    relaxed by half a grid step, so a tie with the closest competitor does not put the statistic exactly on
    the truncation boundary, and the tail is taken from half a step below the observed value."""
    p = np.asarray(p, dtype=float)
    eta = np.asarray(eta, dtype=float)
    s2 = float(eta @ sigma @ eta)
    if not np.isfinite(s2) or s2 <= 0:
        return float("nan")
    c = sigma @ eta / s2
    x = float(eta @ p)
    z = p - c * x
    Az, Ac = A @ z, A @ c
    lo, hi = -np.inf, np.inf
    tol = 1e-12
    h = 0.5 * lattice
    for az, ac, bi in zip(Az, Ac, b):
        if ac > tol:
            hi = min(hi, (bi + h - az) / ac)
        elif ac < -tol:
            lo = max(lo, (bi + h - az) / ac)
        elif az > bi + h + 1e-9:
            return float("nan")
    if not (lo - 1e-9 <= x <= hi + 1e-9):
        return float("nan")
    sd = np.sqrt(s2)
    a_, b_, u = (lo - null_value) / sd, (hi - null_value) / sd, (x - h - null_value) / sd
    u = min(max(u, a_), b_)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.exp(_log_ndtr_diff(u, b_) - _log_ndtr_diff(a_, b_))
    return float(np.clip(np.nan_to_num(val, nan=1.0), 0.0, 1.0))


def simultaneous_quantile(sigma, tail, idx=None, draws=20000, seed=0):
    """c such that P(max_j |Z_j| > c) = tail for Z ~ N(0, R), R the correlation matrix of sigma restricted to idx.

    Monte Carlo from an eigen decomposition (negative eigenvalues clipped); systems with no variance are dropped. Falls
    back to the Bonferroni value when the decomposition fails."""
    S = np.asarray(sigma, dtype=float)
    if idx is not None:
        idx = np.asarray(idx)
        S = S[np.ix_(idx, idx)]
    sd = np.sqrt(np.clip(np.diag(S), 0.0, None))
    ok = sd > 0
    m = int(ok.sum())
    if m == 0:
        return 0.0
    if m == 1:
        return float(stats.norm.ppf(1 - tail / 2))
    R = S[np.ix_(ok, ok)] / np.outer(sd[ok], sd[ok])
    try:
        vals, vecs = np.linalg.eigh((R + R.T) / 2)
        L = vecs * np.sqrt(np.clip(vals, 0.0, None))
        Z = np.random.default_rng(seed).standard_normal((draws, m)) @ L.T
        return float(np.quantile(np.abs(Z).max(axis=1), 1 - tail))
    except np.linalg.LinAlgError:
        return float(stats.norm.ppf(1 - tail / (2 * m)))


def winner_hybrid(p, sigma, w=None, alpha=0.05, beta=None, candidates=None, draws=20000, seed=0):
    """Hybrid estimate and interval of Andrews, Kitagawa and McCloskey (2024) for the winner's true score.

    The conditional (polyhedral) distribution of the winner's score is further conditioned on the event that the
    winner's true score lies in a level 1-beta projection interval (simultaneous over all candidates). That bounds
    the truncation set, so the interval always lies inside the projection interval and stays short when the margin of
    victory is small; the conditional level is adjusted to (alpha-beta)/(1-beta) so that unconditional coverage is at
    least 1-alpha. beta defaults to alpha/10. `estimate` is median-unbiased under the hybrid distribution.
    Undefined (nan) at an exact tie for first place, where the observed score sits on the truncation boundary."""
    p = np.asarray(p, dtype=float)
    cand = np.arange(len(p)) if candidates is None else np.asarray(candidates)
    if w is None:
        w = int(cand[np.argmax(p[cand])])
    beta = alpha / 10 if beta is None else beta
    if not (0 < beta < alpha):
        raise ValueError("need 0 < beta < alpha")
    sd = float(np.sqrt(sigma[w, w]))
    nan = float("nan")
    out = {"winner": int(w), "observed": float(p[w]), "se": sd, "beta": beta, "estimate": nan, "ci": (nan, nan),
           "projection_ci": (nan, nan), "degenerate": True}
    if not np.isfinite(sd) or sd <= 0:
        return out
    lo, hi, ok = truncation_limits(p, sigma, w, cand)
    if not ok:
        raise ValueError("system w is not the maximum among the candidates")
    c = simultaneous_quantile(sigma, beta, idx=cand, draws=draws, seed=seed)
    x = float(p[w])
    out["projection_ci"] = (x - c * sd, x + c * sd)
    a = (alpha - beta) / (1 - beta)

    def F(theta):
        l, h = max(lo, theta - c * sd), min(hi, theta + c * sd)
        return _tn_cdf(x, theta, sd, l, h) if h > l else nan

    def solve(q):
        f = lambda t: F(t) - q
        left, right = x - c * sd + 1e-9 * sd, x + c * sd - 1e-9 * sd
        fl, fr = f(left), f(right)
        if not (np.isfinite(fl) and np.isfinite(fr)) or fl < 0 or fr > 0:
            return nan
        return float(optimize.brentq(f, left, right, xtol=1e-10, maxiter=200))

    est, lower, upper = solve(0.5), solve(1 - a / 2), solve(a / 2)
    if np.isfinite(est) and np.isfinite(lower) and np.isfinite(upper):
        out.update({"estimate": est, "ci": (lower, upper), "degenerate": False})
    return out


def winner_zoom(p, sigma, w=None, alpha=0.05, nu=None, candidates=None, draws=20000, seed=0):
    """Zoom correction of Zrnic and Fithian (2025) for the winner, in its simplest (locally simultaneous) form.

    A share nu of the error budget buys level 1-nu simultaneous intervals for every candidate; the plausible winners
    are the candidates whose upper limit reaches the largest lower limit; the remaining budget alpha-nu buys a
    simultaneous interval over the plausible set only. The point estimate is the reported score (the correction is
    to the interval). nu defaults to alpha/10."""
    p = np.asarray(p, dtype=float)
    cand = np.arange(len(p)) if candidates is None else np.asarray(candidates)
    if w is None:
        w = int(cand[np.argmax(p[cand])])
    nu = alpha / 10 if nu is None else nu
    if not (0 < nu < alpha):
        raise ValueError("need 0 < nu < alpha")
    sd = np.sqrt(np.clip(np.diag(np.asarray(sigma, dtype=float)), 0.0, None))
    c1 = simultaneous_quantile(sigma, nu, idx=cand, draws=draws, seed=seed)
    lcb, ucb = p[cand] - c1 * sd[cand], p[cand] + c1 * sd[cand]
    plausible = cand[ucb >= lcb.max() - 1e-12]
    c2 = simultaneous_quantile(sigma, alpha - nu, idx=plausible, draws=draws, seed=seed + 1)
    x, s = float(p[w]), float(sd[w])
    return {"winner": int(w), "observed": x, "se": s, "estimate": x, "ci": (x - c2 * s, x + c2 * s),
            "plausible": [int(j) for j in plausible], "n_plausible": int(len(plausible)), "c_zoom": c1, "c_final": c2, "nu": nu}


def winner_adjusted(p, sigma, w=None, alpha=0.05, candidates=None):
    """Selection-adjusted estimate and interval for the winner's true score.

    Returns a dict with the naive and selective results. `estimate` is the median-unbiased
    estimator (the theta at which the selective p-value is 0.5).
    """
    p = np.asarray(p, dtype=float)
    cand = np.arange(len(p)) if candidates is None else np.asarray(candidates)
    if w is None:
        w = int(cand[np.argmax(p[cand])])
    sd = float(np.sqrt(sigma[w, w]))
    if not np.isfinite(sd) or sd <= 0:
        # no sampling variability for the winner (for example all-pass or all-fail): nothing to correct
        nan = float("nan")
        return {"winner": int(w), "observed": float(p[w]), "se": sd, "naive_ci": (nan, nan), "estimate": nan,
                "ci": (nan, nan), "truncation": (nan, nan), "lower_gap": nan}
    lo, hi, ok = truncation_limits(p, sigma, w, cand)
    if not ok:
        raise ValueError("system w is not the maximum among the candidates")
    x = float(p[w])
    # keep x strictly inside the truncation region for numerical stability
    eps = 1e-9 * sd
    x_in = min(max(x, lo + eps), hi - eps)
    est = _solve_theta(x_in, sd, lo, hi, 0.5)
    lower = _solve_theta(x_in, sd, lo, hi, 1 - alpha / 2)
    upper = _solve_theta(x_in, sd, lo, hi, alpha / 2)
    zq = stats.norm.ppf(1 - alpha / 2)
    # at an exact tie the observed score sits on the edge of the truncation region and the conditional
    # distribution carries no information about theta: report an undefined estimate and an unbounded interval
    degenerate = not (np.isfinite(est) and np.isfinite(lower) and np.isfinite(upper))
    if degenerate:
        est, lower, upper = float("nan"), -np.inf, np.inf
    return {
        "winner": int(w),
        "observed": x,
        "se": sd,
        "naive_ci": (x - zq * sd, x + zq * sd),
        "estimate": est,
        "ci": (lower, upper),
        "truncation": (lo, hi),
        "lower_gap": x - lo,
        "degenerate": degenerate,
    }
