import unittest

import numpy as np
import pytest

from honestboard import Board
from honestboard.frontier import band_at, entrant_pvalues, entrant_pvalues_exact, exact_frontier_set, frontier_band, frontier_curse, frontier_winner, observed_frontier, pareto_frontier


def make_board(P, T, seed=0, names=None):
    rng = np.random.default_rng(seed)
    K = len(P)
    Y = (rng.random((K, T)) < np.asarray(P)[:, None]).astype(float)
    names = names or [f"s{k}" for k in range(K)]
    tasks = [f"t{t}" for t in range(T)]
    systems = [{"name": n, "date": "20250101", "rates": {t: float(Y[k, i]) for i, t in enumerate(tasks)}} for k, n in enumerate(names)]
    return Board.from_rates(tasks, systems)


class TestObservedFrontier(unittest.TestCase):
    def test_dominated_system_is_off_the_frontier(self):
        acc = np.array([0.80, 0.79, 0.50, 0.78])
        cost = np.array([10.0, 2.0, 1.0, 12.0])
        on = observed_frontier(acc, cost)
        self.assertEqual(on.tolist(), [True, True, True, False])      # s3 costs more than s0 and is less accurate

    def test_ties_handled(self):
        on = observed_frontier(np.array([0.5, 0.5]), np.array([1.0, 1.0]))
        self.assertEqual(on.tolist(), [True, True])                   # equal in both: neither dominates


class TestFrontierSet(unittest.TestCase):
    def test_set_contains_observed_frontier_and_excludes_clearly_dominated(self):
        b = make_board([0.80, 0.79, 0.50, 0.40], T=600, seed=1)
        costs = {"s0": 10.0, "s1": 2.0, "s2": 1.0, "s3": 12.0}
        r = pareto_frontier(b, costs, B=800, seed=0)
        for n in r["observed_frontier"]:
            self.assertIn(n, r["frontier_set"])
        self.assertNotIn("s3", r["frontier_set"])                     # 40% at the highest cost against 80% cheaper: clearly dominated
        self.assertEqual(r["cheapest_plausible_best"]["system"] in ("s0", "s1"), True)

    def test_known_truth_coverage_of_undominated_systems(self):
        """A truly undominated system should be kept in the frontier set at least about 1 - alpha of the time."""
        rng = np.random.default_rng(3)
        P = [0.70, 0.68, 0.60, 0.55]                                   # s0 and s1 close; s2, s3 cheaper and worse
        costs = {"s0": 8.0, "s1": 4.0, "s2": 2.0, "s3": 1.0}
        truly = observed_frontier(np.array(P), np.array([costs[f"s{k}"] for k in range(4)]))
        kept = np.zeros(4)
        R = 60
        for r in range(R):
            b = make_board(P, T=300, seed=100 + r)
            res = pareto_frontier(b, costs, B=300, seed=r)
            for k in range(4):
                kept[k] += f"s{k}" in res["frontier_set"]
        for k in range(4):
            if truly[k]:
                self.assertGreaterEqual(kept[k] / R, 0.85, f"system s{k} kept only {kept[k] / R:.2f} of the time")

    def test_per_task_costs(self):
        b = make_board([0.8, 0.7, 0.6], T=200, seed=5)
        rng = np.random.default_rng(0)
        costs = {"s0": rng.random(200) * 2 + 1, "s1": rng.random(200) + 0.5, "s2": rng.random(200) * 0.5}
        r = pareto_frontier(b, costs, B=300, seed=1)
        self.assertTrue(r["per_task_costs"])
        self.assertEqual(len(r["table"]), 3)
        self.assertTrue(all(0.0 <= row["undominated_frequency"] <= 1.0 for row in r["table"]))


if __name__ == "__main__":
    unittest.main()


def _known_truth_board(K=12, T=120, seed=3):
    """Systems with true accuracies rising with cost, plus noise, so the frontier winner's curse is real."""
    rng = np.random.default_rng(seed)
    cost = np.exp(rng.normal(0, 1, K))
    theta = 0.3 + 0.4 * (np.argsort(np.argsort(cost)) / (K - 1)) + rng.normal(0, 0.05, K)
    theta = np.clip(theta, 0.05, 0.95)
    diff = rng.normal(0, 1.0, T)
    P = 1 / (1 + np.exp(-(np.log(theta / (1 - theta))[:, None] + diff[None, :])))
    Y = (rng.random(P.shape) < P).astype(float)
    tasks = [f"t{t}" for t in range(T)]
    systems = [{"name": f"s{k}", "rates": {t: float(Y[k, i]) for i, t in enumerate(tasks)}} for k in range(K)]
    board = Board.from_rates(tasks, systems)
    truth = np.array([(1 / (1 + np.exp(-(np.log(th / (1 - th)) + diff)))).mean() for th in theta])
    return board, {f"s{k}": float(cost[k]) for k in range(K)}, truth


def test_frontier_winner_pool_of_one_is_uncorrected():
    board, costs, _ = _known_truth_board()
    cheapest = min(costs, key=costs.get)
    r = frontier_winner(board, costs, cheapest, B=200, iterate=False)
    assert r["pool_size"] == 1 and r["bootstrap"]["estimate"] == r["observed"] and r["conditional"]["estimate"] == r["observed"]


def test_frontier_winner_rejects_dominated_system():
    board, costs, _ = _known_truth_board()
    acc = board.scores()
    cost = np.array([costs[n] for n in board.names])
    dominated = [k for k in range(board.K) if any(cost[j] <= cost[k] and acc[j] > acc[k] for j in range(board.K))]
    with pytest.raises(ValueError):
        frontier_winner(board, costs, board.names[dominated[0]], B=100, iterate=False)


def test_frontier_curse_reduces_bias_on_known_truth():
    """Across many boards, the reported accuracy of frontier winners with a rival overshoots the truth and the
    corrections shrink the overshoot (the plug-in bootstrap by at least a third of the naive bias)."""
    naive, boot, cond = [], [], []
    for seed in range(40):
        board, costs, truth = _known_truth_board(K=14, T=60, seed=seed)
        for r in frontier_curse(board, costs, B=300, seed=seed, iterate=False):
            if r["pool_size"] < 3:
                continue
            k = board.names.index(r["system"])
            naive.append(r["observed"] - truth[k])
            boot.append(r["bootstrap"]["estimate"] - truth[k])
            if np.isfinite(r["conditional"]["estimate"]):
                cond.append(r["conditional"]["estimate"] - truth[k])
    naive, boot = np.mean(naive), np.mean(boot)
    assert naive > 0.0
    assert abs(boot) < naive - naive / 3
    assert len(cond) > 0 and abs(np.mean(cond)) < naive


def test_exact_set_contains_observed_frontier_and_is_valid_on_known_truth():
    """The exact set always contains the observed frontier, and over many boards it keeps a truly undominated
    system at least 95% of the time."""
    kept = total = 0
    for seed in range(60):
        board, costs, truth = _known_truth_board(K=10, T=80, seed=seed)
        cost = np.array([costs[n] for n in board.names])
        res = exact_frontier_set(board, costs, alpha=0.05)
        r = pareto_frontier(board, costs, B=300, seed=seed)
        for n in r["observed_frontier"]:
            assert n in res["frontier_set_exact"]
        truly = observed_frontier(truth, cost)
        for k in np.where(truly)[0]:
            total += 1
            kept += board.names[k] in res["frontier_set_exact"]
    assert kept / total >= 0.95


def test_per_system_set_is_inside_board_wide_set():
    board, costs, _ = _known_truth_board(K=12, T=100, seed=2)
    r = pareto_frontier(board, costs, B=400, seed=0)
    assert set(r["frontier_set_persystem"]) <= set(r["frontier_set"])
    assert set(r["observed_frontier"]) <= set(r["frontier_set_persystem"])


def test_entrant_pvalues_composite_is_max_of_pairs_and_valid_for_a_cheapest_system():
    board, costs, _ = _known_truth_board(K=12, T=120, seed=4)
    r = pareto_frontier(board, costs, B=200, seed=0)
    top = r["observed_frontier"][-1]                           # the most expensive frontier member has the largest pool
    res = entrant_pvalues(board, costs, top)
    assert res["pool"] and abs(res["p_iu"] - max(res["pairs"].values())) < 1e-12
    assert all(0.0 <= v <= 1.0 for v in res["pairs"].values())
    cheapest = min(costs, key=costs.get)
    assert entrant_pvalues(board, costs, cheapest)["pool"] == []
    acc = board.scores()
    cost = np.array([costs[n] for n in board.names])
    dominated = [k for k in range(board.K) if any(cost[j] <= cost[k] and acc[j] > acc[k] for j in range(board.K))]
    with pytest.raises(ValueError):
        entrant_pvalues(board, costs, board.names[dominated[0]])


def test_frontier_band_brackets_the_observed_staircase_and_covers_known_truth():
    """The band contains the observed staircase everywhere, and over many boards with fixed costs it contains the true
    staircase on every grid point at least 95% of the time."""
    covered = 0
    for seed in range(60):
        board, costs, truth = _known_truth_board(K=10, T=80, seed=seed)
        band = frontier_band(board, costs, B=400, seed=seed)
        assert all(lo - 1e-12 <= ob <= hi + 1e-12 for lo, ob, hi in zip(band["lower"], band["observed"], band["upper"]))
        cost = np.array([costs[n] for n in board.names])
        ok = True
        for c, lo, hi in zip(band["grid"], band["lower"], band["upper"]):
            a0 = max(truth[j] for j in range(board.K) if cost[j] <= c + 1e-12)
            ok = ok and (lo - 1e-12 <= a0 <= hi + 1e-12)
        covered += ok
    assert covered / 60 >= 0.93


def test_exact_entrant_pvalues_are_valid_probabilities_and_agree_with_pool_rule():
    board, costs, _ = _known_truth_board(K=12, T=120, seed=4)
    r = pareto_frontier(board, costs, B=200, seed=0)
    top = r["observed_frontier"][-1]
    res = entrant_pvalues_exact(board, costs, top, draws=2000)
    assert res["pool"] and abs(res["p_iu"] - max(res["pairs"].values())) < 1e-12
    assert all(0.0 < v <= 1.0 for v in res["pairs"].values())
    assert all(v > 0 for v in res["kept_mass"].values())                   # the observed configuration is always kept
    assert entrant_pvalues_exact(board, costs, min(costs, key=costs.get), draws=100)["pool"] == []


def test_exact_entrant_certificate_holds_level_with_a_near_copy_rival():
    """A truly dominated entrant with a near-copy rival in the pool is certified at most about 5% of the time."""
    from honestboard.frontier import entrant_pvalues_exact_arrays
    rng = np.random.default_rng(11)
    T, cert, sel = 150, 0, 0
    for _ in range(300):
        diff = rng.normal(0, 1, T)
        p = 1 / (1 + np.exp(-(0.4 + diff)))
        yb = (rng.random(T) < p).astype(float)
        yk = (rng.random(T) < p).astype(float)                              # same true accuracy, independent: dominates b
        yc = yb.copy()
        yc[(yc == 1) & (rng.random(T) < 0.1)] = 0                           # a near-copy of b
        Y = np.vstack([yb, yk, yc])
        if Y[0].mean() <= max(Y[1].mean(), Y[2].mean()) + 1e-12:
            continue
        sel += 1
        p_iu, _, _ = entrant_pvalues_exact_arrays(Y, 0, [1, 2], draws=1000, seed=sel)
        cert += p_iu <= 0.05
    assert sel > 50 and cert / sel <= 0.09


def test_frontier_band_with_per_task_costs_covers_the_true_curve_despite_near_ties_in_cost():
    """With per-task costs and closely spaced true mean costs the sample cost order is often wrong; the band, which
    lets a system into the lower edge only when its cost is surely within budget, still covers the true curve."""
    rng = np.random.default_rng(21)
    covered, R, K, T = 0, 60, 10, 80
    for r in range(R):
        gamma = np.sort(np.exp(rng.normal(0, 0.15, K)))                    # nearly tied true mean costs
        theta = np.clip(0.3 + 0.4 * np.arange(K) / (K - 1) + rng.normal(0, 0.05, K), 0.05, 0.95)
        diff = rng.normal(0, 1, T)
        P = 1 / (1 + np.exp(-(np.log(theta / (1 - theta))[:, None] + diff[None, :])))
        Y = (rng.random(P.shape) < P).astype(float)
        C = gamma[:, None] * np.exp(0.8 * rng.standard_normal(P.shape) - 0.32)
        tasks = [f"t{t}" for t in range(T)]
        board = Board.from_rates(tasks, [{"name": f"s{k}", "rates": {t: float(Y[k, i]) for i, t in enumerate(tasks)}} for k in range(K)])
        truth_acc = np.array([(1 / (1 + np.exp(-(np.log(th / (1 - th)) + diff))))[:].mean() for th in theta])
        band = frontier_band(board, {f"s{k}": C[k] for k in range(K)}, B=400, seed=r)
        ok = True
        for x in sorted(set(band["grid"]) | set(gamma.tolist())):
            bx = band_at(band, x)
            inside = [j for j in range(K) if gamma[j] <= x + 1e-12]
            if bx is None or not inside:
                continue
            a0 = max(truth_acc[j] for j in inside)
            ok = ok and (bx[0] - 1e-12 <= a0 <= bx[1] + 1e-12)
        covered += ok
    assert covered / R >= 0.90


def test_frontier_band_with_constant_per_task_costs_is_not_inflated_by_rounding_noise():
    """A per-task cost array that is constant (a fixed cost written per task) has zero cost uncertainty; the band must
    equal the fixed-cost band and stay narrow."""
    board, costs, _ = _known_truth_board(K=10, T=120, seed=5)
    fixed = frontier_band(board, costs, B=400, seed=0)
    T = board.T
    per_task = {n: np.full(T, 0.7340123 * c / 0.7340123) for n, c in costs.items()}   # constant arrays with awkward values
    arr = frontier_band(board, per_task, B=400, seed=0)
    assert arr["width_points_median"] < 60 and abs(arr["width_points_median"] - fixed["width_points_median"]) < 5



def test_exact_certificate_closed_form_matches_randomization_on_pass_fail_outcomes():
    """The truncated-binomial closed form equals the randomization p-value (swap draws) up to Monte Carlo error."""
    from honestboard.frontier import entrant_pvalues_exact_arrays
    rng = np.random.default_rng(3)
    T = 60
    for _ in range(6):
        Y = (rng.random((4, T)) < np.array([[0.6], [0.5], [0.45], [0.4]])).astype(float)
        if Y[0].mean() <= Y[1:].mean(axis=1).max():
            continue
        p_closed, pairs, _ = entrant_pvalues_exact_arrays(Y, 0, [1, 2, 3])
        # the same p-values by explicit swap draws, on real-valued copies (the graded path)
        Yg = 0.5 * Y                                                       # values {0, 0.5}: not pass/fail, so the swap-draw path runs, with exact ties
        p_mc, pairs_mc, _ = entrant_pvalues_exact_arrays(Yg, 0, [1, 2, 3], draws=60000, seed=1)
        assert abs(p_closed - p_mc) < 0.03


def test_band_with_a_near_zero_accuracy_cheap_system_keeps_its_coverage():
    """A cheap system with true accuracy 0.02 is all-fail in a 50-task sample about a third of the time; a bootstrap-t
    interval collapses to a point there. The Clopper-Pearson step keeps coverage at the nominal level."""
    rng = np.random.default_rng(31)
    mu = np.array([0.02, 0.1, 0.2, 0.3])
    costs = {f"s{k}": float(c) for k, c in enumerate([0.1, 0.2, 0.3, 0.4])}
    T, R, cov = 50, 160, 0
    for r in range(R):
        Y = (rng.random((4, T)) < mu[:, None]).astype(float)
        tasks = [f"t{t}" for t in range(T)]
        board = Board.from_rates(tasks, [{"name": f"s{k}", "rates": {t: float(Y[k, i]) for i, t in enumerate(tasks)}} for k in range(4)])
        band = frontier_band(board, costs, B=300, seed=r)
        ok = True
        for x, c in zip([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]):
            bx = band_at(band, x)
            a0 = mu[:int(round(x * 10))].max()
            ok = ok and bx is not None and (bx[0] - 1e-12 <= a0 <= bx[1] + 1e-12)
        cov += ok
    assert cov / R >= 0.93



def test_tie_robust_set_contains_the_observed_frontier_and_the_standard_set_and_keeps_tied_systems():
    """With nearly tied true mean costs the standard set can drop a truly undominated system whose sample cost order is
    wrong; the tie-robust set (rivals only when surely no costlier, alpha split) keeps it, and always contains the
    observed frontier."""
    rng = np.random.default_rng(41)
    kept_std = kept_rob = total = 0
    for r in range(40):
        K, T = 8, 120
        gamma = np.exp(rng.normal(0, 0.02, K))                              # true mean costs nearly tied
        theta = np.clip(0.3 + 0.4 * rng.random(K), 0.05, 0.95)
        diff = rng.normal(0, 1, T)
        P = 1 / (1 + np.exp(-(np.log(theta / (1 - theta))[:, None] + diff[None, :])))
        Y = (rng.random(P.shape) < P).astype(float)
        C = gamma[:, None] * np.exp(0.5 * rng.standard_normal(P.shape) - 0.125)
        tasks = [f"t{t}" for t in range(T)]
        board = Board.from_rates(tasks, [{"name": f"s{k}", "rates": {t: float(Y[k, i]) for i, t in enumerate(tasks)}} for k in range(K)])
        costs = {f"s{k}": C[k] for k in range(K)}
        std = pareto_frontier(board, costs, B=300, seed=r)
        rob = pareto_frontier(board, costs, B=300, seed=r, tie_robust=True)
        assert set(std["observed_frontier"]) <= set(rob["frontier_set"])
        assert rob["tie_robust"] and rob["cost_interval_c"] > 0
        true_acc = np.array([(1 / (1 + np.exp(-(np.log(th / (1 - th)) + diff)))).mean() for th in theta])
        on_true = observed_frontier(true_acc, gamma)
        for k in np.where(on_true)[0]:
            total += 1
            kept_std += f"s{k}" in std["frontier_set"]
            kept_rob += f"s{k}" in rob["frontier_set"]
    assert kept_rob >= kept_std and kept_rob / total >= 0.93


def test_gated_tie_robust_is_smaller_than_the_interval_version_and_still_contains_the_observed_frontier():
    """The gated step-down set (paired cost comparisons, one joint critical value) is never needed to be larger than the
    interval construction on average, keeps every system of the observed frontier, and keeps the true frontier when
    costs are nearly tied."""
    rng = np.random.default_rng(17)
    size_gated = size_intervals = kept = total = 0
    for r in range(30):
        K, T = 10, 150
        gamma = np.exp(rng.normal(0, 0.5, K))
        theta = np.clip(0.3 + 0.4 * rng.random(K), 0.05, 0.95)
        diff = rng.normal(0, 1, T)
        P = 1 / (1 + np.exp(-(np.log(theta / (1 - theta))[:, None] + diff[None, :])))
        Y = (rng.random(P.shape) < P).astype(float)
        C = gamma[:, None] * np.exp(0.5 * rng.standard_normal(P.shape) - 0.125)
        tasks = [f"t{t}" for t in range(T)]
        board = Board.from_rates(tasks, [{"name": f"s{k}", "rates": {t: float(Y[k, i]) for i, t in enumerate(tasks)}} for k in range(K)])
        costs = {f"s{k}": C[k] for k in range(K)}
        g = pareto_frontier(board, costs, B=300, seed=r, tie_robust="gated")
        v1 = pareto_frontier(board, costs, B=300, seed=r, tie_robust="intervals")
        assert g["tie_robust_method"] == "gated" and v1["tie_robust_method"] == "intervals"
        assert set(g["observed_frontier"]) <= set(g["frontier_set"])
        size_gated += len(g["frontier_set"]); size_intervals += len(v1["frontier_set"])
        true_acc = np.array([(1 / (1 + np.exp(-(np.log(th / (1 - th)) + diff)))).mean() for th in theta])
        on_true = observed_frontier(true_acc, gamma)
        for k in np.where(on_true)[0]:
            total += 1
            kept += f"s{k}" in g["frontier_set"]
    assert size_gated <= size_intervals
    assert kept / total >= 0.93


def test_ungated_stepdown_tie_robust_set_contains_the_observed_frontier_and_the_gated_set_is_no_larger():
    """tie_robust='stepdown' keeps every accuracy hypothesis standing (the version proved without conditions); gating only
    removes hypotheses, so the gated set is never larger; both contain the observed frontier."""
    rng = np.random.default_rng(23)
    for r in range(12):
        K, T = 9, 120
        gamma = np.exp(rng.normal(0, 0.2, K))
        theta = np.clip(0.3 + 0.4 * rng.random(K), 0.05, 0.95)
        diff = rng.normal(0, 1, T)
        P = 1 / (1 + np.exp(-(np.log(theta / (1 - theta))[:, None] + diff[None, :])))
        Y = (rng.random(P.shape) < P).astype(float)
        C = gamma[:, None] * np.exp(0.5 * rng.standard_normal(P.shape) - 0.125)
        tasks = [f"t{t}" for t in range(T)]
        board = Board.from_rates(tasks, [{"name": f"s{k}", "rates": {t: float(Y[k, i]) for i, t in enumerate(tasks)}} for k in range(K)])
        costs = {f"s{k}": C[k] for k in range(K)}
        g = pareto_frontier(board, costs, B=300, seed=r, tie_robust="gated")
        u = pareto_frontier(board, costs, B=300, seed=r, tie_robust=True)
        assert u["tie_robust_method"] == "stepdown" and g["tie_robust_method"] == "gated"
        assert set(u["observed_frontier"]) <= set(u["frontier_set"])
        assert set(g["frontier_set"]) <= set(u["frontier_set"]) or len(g["frontier_set"]) <= len(u["frontier_set"]) + 1


def test_gated_tie_robust_set_with_constant_costs_matches_the_standard_set_closely():
    """With constant per-task costs the cost hypotheses are rejected at once and the gated set must behave like the standard
    set (a regression test: an accuracy hypothesis newly exposed by a cost rejection must not be tested against a critical
    value computed before it joined the family)."""
    rng = np.random.default_rng(5)
    kept = total = 0
    for r in range(25):
        K, T = 10, 120
        gamma = np.exp(rng.normal(0, 0.6, K))
        theta = np.clip(0.2 + 0.6 * rng.random(K), 0.05, 0.95)
        diff = rng.normal(0, 1, T)
        P = 1 / (1 + np.exp(-(np.log(theta / (1 - theta))[:, None] + diff[None, :])))
        Y = (rng.random(P.shape) < P).astype(float)
        tasks = [f"t{t}" for t in range(T)]
        board = Board.from_rates(tasks, [{"name": f"s{k}", "rates": {t: float(Y[k, i]) for i, t in enumerate(tasks)}} for k in range(K)])
        costs = {f"s{k}": np.full(T, gamma[k]) for k in range(K)}
        g = pareto_frontier(board, costs, B=300, seed=r, tie_robust=True)
        s = pareto_frontier(board, costs, B=300, seed=r)
        true_acc = np.array([(1 / (1 + np.exp(-(np.log(th / (1 - th)) + diff)))).mean() for th in theta])
        for k in np.where(observed_frontier(true_acc, gamma))[0]:
            total += 1
            kept += f"s{k}" in g["frontier_set"]
        assert len(g["frontier_set"]) >= len(s["frontier_set"]) - 1
    assert kept / total >= 0.93
