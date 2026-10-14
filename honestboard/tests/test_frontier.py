import unittest

import numpy as np
import pytest

from honestboard import Board
from honestboard.frontier import entrant_pvalues, entrant_pvalues_exact, exact_frontier_set, frontier_band, frontier_curse, frontier_winner, observed_frontier, pareto_frontier


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
    assert all(v >= 1 for v in res["kept_swaps"].values())                 # the observed configuration is always kept
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
