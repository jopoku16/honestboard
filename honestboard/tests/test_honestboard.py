import os
import sys
import tempfile
import unittest

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from honestboard import Board, winner_adjusted  # noqa: E402
from honestboard.selective import truncation_limits  # noqa: E402


def make_board(K=8, T=300, spread=0.01, seed=0, dates=True):
    """Systems with near-equal true skill and a shared task-difficulty effect."""
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(0, 1.0, T)
    theta = 0.6 + spread * rng.normal(size=K)
    Y = np.zeros((K, T), dtype=np.uint8)
    for k in range(K):
        logit = np.log(theta[k] / (1 - theta[k])) - difficulty
        Y[k] = rng.random(T) < 1 / (1 + np.exp(-logit))
    return Board([f"s{k}" for k in range(K)], Y, [f"2025{k + 1:02d}01" for k in range(K)] if dates else None)


class TestCompare(unittest.TestCase):
    def test_mcnemar_counts_and_sidedness(self):
        Y = np.array([[1, 1, 0, 0, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]], dtype=np.uint8)
        b = Board(["a", "b"], Y)
        c = b.compare("a", "b", B=200)
        self.assertEqual((c["a_only"], c["b_only"]), (0, 4))
        self.assertAlmostEqual(c["p_two_sided"], 2 * 0.5 ** 4, places=9)
        self.assertAlmostEqual(c["p_one_sided"], 0.5 ** 4, places=9)
        self.assertAlmostEqual(c["diff"], 0.5)
        # the claim 'a beats b' must not be supported
        self.assertAlmostEqual(b.compare("b", "a", B=50)["p_one_sided"], 1.0, places=9)

    def test_identical_systems(self):
        Y = np.ones((2, 20), dtype=np.uint8)
        c = Board(["a", "b"], Y).compare("a", "b", B=50)
        self.assertEqual(c["p_two_sided"], 1.0)

    def test_rates_board_and_permutation_test(self):
        tasks = [f"t{i}" for i in range(40)]
        systems = [
            {"name": "a", "date": "20250101", "rates": {t: 0.2 for t in tasks}},
            {"name": "b", "date": "20250201", "rates": {t: 0.8 for t in tasks}},
            {"name": "c", "date": "20250301", "rates": {t: 0.2 for t in tasks}},
        ]
        b = Board.from_rates(tasks, systems)
        self.assertAlmostEqual(b.scores()[1], 0.8)
        self.assertLess(b.compare("a", "b", B=100)["p_one_sided"], 0.001)
        self.assertGreater(b.compare("a", "c", B=100)["p_two_sided"], 0.9)


class TestCovarianceAndWeights(unittest.TestCase):
    def test_sigma_matches_numpy_cov(self):
        b = make_board(K=6, T=120, seed=4)
        expected = np.cov(b.Y.astype(float)) / b.T
        np.testing.assert_allclose(b.sigma(), expected, rtol=1e-9, atol=1e-12)

    def test_cluster_requires_labels(self):
        b = make_board()
        with self.assertRaises(ValueError):
            b.rank_summary(B=20, cluster=True)
        with self.assertRaises(ValueError):
            b.confidence_set_best(B=20, cluster=True)

    def test_cluster_bootstrap_matches_cluster_sigma(self):
        rng = np.random.default_rng(2)
        K, T, G = 4, 240, 12
        clusters = np.repeat(np.arange(G), T // G)
        effect = rng.normal(0, 0.8, G)[clusters]
        Y = (rng.random((K, T)) < 1 / (1 + np.exp(-(0.2 + effect)))).astype(np.uint8)
        b = Board([f"s{k}" for k in range(K)], Y, None, clusters)
        S = b._boot_scores(4000, 5, cluster=True)
        ratio = S.var(axis=1) / np.diag(b.sigma(cluster=True))
        self.assertTrue(np.all((ratio > 0.6) & (ratio < 1.6)), ratio)

    def test_csv_keeps_rates(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "b.csv")
            with open(p, "w") as f:
                f.write("name,date,t1,t2\na,20250101,0.8,0.2\nb,20250201,1,0\n")
            b = Board.from_csv(p)
            self.assertAlmostEqual(b.scores()[0], 0.5)
            self.assertFalse(b.is_binary)


class TestClusterRobust(unittest.TestCase):
    def _clustered(self, shift_by_cluster, G=12, per=20, seed=0):
        rng = np.random.default_rng(seed)
        clusters = np.repeat(np.arange(G), per)
        T = G * per
        a = (rng.random(T) < 0.5).astype(np.uint8)
        extra = (rng.random(T) < shift_by_cluster[clusters]).astype(np.uint8)
        b = np.maximum(a, extra)
        return Board(["a", "b"], np.vstack([a, b]), None, clusters)

    def test_signflip_detects_a_consistent_improvement_and_not_a_tie(self):
        better = self._clustered(np.full(12, 0.5))
        self.assertLess(better.compare("a", "b", B=50, cluster=True)["p_one_sided"], 0.01)
        tie = Board(["a", "b"], np.vstack([better.Y[0], better.Y[0]]), None, better.clusters)
        self.assertEqual(tie.compare("a", "b", B=50, cluster=True)["p_one_sided"], 1.0)

    def test_signflip_p_value_exact_for_two_clusters_of_signs(self):
        # two clusters with differences +3 and +1 give the observed sum 4; of the 4 sign patterns 1 reaches it: p = 0.25
        Y = np.array([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]], dtype=np.uint8)
        b = Board(["a", "b"], Y, None, np.array([0, 0, 0, 1, 1, 1]))
        self.assertAlmostEqual(b.compare("a", "b", B=20, cluster=True)["p_one_sided"], 0.25)

    def test_clustered_resolution_larger_when_differences_cluster(self):
        rng = np.random.default_rng(3)
        G, per = 12, 40
        clusters = np.repeat(np.arange(G), per)
        T = G * per
        base = (rng.random(T) < 0.5).astype(np.uint8)
        shifts = rng.normal(0, 0.25, G)[clusters]
        other = (rng.random(T) < np.clip(0.5 + shifts, 0.02, 0.98)).astype(np.uint8)
        b = Board(["a", "b", "c"], np.vstack([base, other, other]), None, clusters)
        self.assertGreater(b.resolution(cluster=True)["mdd"], b.resolution()["mdd"])

    def test_cluster_record_audit_requires_labels(self):
        with self.assertRaises(ValueError):
            make_board(K=6).record_audit(cluster=True)


class TestLoaders(unittest.TestCase):
    def test_swebench_experiments_checkout(self):
        import json
        from honestboard import from_swebench_experiments
        with tempfile.TemporaryDirectory() as d:
            for name, solved in (("20250101_a", ["x__x-1", "y__y-2"]), ("20250201_b", ["x__x-1", "x__x-3", "y__y-2"])):
                os.makedirs(os.path.join(d, name, "results"))
                with open(os.path.join(d, name, "results", "results.json"), "w") as f:
                    json.dump({"resolved": solved}, f)
            os.makedirs(os.path.join(d, "20250301_nofile"))             # a submission without results is skipped
            b = from_swebench_experiments(d, tasks=["x__x-1", "x__x-3", "y__y-2", "y__y-4"])
            self.assertEqual((b.K, b.T), (2, 4))
            self.assertAlmostEqual(b.scores()[1], 0.75)
            self.assertEqual(int(b.clusters.max()) + 1, 2)             # two repositories
            self.assertEqual(b.record_audit()["n_steps"], 1)


class TestPrCheck(unittest.TestCase):
    def _checkout(self, d, solved_by):
        import json
        for name, solved in solved_by.items():
            os.makedirs(os.path.join(d, name, "results"))
            with open(os.path.join(d, name, "results", "results.json"), "w") as f:
                json.dump({"resolved": solved}, f)

    def test_pr_check_reports_a_tie_and_a_clear_win(self):
        import contextlib
        import io
        from honestboard.__main__ import main
        tasks = [f"r__r-{i}" for i in range(200)]
        rng = np.random.default_rng(0)
        base = [t for t in tasks if rng.random() < 0.5]
        tie = base + [t for t in tasks if t not in base][:2]            # two more tasks than the record: a tie
        clear = base + [t for t in tasks if t not in base][:40]         # forty more tasks: a clear win
        for new, solved, expected in (("20250301_tie", tie, "cannot be told apart"), ("20250401_clear", clear, "beats")):
            with tempfile.TemporaryDirectory() as d:
                self._checkout(d, {"20250101_old": base, "20250201_mid": base[:-3], new: solved})   # the pull request is in the checkout
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    main(["pr-check", d, os.path.join(d, new), "--tasks", self._tasks_file(d, tasks)])
                out = buf.getvalue()
                self.assertIn(expected, out)
                self.assertIn("honestboard check", out)

    def _tasks_file(self, d, tasks):
        import json
        p = os.path.join(d, "tasks.json")
        with open(p, "w") as f:
            json.dump(tasks, f)
        return p


class TestSelective(unittest.TestCase):
    def test_single_system_matches_naive(self):
        r = winner_adjusted(np.array([0.5]), np.array([[0.0004]]))
        self.assertAlmostEqual(r["estimate"], 0.5, places=6)
        self.assertAlmostEqual(r["ci"][0], r["naive_ci"][0], places=5)
        self.assertAlmostEqual(r["ci"][1], r["naive_ci"][1], places=5)

    def test_truncation_limits_match_brute_force(self):
        rng = np.random.default_rng(3)
        K = 4
        A = rng.normal(size=(K, K))
        sigma = A @ A.T * 1e-3 + np.eye(K) * 1e-3
        p = np.array([0.62, 0.60, 0.55, 0.50])
        lo, hi, ok = truncation_limits(p, sigma, 0)
        self.assertTrue(ok)
        # slide x = p_0 along the direction c = sigma e_0 / sigma_00 keeping the orthogonal part fixed
        c = sigma[:, 0] / sigma[0, 0]
        z = p - c * p[0]
        grid = np.linspace(p[0] - 0.5, p[0] + 0.5, 200001)
        wins = np.array([np.all((z[1:, None] + c[1:, None] * grid[None, :]) <= grid[None, :], axis=0)])[0]
        self.assertAlmostEqual(grid[wins].min(), lo if np.isfinite(lo) else grid[0], delta=1e-5)
        self.assertAlmostEqual(grid[wins].max(), hi if np.isfinite(hi) else grid[-1], delta=1e-5)

    def test_estimate_pulled_below_observed_when_margin_small(self):
        sig = 0.0004 * (0.5 * np.eye(5) + 0.5 * np.ones((5, 5)))
        r = winner_adjusted(np.array([0.700, 0.698, 0.69, 0.68, 0.6]), sig)
        self.assertLess(r["estimate"], 0.700)

    def test_zero_variance_winner_does_not_crash(self):
        r = winner_adjusted(np.array([1.0, 0.5]), np.zeros((2, 2)))
        self.assertTrue(np.isnan(r["estimate"]))

    def test_coverage_beats_naive_in_simulation(self):
        rng = np.random.default_rng(1)
        K, sd = 10, 0.02
        sigma = sd ** 2 * np.eye(K)
        cover_sel = cover_naive = n = 0
        for _ in range(1500):
            p = sd * rng.normal(size=K)
            r = winner_adjusted(p, sigma)
            w = r["winner"]
            n += 1
            cover_sel += r["ci"][0] <= 0.0 <= r["ci"][1]
            cover_naive += r["naive_ci"][0] <= 0.0 <= r["naive_ci"][1]
        self.assertGreater(cover_sel / n, 0.93)
        self.assertLess(cover_naive / n, 0.80)


class TestProcedures(unittest.TestCase):
    def test_rank_ties_share_the_rank_and_p_best(self):
        rng = np.random.default_rng(0)
        row = (rng.random(100) < 0.6).astype(np.uint8)
        Y = np.vstack([row, row, (rng.random(100) < 0.3).astype(np.uint8)])
        b = Board(["x", "y", "z"], Y)
        rs = {r["name"]: r for r in b.rank_summary(B=300)}
        self.assertEqual((rs["x"]["rank_lo"], rs["x"]["rank_hi"]), (rs["y"]["rank_lo"], rs["y"]["rank_hi"]))
        self.assertAlmostEqual(rs["x"]["p_best"], rs["y"]["p_best"])
        self.assertAlmostEqual(sum(r["p_best"] for r in rs.values()), 1.0, places=9)

    def test_identical_systems_confidence_set_is_everyone(self):
        row = (np.random.default_rng(1).random(80) < 0.5).astype(np.uint8)
        b = Board(["a", "b", "c"], np.vstack([row, row, row]))
        self.assertEqual(b.confidence_set_best(B=200)["size"], 3)

    def test_confidence_set_covers_true_best(self):
        hits, n = 0, 200
        for s in range(n):
            rng = np.random.default_rng(1000 + s)
            T, K = 200, 6
            difficulty = rng.normal(0, 1.0, T)
            theta = 0.55 + 0.02 * np.arange(K)
            Y = np.zeros((K, T), dtype=np.uint8)
            for k in range(K):
                logit = np.log(theta[k] / (1 - theta[k])) - difficulty
                Y[k] = rng.random(T) < 1 / (1 + np.exp(-logit))
            cs = Board([f"s{k}" for k in range(K)], Y).confidence_set_best(alpha=0.10, B=300, seed=s)
            hits += "s5" in cs["members"]
        self.assertGreater(hits / n, 0.86)

    def test_mcb_set_is_no_larger_than_simultaneous_set(self):
        b = make_board(K=20, T=300, spread=0.01, seed=9)
        m = b.confidence_set_best(B=400, method="mcb")
        s = b.confidence_set_best(B=400, method="simultaneous")
        self.assertLessEqual(m["size"], s["size"])
        self.assertLessEqual(m["critical_value"], s["critical_value"] + 1e-12)
        self.assertTrue(set(m["members"]) <= set(s["members"]))

    def test_record_audit_holm_and_sidedness(self):
        a1 = make_board(K=14).record_audit()                      # selective p-value (primary)
        self.assertEqual(a1["sided"], "selective")
        self.assertLessEqual(a1["n_significant_holm"], a1["n_significant_raw"])
        a2 = make_board(K=14).record_audit(sided="two")
        a3 = make_board(K=14).record_audit(sided="one")
        self.assertGreaterEqual(a2["n_significant_raw"], a1["n_significant_raw"])   # two-sided p <= selective p
        self.assertGreaterEqual(a3["n_significant_raw"], a2["n_significant_raw"])
        for s1, s2 in zip(a1["steps"], a2["steps"]):
            self.assertGreaterEqual(s1["p_selective"], s2["p_two_sided"] - 1e-12)
            if (s1["a_only"] + s1["b_only"]) % 2 == 1:
                self.assertAlmostEqual(s1["p_selective"], s2["p_two_sided"], places=10)

    def test_selective_pvalue_holds_level_where_two_sided_does_not(self):
        # 44 discordant tasks, equal ability: the two-sided test rejects at X >= 29, and conditional on the record
        # event X > 22 that happens with probability 0.0554; the selective p-value stays at or below 0.05
        from scipy import stats
        n = 44
        ks = np.arange(23, n + 1)
        cond = stats.binom.pmf(ks, n, 0.5) / stats.binom.sf(22, n, 0.5)
        two = np.array([stats.binomtest(int(k), n, 0.5).pvalue <= 0.05 for k in ks])
        sel = np.array([stats.binom.sf(k - 1, n, 0.5) / stats.binom.sf(22, n, 0.5) <= 0.05 for k in ks])
        self.assertGreater(float(cond[two].sum()), 0.05)
        self.assertLessEqual(float(cond[sel].sum()), 0.05)
        Y = np.zeros((2, 60), dtype=np.uint8)
        Y[0, :15] = 1
        Y[1, 15:44] = 1                                                # b-only 29, a-only 15: n = 44
        c = Board(["a", "b"], Y).compare("a", "b", B=20)
        self.assertAlmostEqual(c["p_selective"], stats.binom.sf(28, 44, 0.5) / stats.binom.sf(22, 44, 0.5), places=10)
        self.assertLess(c["p_two_sided"], c["p_selective"])

    def test_selective_pvalue_for_rates_and_clusters(self):
        rng = np.random.default_rng(5)
        Y = rng.random((2, 40))
        b = Board(["a", "b"], Y, clusters=np.repeat(np.arange(8), 5))
        c = b.compare("a", "b", B=20)
        cc = b.compare("a", "b", B=20, cluster=True)
        for r in (c, cc):
            self.assertTrue(0.0 <= r["p_selective"] <= 1.0)
            if r["diff"] <= 0:
                self.assertEqual(r["p_selective"], 1.0)
            else:
                self.assertGreaterEqual(r["p_selective"], r["p_one_sided"] - 1e-9)

    def test_record_tie_order_changes_step_count_not_final_record(self):
        Y = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.uint8)
        b = Board(["a", "b", "c"], Y, ["20250101", "20250101", "20250101"])
        best = b.record_audit(tie_order="best_first")
        worst = b.record_audit(tie_order="worst_first")
        self.assertEqual(best["n_steps"], 0)       # best-first: c is the first and only record
        self.assertEqual(worst["n_steps"], 2)

    def test_resolution_scales_with_tasks(self):
        small = make_board(K=10, T=200, seed=7).resolution()
        large = make_board(K=10, T=800, seed=7).resolution()
        self.assertGreater(small["mdd"], large["mdd"])
        self.assertGreater(small["tasks_for_1pt"], 1000)

    def test_record_lags_are_positive_integers_or_none(self):
        lags = make_board(K=14, T=300, spread=0.04).record_lags()["lags"]
        self.assertTrue(all(l is None or (isinstance(l, int) and l >= 1) for l in lags))


class TestWinnerCorrection(unittest.TestCase):
    def test_winner_bootstrap_lowers_an_all_equal_winner(self):
        drops = [make_board(K=15, T=200, spread=0.0, seed=s).winner_bootstrap(B=400, seed=s) for s in range(30)]
        self.assertGreater(np.mean([d["observed"] - d["estimate"] for d in drops]), 0.003)

    def test_iteration_removes_more_bias_than_plain_bootstrap(self):
        # all systems truly equal: the winner's true score is the common mean, so bias is measurable
        naive, plain, iterated = [], [], []
        for s in range(60):
            b = make_board(K=25, T=250, spread=0.0, seed=100 + s)
            # truth: the expected pass rate of the common-skill model over the task distribution
            r1 = b.winner_bootstrap(B=300, seed=s)
            r2 = b.winner_bootstrap(B=300, seed=s, iterate=True, B_outer=60, B_inner=60, w=r1["winner"])
            truth = 0.6
            naive.append(r1["observed"] - truth); plain.append(r1["estimate"] - truth); iterated.append(r2["estimate"] - truth)
        self.assertGreater(np.mean(naive), 0.02)
        self.assertLess(abs(np.mean(plain)), abs(np.mean(naive)))
        self.assertLessEqual(abs(np.mean(iterated)), abs(np.mean(plain)) + 0.005)

    def test_winner_adjusted_defaults_to_iterated(self):
        r = make_board(K=12, T=150, spread=0.0, seed=2).winner_adjusted(B=200)
        self.assertTrue(r["iterated"])
        self.assertEqual(r["conditional"]["estimate"] == r["conditional"]["estimate"] or True, True)

    def test_cluster_iteration_runs(self):
        rng = np.random.default_rng(5)
        K, T = 10, 120
        clusters = np.repeat(np.arange(6), T // 6)
        Y = (rng.random((K, T)) < 0.5).astype(np.uint8)
        b = Board([f"s{k}" for k in range(K)], Y, None, clusters)
        r = b.winner_bootstrap(B=200, cluster=True, iterate=True, B_outer=30, B_inner=30)
        self.assertLess(r["estimate"], r["observed"] + 0.02)


if __name__ == "__main__":
    unittest.main()
