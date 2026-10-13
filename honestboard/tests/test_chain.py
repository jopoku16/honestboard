import os
import sys
import unittest

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from honestboard import Board, certify_chain, selective_sign_pvalue  # noqa: E402
from honestboard.chain import alpha_spending_level, fallback_level, lord_gamma, lord_level, polyhedral_step_pvalue, tenure_level  # noqa: E402


class TestProposition1Exhaustively(unittest.TestCase):
    """Proposition 1 checked by exact enumeration. Systems: c (earlier), a (holder), b (challenger) and, for the
    certified-holder event, d (a later challenger that arrives between a and b). Per task c passes with its own
    probability; given c's outcome, a and b pass with task- and c-dependent probabilities under which b is no
    likelier than a to be the one that passes alone; d passes with its own probability. For every number n of
    discordant tasks between a and b, the conditional law of n_b given the holder event, the record event and n is
    compared with the fair-coin law restricted to {n_b > n/2}: P(n_b >= k | ...) <= P(X >= k | X > n/2) for every k,
    which is the same as P(p <= u | ...) <= u at every attainable u. Decreasing holder events are run as negative
    controls and must break the bound, so that the check has teeth."""

    Q_C = [0.3, 0.5, 0.7, 0.4, 0.6]                                                   # P(c passes task t)
    S = {1: [0.40, 0.30, 0.50, 0.35, 0.45], 0: [0.10, 0.15, 0.20, 0.05, 0.12]}         # P(both a and b pass | y_c)
    R = {1: [0.20, 0.25, 0.15, 0.22, 0.18], 0: [0.15, 0.20, 0.10, 0.25, 0.20]}         # P(a alone passes | y_c)
    Q_D = [0.5, 0.6, 0.4, 0.55, 0.45]                                                 # P(d passes task t)

    def _laws(self, holder_event, b_factor=1.0, with_d=False, T=5):
        import itertools
        singles = [(yc, ya, yb, yd) for yc in (0, 1) for ya in (0, 1) for yb in (0, 1) for yd in ((0, 1) if with_d else (0,))]

        def prob(t, yc, ya, yb, yd):
            pc = self.Q_C[t] if yc else 1 - self.Q_C[t]
            ra, rb = self.R[yc][t], b_factor * self.R[yc][t]
            joint = {(1, 1): self.S[yc][t], (1, 0): ra, (0, 1): rb, (0, 0): 1 - self.S[yc][t] - ra - rb}
            pd = (self.Q_D[t] if yd else 1 - self.Q_D[t]) if with_d else 1.0
            return pc * joint[(ya, yb)] * pd

        laws = {}
        for config in itertools.product(singles, repeat=T):
            p_cfg = 1.0
            for t, o in enumerate(config):
                p_cfg *= prob(t, *o)
            yc = [o[0] for o in config]; ya = [o[1] for o in config]; yb = [o[2] for o in config]; yd = [o[3] for o in config]
            n_b = sum(1 for o in config if o[2] == 1 and o[1] == 0)
            n_a = sum(1 for o in config if o[1] == 1 and o[2] == 0)
            if n_b > n_a and holder_event(yc, ya, yd):
                law = laws.setdefault(n_a + n_b, {})
                law[n_b] = law.get(n_b, 0.0) + p_cfg
        return laws

    @staticmethod
    def _excess(laws):
        """Largest value of P(n_b >= k | n, event, record) - P(X >= k | X > n/2) over all n and k."""
        worst, n_checked = -1.0, 0
        for n, law in laws.items():
            tot = sum(law.values())
            if tot < 1e-9:
                continue
            n_checked += 1
            fair = stats.binom.pmf(np.arange(n + 1), n, 0.5)
            fair_tail = fair[n // 2 + 1:].sum()
            for k in range(n // 2 + 1, n + 1):
                obs = sum(m for nb, m in law.items() if nb >= k) / tot
                worst = max(worst, obs - fair[k:].sum() / fair_tail)
        return worst, n_checked

    @staticmethod
    def _certified_against(y_ref, y_new, level):
        n_new = sum(1 for r, x in zip(y_ref, y_new) if x == 1 and r == 0)
        n_ref = sum(1 for r, x in zip(y_ref, y_new) if r == 1 and x == 0)
        return n_new > n_ref and selective_sign_pvalue(n_new, n_ref) <= level

    def _assert_valid(self, laws, min_laws=2):
        worst, n_checked = self._excess(laws)
        self.assertGreaterEqual(n_checked, min_laws)
        self.assertLessEqual(worst, 1e-9)

    def test_record_event_equal_and_strict_null(self):
        for b_factor in (1.0, 0.7):
            self._assert_valid(self._laws(lambda yc, ya, yd: sum(ya) > sum(yc), b_factor))

    def test_increasing_event_that_is_not_a_function_of_the_total(self):
        self._assert_valid(self._laws(lambda yc, ya, yd: ya[0] == 1 and ya[1] == 1 and sum(ya) > sum(yc)))

    def test_certified_holder_event_with_a_later_challenger(self):
        # a was certified against c at level 0.5, and the later submission d was not certified against a
        ev = lambda yc, ya, yd: self._certified_against(yc, ya, 0.5) and not self._certified_against(ya, yd, 0.5)
        self._assert_valid(self._laws(ev, with_d=True, T=4), min_laws=1)
        self._assert_valid(self._laws(ev, with_d=True, T=5), min_laws=2)

    def test_negative_controls_break_the_bound(self):
        # decreasing holder events violate the hypothesis of Proposition 1 and must fail the check
        for ev in (lambda yc, ya, yd: sum(ya) <= 1, lambda yc, ya, yd: ya[0] == 0 and sum(ya) <= 2):
            worst, _ = self._excess(self._laws(ev))
            self.assertGreater(worst, 0.05)


class TestMonotonicityLemma(unittest.TestCase):
    def test_selective_pvalue_monotone_in_added_tasks(self):
        # adding a task won by the challenger lowers the p-value; adding one won by the holder raises it
        from fractions import Fraction
        from math import comb

        def exact(k, n):                                                   # exact rational selective p-value
            return Fraction(sum(comb(n, j) for j in range(k, n + 1)), sum(comb(n, j) for j in range(n // 2 + 1, n + 1)))

        for n in range(1, 160):
            for k in range(n // 2 + 1, n + 1):
                p = exact(k, n)
                self.assertLessEqual(exact(k + 1, n + 1), p)                # b-only task added
                self.assertGreaterEqual(exact(k, n + 1), p)                 # a-only task added
        for n in (5, 44, 201):                                             # the float implementation matches
            for k in range(n // 2 + 1, n + 1):
                self.assertAlmostEqual(selective_sign_pvalue(k, n - k) / float(exact(k, n)), 1.0, places=9)


class TestSelectivePValue(unittest.TestCase):
    def test_matches_brute_force(self):
        for n_a, n_b in ((1, 4), (3, 7), (0, 6), (5, 6), (4, 4)):
            n = n_a + n_b
            if n_b <= n_a:
                self.assertEqual(selective_sign_pvalue(n_b, n_a), 1.0)
                continue
            pmf = stats.binom.pmf(np.arange(n + 1), n, 0.5)
            brute = pmf[n_b:].sum() / pmf[n // 2 + 1:].sum()
            self.assertAlmostEqual(selective_sign_pvalue(n_b, n_a), brute, places=12)

    def test_uniform_given_record_event(self):
        # under equal ability, conditional on n_b > n_a, P(p <= u) <= u (super-uniform)
        rng = np.random.default_rng(0)
        n = 41
        X = rng.binomial(n, 0.5, 200000)
        X = X[X > n / 2]
        ps = np.array([selective_sign_pvalue(x, n - x) for x in range(n // 2 + 1, n + 1)])
        for u in (0.01, 0.05, 0.1, 0.3):
            frac = np.mean(ps[X - (n // 2 + 1)] <= u)
            self.assertLessEqual(frac, u + 0.01)


class TestLevels(unittest.TestCase):
    def test_spending_sums_to_alpha(self):
        s = sum(alpha_spending_level(i, 0.05) for i in range(1, 20000))
        self.assertAlmostEqual(s, 0.05, places=4)

    def test_fallback_level_dominates_spending_and_accumulates_along_runs(self):
        lvl, prev = [], (0.0, False)
        for t in range(1, 6):
            l = fallback_level(t, 0.05, *prev)
            self.assertGreaterEqual(l, alpha_spending_level(t, 0.05) - 1e-15)
            lvl.append(l)
            prev = (l, True)                                              # every record certified: the budget accumulates
        self.assertAlmostEqual(lvl[-1], sum(alpha_spending_level(t, 0.05) for t in range(1, 6)), places=12)
        self.assertAlmostEqual(fallback_level(3, 0.05, 0.04, False), alpha_spending_level(3, 0.05), places=12)

    def test_tenure_levels_sum_to_alpha_over_any_chain(self):
        # the levels available over a whole chain: sum over holders j of gamma(j) times sum over m of gamma(m) <= alpha
        total = sum(tenure_level(0.05, j, m) for j in range(1, 300) for m in range(1, 300))
        self.assertLessEqual(total, 0.05 + 1e-9)
        self.assertGreater(total, 0.049)

    def test_tenure_rule_controls_fwer_and_is_certified_baseline_only(self):
        rng = np.random.default_rng(13)
        K, T, R = 12, 150, 300
        any_cert = 0
        for _ in range(R):
            Y = (rng.random((K, T)) < 0.5).astype(np.uint8)
            b = Board([f"s{k}" for k in range(K)], Y, [f"{20250101 + k}" for k in range(K)])
            any_cert += certify_chain(b, rule="tenure", baseline="certified")["n_certified"] > 0
        self.assertLessEqual(any_cert / R, 0.085)
        with self.assertRaises(ValueError):
            certify_chain(b, rule="tenure", baseline="leader")

    def test_same_date_best_first_needs_the_date_correction(self):
        # global null, one submission on day 1 and M on day 2, best-first ordering: the day-2 challenger is the best
        # of M, a selection the record event does not cover; the per-date Bonferroni factor restores the level
        rng = np.random.default_rng(14)
        M, T, R = 20, 100, 600
        fw_corr = fw_raw = 0
        for _ in range(R):
            Y = (rng.random((M + 1, T)) < 0.5).astype(np.uint8)
            b = Board([f"s{k}" for k in range(M + 1)], Y, ["20250101"] + ["20250102"] * M)
            fw_corr += certify_chain(b, rule="spending")["n_certified"] > 0
            fw_raw += certify_chain(b, rule="spending", date_correction=False)["n_certified"] > 0
        self.assertLessEqual(fw_corr / R, 0.08)
        self.assertGreater(fw_raw / R, 0.065)
        a = b.record_audit()
        self.assertTrue(a["date_correction"])
        self.assertTrue(all(s["date_group"] == M and s["p"] >= s["p_uncorrected"] for s in a["steps"]))

    def test_fallback_controls_fwer_under_global_null(self):
        rng = np.random.default_rng(11)
        K, T, R = 12, 150, 300
        any_cert = 0
        for _ in range(R):
            Y = (rng.random((K, T)) < 0.5).astype(np.uint8)
            b = Board([f"s{k}" for k in range(K)], Y, [f"{20250101 + k}" for k in range(K)])
            any_cert += certify_chain(b, rule="fallback")["n_certified"] > 0
        self.assertLessEqual(any_cert / R, 0.085)                         # nominal 0.05, Monte Carlo error about 0.013

    def test_fallback_certifies_at_least_as_many_as_spending_on_a_staircase(self):
        rng = np.random.default_rng(12)
        K, T = 8, 1000                                                    # 12-point jumps, well above the board's resolution
        Y = np.vstack([(rng.random(T) < 0.2 + 0.12 * k).astype(np.uint8) for k in range(K)])
        b = Board([f"s{k}" for k in range(K)], Y, [f"{20250101 + k}" for k in range(K)])
        sp = certify_chain(b, rule="spending")
        fb = certify_chain(b, rule="fallback")
        self.assertEqual(fb["n_tests"], sp["n_tests"])
        self.assertGreaterEqual(fb["n_certified"], sp["n_certified"])
        self.assertGreater(fb["n_certified"], 3)

    def test_lord_gamma_sums_to_about_one_and_lord_levels_nonnegative(self):
        g = sum(lord_gamma(j) for j in range(1, 100000))
        self.assertAlmostEqual(g, 1.0, places=4)
        # the published sequence sums to one only in the limit; any partial sum must stay below one
        gl = sum(lord_gamma(j, "log") for j in range(1, 100000))
        self.assertGreater(gl, 0.4)
        self.assertLessEqual(gl, 1.0)
        self.assertTrue(all(lord_gamma(j, "log") >= lord_gamma(j + 1, "log") for j in range(2, 2000)))
        lv = [lord_level(t, 0.05, 0.025, [3, 7]) for t in range(1, 30)]
        self.assertTrue(all(v >= 0 for v in lv))
        self.assertGreater(lord_level(8, 0.05, 0.025, [3, 7]), lord_level(8, 0.05, 0.025, []))


def make_chain_board(K, T, levels, rng, base=0.6):
    """K submissions on T tasks; true pass probability steps up by `levels` (per-system logit offsets)."""
    difficulty = rng.normal(0, 1.0, T)
    Y = np.zeros((K, T), dtype=np.uint8)
    for k in range(K):
        logit = np.log(base / (1 - base)) + levels[k] - difficulty
        Y[k] = rng.random(T) < 1 / (1 + np.exp(-logit))
    return Board([f"s{k}" for k in range(K)], Y, [f"{20250101 + k}" for k in range(K)])


class TestCertifyChain(unittest.TestCase):
    def test_spending_controls_fwer_under_global_null(self):
        rng = np.random.default_rng(1)
        any_cert = 0
        R = 400
        for _ in range(R):
            b = make_chain_board(25, 200, np.zeros(25), rng)
            any_cert += certify_chain(b, rule="spending")["n_certified"] > 0
        self.assertLess(any_cert / R, 0.05 + 2.5 * np.sqrt(0.05 * 0.95 / R))

    def test_lord_certifies_a_large_true_jump(self):
        rng = np.random.default_rng(2)
        hits = 0
        for _ in range(30):
            levels = np.zeros(20); levels[10:] = 1.5          # a big genuine improvement at submission 11
            b = make_chain_board(20, 300, levels, rng)
            res = certify_chain(b, rule="lord")
            hits += "s10" in res["certified"] or any(s["certified"] and int(s["new"][1:]) >= 10 for s in res["steps"])
        self.assertGreater(hits / 30, 0.8)

    def test_certified_baseline_moves_only_on_certification(self):
        rng = np.random.default_rng(3)
        b = make_chain_board(15, 150, np.zeros(15), rng)
        res = certify_chain(b, rule="spending", baseline="certified")
        refs = [s["reference"] for s in res["steps"]]
        # every reference is either the first submission or a certified record
        allowed = {"s0"} | set(res["certified"])
        self.assertTrue(all(r in allowed for r in refs))

    def test_holm_and_unadjusted_run(self):
        rng = np.random.default_rng(4)
        b = make_chain_board(12, 100, np.zeros(12), rng)
        h = certify_chain(b, rule="holm")
        u = certify_chain(b, rule="unadjusted")
        self.assertLessEqual(h["n_certified"], u["n_certified"])

    def test_polyhedral_step_pvalue_valid_under_null(self):
        # Gaussian scores, all equal means: among record steps the selective p-value is super-uniform
        rng = np.random.default_rng(5)
        K = 6
        A = rng.normal(size=(K, K)); sigma = (A @ A.T) * 1e-4 + np.eye(K) * 1e-4
        L = np.linalg.cholesky(sigma)
        ps = []
        for _ in range(4000):
            p = L @ rng.normal(size=K)
            earlier = list(range(K - 1)); b = K - 1
            a = int(np.argmax(p[:-1]))
            if p[b] > p[a]:
                ps.append(polyhedral_step_pvalue(p, sigma, b, a, earlier))
        ps = np.array(ps)
        self.assertGreater(len(ps), 300)
        for u in (0.05, 0.1, 0.2):
            self.assertLessEqual(np.mean(ps <= u), u + 0.03)


if __name__ == "__main__":
    unittest.main()
