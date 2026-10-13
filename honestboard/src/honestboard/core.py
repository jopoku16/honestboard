"""Core data structure and procedures: paired comparison, rank uncertainty, confidence set for the
best system, record-claim audit, winner's-curse correction, resolution limits. All procedures use
per-task outcomes of every system on the same task set. Tasks are the resampling unit; an optional
cluster label (for example the repository) gives a cluster bootstrap as a sensitivity check.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import stats

from . import selective
from .chain import selective_sign_pvalue


def _cluster_labels(tasks, cluster_fn):
    labels = [cluster_fn(t) for t in tasks]
    uniq = {l: i for i, l in enumerate(sorted(set(labels)))}
    return np.array([uniq[l] for l in labels])


@dataclass
class Board:
    names: list
    Y: np.ndarray                      # K x T matrix of outcomes in [0, 1]
    dates: Optional[list] = None       # sortable labels such as "20251205"
    clusters: Optional[np.ndarray] = None   # length T cluster labels (ints)
    task_ids: Optional[list] = None

    def __post_init__(self):
        self.Y = np.asarray(self.Y)
        if self.Y.ndim != 2:
            raise ValueError("Y must be a K x T matrix of outcomes")
        if len(self.names) != self.Y.shape[0]:
            raise ValueError("names must have one entry per system")
        if len(set(self.names)) != len(self.names):
            raise ValueError("system names must be unique")
        if self.Y.shape[1] < 2:
            raise ValueError("a board needs at least two tasks")
        if np.any(self.Y < 0) or np.any(self.Y > 1):
            raise ValueError("outcomes must lie in [0, 1]")
        if self.dates is not None and len(self.dates) != self.Y.shape[0]:
            raise ValueError("dates must have one entry per system")
        if self.clusters is not None:
            c = np.asarray(self.clusters)
            if c.shape != (self.Y.shape[1],):
                raise ValueError("clusters must give one label per task")
            self.clusters = np.unique(c, return_inverse=True)[1]      # contiguous labels 0..G-1, no empty clusters

    # ---- construction -------------------------------------------------
    @classmethod
    def from_resolved(cls, tasks: Sequence[str], systems: Sequence[dict], cluster_fn=None, clusters=None):
        """tasks: all task ids; systems: dicts with name, resolved (iterable of ids), optional date."""
        idx = {t: i for i, t in enumerate(tasks)}
        if len(idx) != len(tasks):
            raise ValueError("duplicate task ids")
        Y = np.zeros((len(systems), len(tasks)), dtype=np.uint8)
        for k, s in enumerate(systems):
            for t in s["resolved"]:
                if t in idx:
                    Y[k, idx[t]] = 1
        if cluster_fn is not None:
            clusters = _cluster_labels(tasks, cluster_fn)
        dates = [s.get("date") for s in systems]
        return cls([s["name"] for s in systems], Y, dates if any(dates) else None, clusters, list(tasks))

    @classmethod
    def from_rates(cls, tasks: Sequence[str], systems: Sequence[dict], cluster_fn=None, clusters=None):
        """Like from_resolved, but each system gives a per-task success rate in [0, 1] (for example the
        share of repeated trials that passed): systems = [{"name", "date", "rates": {task: rate}}]."""
        if len(set(tasks)) != len(tasks):
            raise ValueError("duplicate task ids")
        for s in systems:
            missing = [t for t in tasks if t not in s["rates"]]
            if missing:
                raise ValueError(f"system {s['name']} has no rate for {len(missing)} tasks (first: {missing[0]})")
        Y = np.array([[s["rates"][t] for t in tasks] for s in systems], dtype=float)
        if cluster_fn is not None:
            clusters = _cluster_labels(tasks, cluster_fn)
        dates = [s.get("date") for s in systems]
        return cls([s["name"] for s in systems], Y, dates if any(dates) else None, clusters, list(tasks))

    @classmethod
    def from_json(cls, path, cluster_fn=None):
        """Board file: {"tasks": [...], "systems": [...], optional "clusters": {task: label}}."""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        clusters = None
        if d.get("clusters"):
            labels = [d["clusters"][t] for t in d["tasks"]]
            uniq = {l: i for i, l in enumerate(sorted(set(labels)))}
            clusters = np.array([uniq[l] for l in labels])
        loader = cls.from_rates if (d["systems"] and "rates" in d["systems"][0]) else cls.from_resolved
        return loader(d["tasks"], d["systems"], cluster_fn=cluster_fn, clusters=clusters)

    @classmethod
    def from_csv(cls, path):
        """CSV with header `name,date,<task ids...>`; cells are 0/1 or pass rates in [0, 1]."""
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        tasks = rows[0][2:]
        names, dates, Y = [], [], []
        for r in rows[1:]:
            names.append(r[0]); dates.append(r[1] or None)
            Y.append([float(v) for v in r[2:]])
        Y = np.array(Y, dtype=float)
        if np.all((Y == 0) | (Y == 1)):
            Y = Y.astype(np.uint8)
        return cls(names, Y, dates if any(dates) else None, None, tasks)

    # ---- basics -------------------------------------------------------
    @property
    def K(self):
        return self.Y.shape[0]

    @property
    def T(self):
        return self.Y.shape[1]

    @property
    def is_binary(self):
        return bool(np.all((self.Y == 0) | (self.Y == 1)))

    def scores(self):
        return self.Y.mean(axis=1)

    def index(self, name_or_idx):
        if isinstance(name_or_idx, (int, np.integer)):
            return int(name_or_idx)
        return self.names.index(name_or_idx)

    def _unit_of_task(self, cluster):
        """Resampling unit of each task: the task itself, or its cluster."""
        if cluster:
            if self.clusters is None:
                raise ValueError("cluster=True needs cluster labels; pass cluster_fn or a 'clusters' map when loading")
            if int(self.clusters.max()) + 1 < 2:
                raise ValueError("cluster resampling needs at least two clusters")
            return np.asarray(self.clusters)
        return np.arange(self.T)

    def sigma(self, cluster=False):
        """Covariance matrix of the K scores (K x K) under task (or cluster) sampling."""
        Y = self.Y.astype(float)
        R = Y - Y.mean(axis=1, keepdims=True)
        if cluster:
            unit = self._unit_of_task(True)
            G = int(unit.max()) + 1
            S = np.zeros((self.K, G))
            for g in range(G):
                S[:, g] = R[:, unit == g].sum(axis=1)
            return (G / (G - 1)) * (S @ S.T) / self.T ** 2
        return (R @ R.T) / (self.T * (self.T - 1))

    def _boot_scores(self, B, seed, cluster=False):
        """K x B bootstrap scores; resamples tasks, or clusters with their tasks."""
        rng = np.random.default_rng(seed)
        unit = self._unit_of_task(cluster)
        U = int(unit.max()) + 1
        counts = rng.multinomial(U, np.full(U, 1.0 / U), size=B).T.astype(float)     # U x B
        W = counts[unit, :]                                                         # T x B
        return (self.Y.astype(float) @ W) / W.sum(axis=0)[None, :]

    # ---- 1. paired comparison ----------------------------------------
    def compare(self, a, b, B=5000, seed=0, cluster=False):
        """Does system b beat system a? Exact paired test plus a paired bootstrap interval.

        0/1 data: exact McNemar. Rates: paired sign-flip permutation test on the mean difference.
        `p_one_sided` tests the claim 'b is better than a'; `p_two_sided` is the usual two-sided p-value;
        `p_selective` is the record test, P(statistic >= observed | statistic > 0): the one-sided tail conditional
        on b having scored higher, which is the event that makes a record step a record step. For 0/1 data it
        equals the two-sided p-value when the number of discordant tasks is odd and is slightly larger when it is
        even (the two-sided test then exceeds its level given the record event; with 44 discordant tasks its
        conditional size is 5.5%)."""
        i, j = self.index(a), self.index(b)
        ya, yb = self.Y[i], self.Y[j]
        diff = float(yb.mean() - ya.mean())
        if cluster:
            return self._compare_clustered(i, j, B, seed)
        if self.is_binary:
            n_b_only = int(((yb == 1) & (ya == 0)).sum())
            n_a_only = int(((ya == 1) & (yb == 0)).sum())
            disc = n_a_only + n_b_only
            if disc == 0:
                p1 = p2 = 1.0
            else:
                p1 = float(stats.binomtest(n_b_only, disc, 0.5, alternative="greater").pvalue)
                p2 = float(stats.binomtest(n_b_only, disc, 0.5).pvalue)
            ps = selective_sign_pvalue(n_b_only, n_a_only)
        else:
            d = (yb - ya).astype(float)
            n_b_only, n_a_only = int((d > 0).sum()), int((d < 0).sum())
            rng_p = np.random.default_rng(seed + 17)
            signs = rng_p.choice([-1.0, 1.0], size=(20000, d.size))
            null = (signs * d).mean(axis=1)
            obs = d.mean()
            p1 = float((1 + (null >= obs - 1e-15).sum()) / (1 + null.shape[0]))
            p2 = float((1 + (np.abs(null) >= abs(obs) - 1e-15).sum()) / (1 + null.shape[0]))
            pos = int((null > 1e-15).sum())
            ps = 1.0 if (obs <= 1e-15 or pos == 0) else float(min(1.0, (1 + int((null >= obs - 1e-15).sum())) / (1 + pos)))
        S = self._boot_scores(B, seed, cluster)
        dd = S[j] - S[i]
        return {
            "a": self.names[i], "b": self.names[j], "diff": diff,
            "ci": (float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))),
            "p_one_sided": p1, "p_two_sided": p2, "p_exact_mcnemar": p2, "p_selective": ps,
            "b_only": n_b_only, "a_only": n_a_only,
        }

    def _compare_clustered(self, i, j, B, seed):
        """Paired comparison that respects clusters (for example repositories).

        The p-values come from a cluster-level sign-flip randomisation test: the per-task differences are
        summed within each cluster and the cluster sums are given random signs (all 2^G patterns when
        G <= 14, otherwise 40000 random ones). It is valid if the cluster sums of the differences are
        symmetric about zero under the null. With few clusters it has little power, which is the point."""
        unit = self._unit_of_task(True)
        G = int(unit.max()) + 1
        d = self.Y[j].astype(float) - self.Y[i].astype(float)
        Dg = np.array([d[unit == g].sum() for g in range(G)])
        obs = Dg.sum()
        if G <= 14:
            signs = ((np.arange(2 ** G)[:, None] >> np.arange(G)) & 1) * 2.0 - 1.0
        else:
            signs = np.random.default_rng(seed + 29).choice([-1.0, 1.0], size=(40000, G))
        null = signs @ Dg
        n = null.shape[0]
        c1 = int((null >= obs - 1e-12).sum())
        c2 = int((np.abs(null) >= abs(obs) - 1e-12).sum())
        c_pos = int((null > 1e-12).sum())
        if G <= 14:                                   # exact enumeration
            p1, p2 = c1 / n, c2 / n
            ps = 1.0 if (obs <= 1e-12 or c_pos == 0) else min(1.0, c1 / c_pos)
        else:                                         # Monte Carlo: the observed pattern counts as one draw
            p1, p2 = (1 + c1) / (1 + n), (1 + c2) / (1 + n)
            ps = 1.0 if (obs <= 1e-12 or c_pos == 0) else min(1.0, (1 + c1) / (1 + c_pos))
        S = self._boot_scores(B, seed, True)
        dd = S[j] - S[i]
        return {"a": self.names[i], "b": self.names[j], "diff": float(d.mean()),
                "ci": (float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))),
                "p_one_sided": p1, "p_two_sided": p2, "p_exact_mcnemar": p2, "p_selective": float(ps),
                "b_only": int((d > 0).sum()), "a_only": int((d < 0).sum()), "clustered": True}

    # ---- 2. rank uncertainty -----------------------------------------
    def rank_summary(self, B=4000, seed=0, cluster=False, top=None):
        S = self._boot_scores(B, seed, cluster)
        ranks = stats.rankdata(-S, method="min", axis=0)          # tied systems share the better rank
        mx = S.max(axis=0, keepdims=True)
        is_best = (S >= mx - 1e-12)
        p_best = (is_best / is_best.sum(axis=0, keepdims=True)).mean(axis=1)
        sc = self.scores()
        order = np.argsort(-sc, kind="stable")
        if top:
            order = order[:top]
        return [{
            "name": self.names[k], "score": float(sc[k]),
            "rank_lo": float(np.percentile(ranks[k], 2.5, method="lower")),
            "rank_hi": float(np.percentile(ranks[k], 97.5, method="higher")),
            "p_best": float(p_best[k]),
        } for k in order]

    # ---- 3. confidence set for the best ------------------------------
    def confidence_set_best(self, alpha=0.05, B=4000, seed=0, cluster=False, pool=None, method="mcb"):
        """Systems that cannot be ruled out as the best (paired-bootstrap multiple comparisons with the best).

        For system i let M_i be the largest studentised paired deviation of any rival j from i,
        max_j [(p*_j - p*_i) - (p_j - p_i)] / se_ij, over bootstrap resamples. System i stays in the set if
        no rival beats it by more than c standard errors.
        method="mcb" (default, in the spirit of Hsu's multiple comparisons with the best): c is the largest
        over i of the (1 - alpha) quantile of M_i. A fixed truly best system is then retained with
        probability about 1 - alpha, which is all the set promises.
        method="simultaneous": c is the (1 - alpha) quantile of max_i M_i, which covers every system's
        comparison at once (a Tukey-type all-pairs bound); the set is much larger and the guarantee stronger.
        `pool` restricts the comparison to the top-scoring systems (faster on very large boards) but then
        conditions on a data-chosen pool. Coverage is approximate in both cases."""
        if self.K < 2:
            raise ValueError("a confidence set needs at least two systems")
        sc = self.scores()
        order = np.argsort(-sc, kind="stable")
        if pool is not None:
            order = order[:min(pool, self.K)]
        P = len(order)
        S = self._boot_scores(B, seed, cluster)[order]                # P x B
        p = sc[order]
        sig = self.sigma(cluster)[np.ix_(order, order)]
        var_d = np.diag(sig)[:, None] + np.diag(sig)[None, :] - 2 * sig
        se = np.sqrt(np.maximum(var_d, 1e-18))
        eye = np.eye(P, dtype=bool)
        se[eye] = 1.0
        gap0 = p[None, :] - p[:, None]                                # [i,j] = p_j - p_i
        chunk = max(1, int(2e7 // (P * P)))
        M = np.empty((P, B))                                          # M[i, b] = max_j studentised deviation of j from i
        for s in range(0, B, chunk):
            Sc = S[:, s:s + chunk]
            D = (Sc[None, :, :] - Sc[:, None, :]) - gap0[:, :, None]  # [i,j,b]
            T = D / se[:, :, None]
            T[eye] = -np.inf
            M[:, s:s + chunk] = T.max(axis=1)
        if method == "simultaneous":
            crit = float(np.quantile(M.max(axis=0), 1 - alpha))
        elif method == "mcb":
            crit = float(np.max(np.quantile(M, 1 - alpha, axis=1)))
        else:
            raise ValueError("method must be 'mcb' or 'simultaneous'")
        keep = np.all((gap0 <= crit * se + 1e-15) | eye, axis=1)
        kept = [self.names[order[i]] for i in range(P) if keep[i]]
        return {"alpha": alpha, "critical_value": crit, "size": len(kept), "members": kept, "method": method,
                "pool": None if pool is None else P, "pool_saturated": bool(pool is not None and keep[-1])}

    # ---- 4. record-claim audit ---------------------------------------
    def _record_chain(self, tie_order="best_first"):
        """Indices of the successive record holders, walking the submissions in date order.

        Submissions with the same date are processed best-first (fewest, largest steps: the most
        generous reading for the leaderboard) or worst-first (most steps)."""
        if self.dates is None:
            raise ValueError("record analysis needs dates")
        undated = [n for n, d in zip(self.names, self.dates) if not d]
        if undated:
            raise ValueError(f"{len(undated)} systems have no date (first: {undated[0]}); the record chain needs every date")
        if tie_order not in ("best_first", "worst_first"):
            raise ValueError("tie_order must be 'best_first' or 'worst_first'")
        sc = self.scores()
        sign = -1.0 if tie_order == "best_first" else 1.0
        order = sorted(range(self.K), key=lambda k: (self.dates[k] or "", sign * sc[k]))
        chain, cur = [], None
        for k in order:
            if cur is None or sc[k] > sc[cur] + 1e-12:
                chain.append(k)
                cur = k
        return chain

    def record_audit(self, alpha=0.05, tie_order="best_first", sided="selective", cluster=False):
        """Test every new record against the record it replaced.

        A record step is selected because the challenger scored higher. sided='selective' (default) uses
        the selective p-value, the one-sided tail conditional on that event, which is exactly
        super-uniform given the record event under the task-level null (and conservative when the holder
        is a running maximum). sided='two' is the two-sided exact p-value (equal to the selective one for
        an odd number of discordant tasks, slightly smaller for an even number); sided='one' is the
        one-sided p-value at level alpha, whose size among selected steps is about 2 alpha. Significance
        is p <= alpha; Holm's correction is applied over the record steps of this board. cluster=True uses
        the cluster-level sign-flip test instead (needs cluster labels)."""
        if sided not in ("one", "two", "selective"):
            raise ValueError("sided must be 'one', 'two' or 'selective'")
        chain = self._record_chain(tie_order)
        steps = []
        for a, b in zip(chain[:-1], chain[1:]):
            c = self.compare(a, b, B=10, seed=0, cluster=cluster)
            steps.append({"previous": self.names[a], "new": self.names[b], "date": self.dates[b],
                          "margin": c["diff"], "p_one_sided": c["p_one_sided"], "p_two_sided": c["p_two_sided"],
                          "p_selective": c["p_selective"],
                          "p": {"one": c["p_one_sided"], "two": c["p_two_sided"], "selective": c["p_selective"]}[sided],
                          "b_only": c["b_only"], "a_only": c["a_only"]})
        m = len(steps)
        ps = np.array([s["p"] for s in steps]) if m else np.array([])
        holm = np.zeros(m, dtype=bool)
        for rank, idx in enumerate(np.argsort(ps)):
            if ps[idx] <= alpha / (m - rank):
                holm[idx] = True
            else:
                break
        for s, h in zip(steps, holm):
            s["significant_raw"] = bool(s["p"] <= alpha)
            s["significant_holm"] = bool(h)
        return {"alpha": alpha, "sided": sided, "tie_order": tie_order, "n_steps": m,
                "n_significant_raw": int(sum(s["significant_raw"] for s in steps)),
                "n_significant_holm": int(holm.sum()), "steps": steps}

    def record_lags(self, alpha=0.05, tie_order="best_first", sided="selective"):
        """For each record, how many records back one has to go before the record beats that earlier
        record significantly (the record test as in record_audit, no multiplicity correction). None
        if it never does within the board's history."""
        chain = self._record_chain(tie_order)
        key = {"one": "p_one_sided", "two": "p_two_sided", "selective": "p_selective"}[sided]
        lags = []
        for i in range(1, len(chain)):
            lag = None
            for j in range(1, i + 1):
                if self.compare(chain[i - j], chain[i], B=10, seed=0)[key] <= alpha:
                    lag = j
                    break
            lags.append(lag)
        found = [l for l in lags if l is not None]
        return {"lags": lags, "median_lag": float(np.median(found)) if found else None,
                "share_adjacent": float(np.mean([l == 1 for l in lags])) if lags else None,
                "share_never": float(np.mean([l is None for l in lags])) if lags else None}

    # ---- resolution limits -------------------------------------------
    def resolution(self, alpha=0.05, power=0.8, top=10, cluster=False):
        """What the benchmark can resolve. Uses the paired spread of per-task differences between
        neighbours in the ranking of the `top` highest-scoring systems, which is what matters when the
        leaders are close.

        mdd: smallest difference in score detected with the stated power by the two-sided paired test at
        level alpha (one-sided at alpha/2; the selective record test has the same critical value up to the
        even-n correction).
        tasks_for_1pt: number of tasks needed to resolve a one-point difference. Normal approximation.
        cluster=True uses the cluster-robust variance of the paired difference, which is larger when tasks
        within a cluster move together (the number of tasks needed then assumes the same cluster structure)."""
        order = np.argsort(-self.scores(), kind="stable")[:top]
        if len(order) < 2:
            raise ValueError("need at least two systems")
        sig = self.sigma(cluster)
        sds = [float(np.sqrt(max(sig[a, a] + sig[b, b] - 2 * sig[a, b], 0.0) * self.T)) for a, b in zip(order[:-1], order[1:])]
        sd = float(np.median(sds))
        z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
        return {"paired_sd": sd, "mdd": float(z * sd / np.sqrt(self.T)),
                "tasks_for_1pt": float((z * sd / 0.01) ** 2), "tasks_for_2pt": float((z * sd / 0.02) ** 2)}

    # ---- 5. selection-adjusted winner --------------------------------
    def winner_conditional(self, alpha=0.05, cluster=False, pool=None, w=None):
        """Exact-under-normality conditional (polyhedral) estimate and interval for the winner.

        Unbounded when the winner's margin over the runner-up is (near) zero. `pool` conditions on
        a data-chosen subset without conditioning on the choice of the subset; leave it None."""
        sc = self.scores()
        cand = None if pool is None else np.argsort(-sc)[:pool]
        res = selective.winner_adjusted(sc, self.sigma(cluster), w=w, alpha=alpha, candidates=cand)
        res["name"] = self.names[res["winner"]]
        return res

    def winner_hybrid(self, alpha=0.05, cluster=False, w=None, beta=None, seed=0):
        """Hybrid conditional estimate and interval of Andrews, Kitagawa and McCloskey (2024) for the winner:
        the conditional method with its truncation set bounded by a level 1-beta projection interval, so the
        interval is never unbounded (see selective.winner_hybrid). Undefined at an exact tie for first place."""
        res = selective.winner_hybrid(self.scores(), self.sigma(cluster), w=w, alpha=alpha, beta=beta, seed=seed)
        res["name"] = self.names[res["winner"]]
        return res

    def winner_zoom(self, alpha=0.05, cluster=False, w=None, nu=None, seed=0):
        """Zoom (locally simultaneous) interval of Zrnic and Fithian (2025) for the winner's true score: a
        simultaneous interval over the plausible winners only (see selective.winner_zoom). The point estimate
        is the reported score."""
        res = selective.winner_zoom(self.scores(), self.sigma(cluster), w=w, alpha=alpha, nu=nu, seed=seed)
        res["name"] = self.names[res["winner"]]
        res["plausible_names"] = [self.names[j] for j in res["plausible"]]
        return res

    def winner_bootstrap(self, alpha=0.05, B=4000, seed=0, cluster=False, w=None, iterate=False, B_outer=150, B_inner=150):
        """Plug-in bootstrap correction of the winner's curse.

        In the bootstrap world the true scores are the observed scores, so the selection bias of the
        winner can be measured directly: delta* = (score of the bootstrap winner on the bootstrap
        sample) - (its observed score). The reported score is corrected by subtracting delta*, and
        the quantiles of delta* give a basic-bootstrap interval. Ties are broken at random.

        iterate=True adds a second bootstrap level that estimates and removes the bias left in the
        first correction (bias iteration: est = p - 2*b1 + mean(b1*))."""
        sc = self.scores()
        unit = self._unit_of_task(cluster)
        U = int(unit.max()) + 1
        Yf = self.Y.astype(float)
        S = self._boot_scores(B, seed, cluster)
        rng = np.random.default_rng(seed + 1)
        wstar = np.argmax(S + rng.random(S.shape) * 1e-12, axis=0)
        delta = S[wstar, np.arange(B)] - sc[wstar]
        if w is None:
            w = int(np.argmax(sc + rng.random(self.K) * 1e-12))
        bias = float(delta.mean())
        est = float(sc[w] - bias)
        if iterate:
            c_out = rng.multinomial(U, np.full(U, 1.0 / U), size=B_outer).astype(float)     # B_outer x U
            b1 = np.empty(B_outer)
            for b in range(B_outer):
                w_out = c_out[b][unit]                                                      # task weights, T
                p_b = (Yf @ w_out) / w_out.sum()                                           # outer-sample scores
                c_in = rng.multinomial(U, c_out[b] / c_out[b].sum(), size=B_inner).astype(float)
                w_in = c_in[:, unit]                                                        # B_inner x T
                S_in = (w_in @ Yf.T) / w_in.sum(axis=1, keepdims=True)                     # B_inner x K
                ws = np.argmax(S_in + rng.random(S_in.shape) * 1e-12, axis=1)
                b1[b] = float((S_in[np.arange(B_inner), ws] - p_b[ws]).mean())
            est = float(sc[w] - 2 * bias + b1.mean())
        centred = delta - delta.mean()
        lo = float(est - np.quantile(centred, 1 - alpha / 2))
        hi = float(est - np.quantile(centred, alpha / 2))
        return {"winner": w, "name": self.names[w], "observed": float(sc[w]), "bias": float(sc[w] - est),
                "estimate": est, "ci": (lo, hi), "iterated": bool(iterate)}

    def winner_adjusted(self, alpha=0.05, cluster=False, B=4000, seed=0, iterate=True, w=None):
        """Selection-adjusted score for the current winner (iterated bootstrap by default), with the
        conditional method alongside for reference. `w` names the winner when the top is tied."""
        res = self.winner_bootstrap(alpha=alpha, B=B, seed=seed, cluster=cluster, iterate=iterate, w=w)
        cond = self.winner_conditional(alpha=alpha, cluster=cluster, w=res["winner"])
        sc = self.scores()
        z = self.sigma(cluster)[res["winner"], res["winner"]] ** 0.5
        zq = stats.norm.ppf(1 - alpha / 2)
        res["naive_ci"] = (float(sc[res["winner"]] - zq * z), float(sc[res["winner"]] + zq * z))
        res["conditional"] = {"estimate": cond["estimate"], "ci": cond["ci"], "lower_gap": cond["lower_gap"]}
        return res

    # ---- one-call check ----------------------------------------------
    def check_claim(self, system, alpha=0.05, cluster=False, B=4000):
        """Is this system's claim to be the best supported? Plain-language facts in a dict."""
        if self.K < 2:
            raise ValueError("check_claim needs at least two systems")
        k = self.index(system)
        sc = self.scores()
        others = [j for j in np.argsort(-sc, kind="stable") if j != k]
        runner = others[0]
        rank_tbl = {r["name"]: r for r in self.rank_summary(B=B, cluster=cluster)}
        cs = self.confidence_set_best(alpha=alpha, B=B, cluster=cluster)
        out = {
            "system": self.names[k], "score": float(sc[k]), "rank_observed": int(1 + (sc > sc[k]).sum()),
            "rank_interval_95": (rank_tbl[self.names[k]]["rank_lo"], rank_tbl[self.names[k]]["rank_hi"]),
            "p_best": rank_tbl[self.names[k]]["p_best"],
            "in_confidence_set_of_best": self.names[k] in cs["members"],
            "confidence_set_size": cs["size"],
            "vs_best_other": self.compare(runner, k, cluster=cluster) if sc[k] >= sc[runner] else self.compare(k, runner, cluster=cluster),
        }
        if k == int(np.argmax(sc)):
            out["winner_adjusted"] = self.winner_adjusted(alpha=alpha, cluster=cluster)
        return out
