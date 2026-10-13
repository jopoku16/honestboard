"""Hybrid (Andrews, Kitagawa and McCloskey) and zoom (Zrnic and Fithian) corrections for the winner."""
import numpy as np
import pytest
from scipy import stats

from honestboard import Board
from honestboard.selective import simultaneous_quantile, winner_hybrid, winner_zoom


def _equal_board(rng, K=8, T=300, theta=0.5):
    Y = (rng.random((K, T)) < theta).astype(float)
    return Board([f"s{k}" for k in range(K)], Y)


def test_simultaneous_quantile_between_pointwise_and_bonferroni():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 40))
    sigma = A @ A.T / 40
    c = simultaneous_quantile(sigma, 0.05, seed=1)
    assert stats.norm.ppf(0.975) < c <= stats.norm.ppf(1 - 0.05 / 12) + 0.05
    # independent systems: the quantile of max |Z| over m coordinates is Phi^-1((1 + (1-tail)^(1/m)) / 2)
    c_ind = simultaneous_quantile(np.eye(6), 0.05, seed=1)
    assert abs(c_ind - stats.norm.ppf((1 + 0.95 ** (1 / 6)) / 2)) < 0.05


def test_hybrid_inside_projection_and_covers_equal_leaders():
    rng = np.random.default_rng(1)
    cover, inside, defined = [], [], 0
    for _ in range(200):
        b = _equal_board(rng)
        r = b.winner_hybrid(seed=int(rng.integers(1 << 30)))
        if r["degenerate"]:
            continue
        defined += 1
        lo, hi = r["ci"]
        plo, phi = r["projection_ci"]
        inside.append(plo - 1e-9 <= lo <= hi <= phi + 1e-9)
        cover.append(lo <= 0.5 <= hi)
        assert lo <= r["estimate"] <= hi
    assert defined > 100
    assert all(inside)
    assert np.mean(cover) >= 0.92          # nominal 0.95, Monte Carlo error about 0.015


def test_hybrid_undefined_at_exact_tie():
    Y = np.zeros((3, 20))
    Y[0, :10] = 1
    Y[1, 10:] = 1
    r = Board(["a", "b", "c"], Y).winner_hybrid(w=0)
    assert r["degenerate"] and np.isnan(r["estimate"])


def test_zoom_contains_winner_and_covers():
    rng = np.random.default_rng(2)
    cover = []
    for _ in range(200):
        b = _equal_board(rng)
        r = b.winner_zoom(seed=int(rng.integers(1 << 30)))
        assert r["winner"] in r["plausible"]
        assert r["c_final"] <= r["c_zoom"] + 1e-9     # fewer systems and a larger budget: never a wider final quantile
        lo, hi = r["ci"]
        cover.append(lo <= 0.5 <= hi)
    assert np.mean(cover) >= 0.92


def test_zoom_shrinks_to_a_clear_winner():
    rng = np.random.default_rng(3)
    Y = (rng.random((6, 400)) < 0.3).astype(float)
    Y[0] = (rng.random(400) < 0.8)
    r = Board([f"s{k}" for k in range(6)], Y).winner_zoom()
    assert r["n_plausible"] == 1
    assert abs(r["c_final"] - stats.norm.ppf(1 - 0.045 / 2)) < 1e-6


def test_hybrid_rejects_bad_beta():
    b = _equal_board(np.random.default_rng(4))
    with pytest.raises(ValueError):
        b.winner_hybrid(beta=0.1)
