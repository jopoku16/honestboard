"""Calibration of the record-claim test and of the resolution formula, simulated from a model fitted to a board.

Design (fitted to SWE-bench Verified unless another board is given):
- task difficulty L_t: the fitted per-task logit of the mean pass rate of the 30 strongest systems, resampled;
- system k passes task t with probability sigmoid(a_k + L_t), with a_k set to give the target score;
- outcomes of different systems on the same task share a Gaussian copula with correlation rho, tuned so that the
  paired spread of per-task differences between two equally able systems matches the median paired spread between
  neighbouring top-10 systems on the real board (real systems share base models and harnesses, so they agree on far
  more tasks than shared difficulty alone would give).

N1  Global null: K systems submitted in sequence, all equally able, so every record is luck and the holder is a
    running maximum. Rejection rate among record steps (selective, one- and two-sided) and the chance that Holm
    rejects at least one step on a board.
N2  Selected-step null: holder and challenger equally able and the holder NOT luck-inflated; rejection rate among
    the steps where the challenger beat the holder (the record condition). Selection on the sign of the difference
    doubles the size of a one-sided test; the two-sided test exceeds its level slightly for an even number of
    discordant tasks; the selective test is exact.
P   Power of the selective and two-sided tests for true differences of 1, 2, 3 and 5 points, and at the formula's
    minimum detectable difference.

usage: python validate_record_test.py <R_null> [board json] [output tag]
Outputs results/validation_recordtest.json, or results/validation_recordtest_<tag>.json when a tag is given (used to
repeat the calibration with the model fitted to another board).
"""
import json
import os
import sys
import time

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "honestboard", "src"))
from honestboard import Board  # noqa: E402

RES = os.path.join(ROOT, "results")
rng = np.random.default_rng(2026)
Z2 = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)


def fit_board(path, ref=30):
    b = Board.from_json(path)
    Y = b.Y.astype(float)
    order = np.argsort(-Y.mean(1), kind="stable")
    mt = np.clip(Y[order[:ref]].mean(0), 0.03, 0.97)
    L = np.log(mt / (1 - mt))
    base = float(Y[order[:10]].mean())
    return L, base, float(b.resolution()["paired_sd"]), int(b.T)


def solve_a(L, target):
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + L)))).mean() < target:
            lo = mid
        else:
            hi = mid
    return mid


def draw(Lt, offsets, rho):
    """Outcomes (uint8 arrays shaped like Lt) for systems with ability offsets on tasks with difficulties Lt."""
    u = rng.normal(size=Lt.shape)
    out = []
    for a in offsets:
        p = 1 / (1 + np.exp(-(a + Lt)))
        z = np.sqrt(rho) * u + np.sqrt(1 - rho) * rng.normal(size=Lt.shape)
        out.append((stats.norm.cdf(z) < p).astype(np.uint8))
    return out


def paired_sd(rho, Lpool, base, n=60000):
    a = solve_a(Lpool, base)
    ya, yb = draw(rng.choice(Lpool, n), [a, a], rho)
    return float(np.std(yb.astype(float) - ya.astype(float), ddof=1))


def calibrate_rho(Lpool, base, target):
    lo, hi = 0.0, 0.995
    for _ in range(22):
        mid = (lo + hi) / 2
        if paired_sd(mid, Lpool, base) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def null_global(K, T, R, Lpool, base, rho):
    a = solve_a(Lpool, base)
    steps = sig1 = sig2 = sig3 = holm_any = 0
    pos = {"1": [0, 0, 0, 0], "2": [0, 0, 0, 0], "3+": [0, 0, 0, 0]}   # steps, one-sided, two-sided, selective rejections
    t0 = time.time()
    for _ in range(R):
        Lt = rng.choice(Lpool, T)
        Y = np.vstack(draw(Lt, [a] * K, rho))
        b = Board([f"s{k}" for k in range(K)], Y, [f"{20250101 + k}" for k in range(K)])
        sel = b.record_audit()                                    # selective p-value (primary)
        steps += sel["n_steps"]
        sig3 += sel["n_significant_raw"]
        holm_any += sel["n_significant_holm"] > 0
        for i, s in enumerate(sel["steps"]):
            key = "1" if i == 0 else ("2" if i == 1 else "3+")
            r1, r2 = s["p_one_sided"] <= 0.05, s["p_two_sided"] <= 0.05
            sig1 += r1
            sig2 += r2
            pos[key][0] += 1
            pos[key][1] += r1
            pos[key][2] += r2
            pos[key][3] += s["significant_raw"]
    out = {"K": K, "T": T, "R": R, "steps": steps, "mean_steps_per_board": steps / R,
           "per_step_rejection_one_sided": sig1 / steps, "per_step_rejection_two_sided": sig2 / steps,
           "per_step_rejection_selective": sig3 / steps,
           "holm_any_rejection_per_board": holm_any / R,
           "by_position": {k: {"steps": v[0], "one": v[1], "two": v[2], "sel": v[3]} for k, v in pos.items()}}
    bp = " | ".join(f"pos {k}: {v[0]} steps, one {v[1] / max(v[0], 1):.3f}, two {v[2] / max(v[0], 1):.3f}, sel {v[3] / max(v[0], 1):.3f}" for k, v in pos.items())
    print(f"N1 global null K={K} T={T}: steps/board {steps / R:.1f}, per-step rejection one-sided {sig1 / steps:.3f}, "
          f"two-sided {sig2 / steps:.3f}, selective {sig3 / steps:.3f}, Holm any {holm_any / R:.3f} ({time.time() - t0:.0f}s)\n   by position: {bp}", flush=True)
    return out


def null_selected(T, R, Lpool, base, rho, chunk=2000):
    a = solve_a(Lpool, base)
    n = rej1 = rej2 = rej3 = 0
    t0 = time.time()
    for start in range(0, R, chunk):
        r = min(chunk, R - start)
        Lt = rng.choice(Lpool, (r, T))
        ya, yb = draw(Lt, [a, a], rho)
        nb = ((yb == 1) & (ya == 0)).sum(1)
        na = ((ya == 1) & (yb == 0)).sum(1)
        sel = nb > na
        for x, m in zip(nb[sel], (nb + na)[sel]):
            n += 1
            rej1 += stats.binomtest(int(x), int(m), 0.5, alternative="greater").pvalue <= 0.05
            rej2 += stats.binomtest(int(x), int(m), 0.5).pvalue <= 0.05
            rej3 += stats.binom.sf(int(x) - 1, int(m), 0.5) / stats.binom.sf(int(m) // 2, int(m), 0.5) <= 0.05
    out = {"T": T, "pairs": R, "selected_steps": n, "rejection_one_sided": rej1 / n, "rejection_two_sided": rej2 / n,
           "rejection_selective": rej3 / n}
    print(f"N2 selected null T={T}: {n} record steps of {R} pairs; rejection one-sided {rej1 / n:.3f}, two-sided {rej2 / n:.3f}, "
          f"selective {rej3 / n:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return out


def power_study(T, deltas, Lpool, base, rho, R=3000):
    a0 = solve_a(Lpool, base)
    res = {"T": T, "base": base, "R": R, "power": {}}

    def power_at(delta):
        a1 = solve_a(Lpool, base + delta)
        Lt = rng.choice(Lpool, (R, T))
        ya, yb = draw(Lt, [a0, a1], rho)
        nb = ((yb == 1) & (ya == 0)).sum(1)
        na = ((ya == 1) & (yb == 0)).sum(1)
        nn = nb + na
        rej2 = [stats.binomtest(int(x), int(m), 0.5).pvalue <= 0.05 if m > 0 else False for x, m in zip(nb, nn)]
        rej1 = [stats.binomtest(int(x), int(m), 0.5, alternative="greater").pvalue <= 0.05 if m > 0 else False for x, m in zip(nb, nn)]
        rej3 = [(x > m / 2) and stats.binom.sf(int(x) - 1, int(m), 0.5) / stats.binom.sf(int(m) // 2, int(m), 0.5) <= 0.05 if m > 0 else False
                for x, m in zip(nb, nn)]
        sd = float(np.sqrt(((yb.astype(float) - ya.astype(float)).var(axis=1, ddof=1)).mean()))
        return float(np.mean(rej2)), float(np.mean(rej1)), float(np.mean(rej3)), sd

    for d in deltas:
        p2, p1, p3, sd = power_at(d)
        res["power"][f"{d:.3f}"] = {"power_two_sided": p2, "power_one_sided": p1, "power_selective": p3, "paired_sd": sd}
    sd0 = float(np.mean([v["paired_sd"] for v in res["power"].values()]))
    mdd = Z2 * sd0 / np.sqrt(T)
    p2, p1, p3, _ = power_at(float(mdd))
    res["formula_mdd"] = float(mdd)
    res["power_at_formula_mdd"] = p2
    res["power_at_formula_mdd_selective"] = p3
    res["paired_sd_sim"] = sd0
    print(f"P power T={T}: " + ", ".join(f"{float(k) * 100:.0f}pt -> {v['power_selective']:.2f}" for k, v in res["power"].items())
          + f" | formula MDD {mdd * 100:.1f} pts, simulated power there {p2:.2f} (two-sided), {p3:.2f} (selective)", flush=True)
    return res


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    board_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "data", "boards", "swebench_verified.json")
    Lpool, base, sd_real, T_real = fit_board(board_path)
    rho = calibrate_rho(Lpool, base, sd_real)
    sd_sim = paired_sd(rho, Lpool, base)
    design = {"board": os.path.basename(board_path), "difficulty_sd_logit": float(np.std(Lpool)), "base_score": base, "rho": rho,
              "paired_sd_real": sd_real, "paired_sd_sim": sd_sim, "K_null": 40, "T_real": T_real}
    print("design", json.dumps(design), flush=True)
    out = {"design": design,
           "null_global": [null_global(40, T, R, Lpool, base, rho) for T in (300, 500, 2000)],
           "null_selected": [null_selected(T, 20000, Lpool, base, rho) for T in (300, 500, 2000)],
           "power": [power_study(T, [0.01, 0.02, 0.03, 0.05], Lpool, base, rho) for T in (300, 500, 2000)]}
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    with open(os.path.join(RES, f"validation_recordtest{'_' + tag if tag else ''}.json"), "w") as f:
        json.dump(out, f, indent=2)
