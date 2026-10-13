"""Paper figures. Writes figures/*.pdf and *.png.

fig1_forest        Hero: every record claim on four leaderboards, margin over the previous record with a 95% interval.
fig2_resolution    What a point costs: the smallest detectable difference against the number of tasks, with the six boards.
fig3_rank_intervals Rank uncertainty of the leaders, with probability of being best.
fig4_winners_curse  Overstatement of the winner's score, reported versus corrected.
fig5_ladders       Appendix: the SOTA ladders with significance marked.

Palette: two validated categorical slots (blue, orange) plus neutral ink (dataviz skill palette).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "honestboard", "src"))
from honestboard import Board  # noqa: E402

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

SURFACE, INK, INK2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, WASH = "#e1e0d9", "#c3c2b7", "#f4f3ef"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def short(name, n=34):
    s = name.split("_", 1)[1] if name[:8].isdigit() and "_" in name else name
    s = s.replace("__", " / ")
    return s if len(s) <= n else s[: n - 1] + "…"


def load(fname):
    return Board.from_json(os.path.join(ROOT, "data", "boards", fname))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


def panel_label(ax, text, x=-0.02, y=1.04):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, ha="left", va="bottom")


# ---------------------------------------------------------------- Fig 1: forest of record claims
def fig_forest():
    spec = [("SWE-bench Verified", "swebench_verified.json"), ("SWE-bench Lite", "swebench_lite.json"),
            ("Terminal-Bench 2", "terminalbench.json"), ("SWE-bench Multimodal", "swebench_multimodal.json")]
    data = []
    for label, fname in spec:
        b = load(fname)
        a = b.record_audit()
        rows = []
        S = b._boot_scores(3000, 1)                 # paired bootstrap; the 95% interval matches the two-sided 5% test
        for s in a["steps"]:
            d = S[b.names.index(s["new"])] - S[b.names.index(s["previous"])]
            rows.append({"date": s["date"], "name": s["new"], "margin": 100 * s["margin"], "lo": 100 * np.percentile(d, 2.5), "hi": 100 * np.percentile(d, 97.5),
                         "sig": s["significant_raw"]})
        data.append((label, a, rows))
    heights = [len(d[2]) for d in data]
    fig = plt.figure(figsize=(7.4, 9.6))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.12, 1], height_ratios=heights[1:],
                  hspace=0.62, wspace=0.52, left=0.0, right=1.0, top=0.975, bottom=0.045)
    # Verified takes the whole left column; the other three stack on the right
    axes = [fig.add_subplot(gs[:, 0])] + [fig.add_subplot(gs[i, 1]) for i in range(3)]
    for k, (ax, (label, a, rows)) in enumerate(zip(axes, data)):
        n = len(rows)
        ys = np.arange(n, 0, -1)                    # newest on top
        xmax = 24 if k == 0 else 26
        ax.axvspan(-3, 0, color=WASH, zorder=0)
        ax.axvline(0, color=MUTED, lw=0.9, zorder=1)
        for y, r in zip(ys, rows[::-1]):
            col = BLUE if r["sig"] else MUTED
            ax.plot([r["lo"], min(r["hi"], xmax)], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=2, alpha=1.0 if r["sig"] else 0.8)
            ax.plot([r["margin"]], [y], "o", ms=5.2, mfc=BLUE if r["sig"] else SURFACE, mec=col, mew=1.2, zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{r['date'][:4]}-{r['date'][4:6]}  {short(r['name'], 24)}" for r in rows[::-1]], fontsize=6.6 if k == 0 else 7.0, color=INK2)
        ax.set_ylim(0.2, n + 0.9)
        ax.set_xlim(-4, xmax)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4 if k == 0 else 5))
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=7.5)
        ax.set_title(f"{label}: {a['n_significant_raw']} of {a['n_steps']} records distinguishable", loc="left", fontsize=9, color=INK, pad=5)
        if k in (0, 3):
            ax.set_xlabel("Margin over the previous record (percentage points)", fontsize=8)
    axes[0].text(-3.75, len(data[0][2]) + 0.45, "behind", fontsize=6.5, color=MUTED, ha="left", va="center")
    # legend in the empty right-hand part of the Multimodal panel (its points sit left of 6 points)
    ax = axes[3]
    ax.plot([], [], "o", ms=5.2, mfc=BLUE, mec=BLUE, mew=1.2, label="beats the record it replaced\n(selective exact p ≤ 0.05)")
    ax.plot([], [], "o", ms=5.2, mfc=SURFACE, mec=MUTED, mew=1.2, label="interval reaches zero:\nnot distinguishable")
    ax.legend(loc="center right", frameon=False, fontsize=6.6, labelcolor=INK2, handletextpad=0.4, borderaxespad=0.0, labelspacing=0.9)
    save(fig, "fig1_forest")


# ---------------------------------------------------------------- Fig 2: what a point costs
def fig_resolution():
    a = json.load(open(os.path.join(ROOT, "results", "audit_summary.json")))
    boards = [b for b in a["boards"] if "resolution" in b]
    z = 2.8016                                     # Phi^-1(0.975) + Phi^-1(0.80): two-sided 5% test, 80% power
    sds = [b["resolution"]["paired_sd"] for b in boards]
    sd_v = next(b["resolution"]["paired_sd"] for b in boards if b["board"] == "SWE-bench Verified")
    T = np.logspace(np.log10(60), np.log10(30000), 200)
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.fill_between(T, 100 * z * min(sds) / np.sqrt(T), 100 * z * max(sds) / np.sqrt(T), color=BLUE, alpha=0.10, lw=0, zorder=1)
    ax.plot(T, 100 * z * sd_v / np.sqrt(T), color=BLUE, lw=2, zorder=2)
    for yv, lab in ((1, "one point"), (2, "two points")):
        ax.axhline(yv, color=ORANGE, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(62, yv * 1.05, f"{lab}", color=ORANGE, fontsize=7.5, va="bottom", ha="left")
        t_need = round((z * sd_v / (yv / 100)) ** 2, -2)
        ax.plot([t_need], [yv], "o", ms=5.5, mfc=SURFACE, mec=ORANGE, mew=1.4, zorder=4)
        ax.annotate(f"about {t_need:,.0f} tasks", (t_need, yv), textcoords="offset points", xytext=(-7, -10), color=INK2, fontsize=7.4,
                    va="center", ha="right", fontweight="bold")
    offs = {"SWE-bench Verified": (9, 7, "left"), "SWE-bench Lite": (9, -9, "left"), "SWE-bench Test": (9, 8, "left"),
            "SWE-bench Multimodal": (9, -9, "left"), "SWE-bench Multilingual": (9, 9, "left"), "Terminal-Bench 2": (9, -10, "left"),
            "LiveBench": (-9, -14, "right")}
    for b in boards:
        x, y = b["tasks"], 100 * b["resolution"]["mdd"]
        agent = b.get("agent", True)                # the non-agent comparison board is drawn hollow
        ax.plot([x], [y], "o", ms=7.5, mfc=BLUE if agent else SURFACE, mec=SURFACE if agent else BLUE, mew=1.6, zorder=5)
        dx, dy, ha = offs.get(b["board"], (8, 8, "left"))
        ax.annotate(b["board"] + ("" if agent else " (language models)"), (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=7.3, color=INK, ha=ha, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(60, 30000)
    ax.set_ylim(0.5, 16)
    ax.set_xticks([100, 300, 1000, 3000, 10000, 30000])
    ax.set_xticklabels(["100", "300", "1,000", "3,000", "10,000", "30,000"])
    ax.set_yticks([0.5, 1, 2, 4, 8, 16])
    ax.set_yticklabels(["0.5", "1", "2", "4", "8", "16"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Number of tasks in the benchmark")
    ax.set_ylabel("Smallest difference detected (points)")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    save(fig, "fig2_resolution")


# ---------------------------------------------------------------- Fig 3: rank intervals
def fig_ranks():
    panels = [("SWE-bench Verified (500 tasks)", load("swebench_verified.json"), 10),
              ("Terminal-Bench 2 (89 tasks)", load("terminalbench.json"), 8)]
    fig = plt.figure(figsize=(6.8, 6.0))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[5, 1.1], height_ratios=[10, 8], hspace=0.5, wspace=0.05)
    for row, (title, b, top) in enumerate(panels):
        ax = fig.add_subplot(gs[row, 0])
        axp = fig.add_subplot(gs[row, 1], sharey=ax)
        rs = b.rank_summary(B=4000, top=top)
        sc = b.scores()
        obs_rank = [1 + int((sc > r["score"] + 1e-12).sum()) for r in rs]
        ymax = len(rs)
        xmax = max(r["rank_hi"] for r in rs)
        for i, (r, orank) in enumerate(zip(rs, obs_rank)):
            y = ymax - i
            if i % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color=WASH, zorder=0, lw=0)
                axp.axhspan(y - 0.5, y + 0.5, color=WASH, zorder=0, lw=0)
            ax.plot([r["rank_lo"], r["rank_hi"]], [y, y], color=BLUE, lw=2, solid_capstyle="round", zorder=2)
            ax.plot([orank], [y], "o", ms=6, mfc=BLUE, mec=SURFACE, mew=1.5, zorder=3)
            axp.barh([y], [r["p_best"]], height=0.46, color=ORANGE, zorder=2)
            if r["p_best"] >= 0.05:
                axp.text(r["p_best"] + 0.03, y, f"{r['p_best']:.2f}", va="center", fontsize=7, color=INK2)
        ax.set_yticks([ymax - i for i in range(len(rs))])
        ax.set_yticklabels([f"{short(r['name'], 38)}  {r['score'] * 100:.1f}%" for r in rs], fontsize=7.3, color=INK2)
        ax.set_xlim(0, xmax + 1)
        ax.set_xticks([1] + [t for t in range(5, int(xmax) + 1, 5)])
        ax.set_xlabel("Rank under task resampling (95% interval)")
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_ylim(0.4, ymax + 0.6)
        ax.set_title(title, loc="left", fontsize=9, color=INK, pad=6)
        axp.set_xlim(0, 1.0)
        axp.set_xticks([0, 0.5, 1.0])
        axp.set_xlabel("Top-rank freq.")
        axp.tick_params(length=0, labelleft=False)
        axp.spines["left"].set_visible(False)
        axp.grid(axis="x", color=GRID, lw=0.5)
        axp.set_axisbelow(True)
    save(fig, "fig3_rank_intervals")


# ---------------------------------------------------------------- Fig 4: winner's curse
def fig_bias():
    path = os.path.join(ROOT, "results", "validation_verified.json")
    if not os.path.exists(path):
        print("no validation results yet; skipping fig4")
        return
    v = json.load(open(path))
    if "V1b" not in v or v.get("R", 0) < 200:
        print("validation file predates the revised design or is a smoke run; skipping fig4")
        return
    groups = [("Known truth", [(f"{n} tasks", v["V1"][n]) for n in ("100", "250", "500") if n in v["V1"]]),
              ("Equal leaders", [(f"{n} tasks", v["V1b"][n]) for n in ("250", "500") if n in v["V1b"]]),
              ("Real held-out data", [("random tasks", v["V2"]), ("other repositories", v["V3"])])]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    y = 0
    ticks, labels, gaps = [], [], []
    allvals = []
    for gname, rows in groups:
        top = y
        for lab, d in rows:
            y -= 1
            nb, cb = d["naive"]["bias_pts"], d["iterated"]["bias_pts"]
            allvals += [nb, cb]
            ax.plot([cb, nb], [y, y], color=AXIS, lw=1.4, zorder=1, solid_capstyle="round")
            ax.plot([nb], [y], "o", ms=7.5, mfc=ORANGE, mec=SURFACE, mew=1.6, zorder=3)
            ax.plot([cb], [y], "o", ms=7.5, mfc=BLUE, mec=SURFACE, mew=1.6, zorder=3)
            fmt = lambda v: "0.0" if abs(v) < 0.05 else f"{v:+.1f}"
            ax.text(nb + 0.14, y, fmt(nb), va="center", fontsize=7.3, color=INK2)
            ax.text(cb - 0.14, y, fmt(cb), va="center", ha="right", fontsize=7.3, color=INK2)
            ticks.append(y); labels.append(lab)
        ax.text(-0.015, top - 0.35, gname, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.8, fontweight="bold", color=INK, clip_on=False)
        y -= 0.55
    ax.axvline(0, color=AXIS, lw=0.9, zorder=0)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, color=INK2, fontsize=7.8)
    ax.tick_params(axis="y", pad=4)
    ax.set_xlabel("Average overstatement of the winner's score (percentage points)")
    lo, hi = min(allvals + [0]) - 0.9, max(allvals) + 1.0
    ax.set_xlim(lo, hi)
    ax.set_ylim(y + 0.2, 0.2)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.plot([], [], "o", ms=6.5, mfc=ORANGE, mec=SURFACE, label="reported score")
    ax.plot([], [], "o", ms=6.5, mfc=BLUE, mec=SURFACE, label="corrected (iterated bootstrap)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=7.8, labelcolor=INK2)
    fig.subplots_adjust(left=0.30)
    save(fig, "fig4_winners_curse")


# ---------------------------------------------------------------- Fig 5: ladders (appendix)
def fig_ladders():
    boards = [("SWE-bench Verified", "swebench_verified.json"), ("SWE-bench Lite", "swebench_lite.json"),
              ("SWE-bench Test", "swebench_test.json"), ("SWE-bench Multimodal", "swebench_multimodal.json"),
              ("Terminal-Bench 2", "terminalbench.json")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), gridspec_kw={"hspace": 0.55, "wspace": 0.32})
    flat = axes.ravel()
    for ax, (label, fname) in zip(flat, boards):
        b = load(fname)
        a = b.record_audit()
        sc = b.scores()
        steps = a["steps"]
        xs = np.arange(1, len(steps) + 1)
        ys = [sc[b.names.index(s["new"])] * 100 for s in steps]
        ax.plot(xs, ys, color=AXIS, lw=1.2, zorder=1)
        for x, y, s in zip(xs, ys, steps):
            ax.plot([x], [y], "o", ms=7, mfc=BLUE if s["significant_raw"] else SURFACE, mec=BLUE if s["significant_raw"] else MUTED, mew=1.2, zorder=3)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_title(f"{label}\n{a['n_significant_raw']} of {a['n_steps']} steps significant", loc="left", fontsize=8.5, color=INK)
        ax.set_xlabel("Record step")
        ax.set_ylabel("Record score (%)")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    ax = flat[-1]
    ax.axis("off")
    ax.plot([], [], "o", ms=7, mfc=BLUE, mec=BLUE, mew=1.2, label="beats previous record (selective exact p ≤ 0.05)")
    ax.plot([], [], "o", ms=7, mfc=SURFACE, mec=MUTED, mew=1.2, label="not distinguishable from it")
    ax.legend(loc="center left", frameon=False, fontsize=8, labelcolor=INK2, handletextpad=0.6)
    save(fig, "fig5_ladders")


# ---------------------------------------------------------------- Fig 6: certified record histories
def fig_certified():
    from honestboard.chain import _submission_order
    path = os.path.join(ROOT, "results", "chain_audit.json")
    if not os.path.exists(path):
        print("no chain audit yet; skipping fig6")
        return
    ca = {e["board"]: e for e in json.load(open(path))["boards"]}
    panels = [("SWE-bench Verified", "swebench_verified.json"), ("SWE-bench Lite", "swebench_lite.json")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.28})
    for ax, (label, fname) in zip(axes, panels):
        b = load(fname)
        order = _submission_order(b)
        sc = 100 * b.scores()[order]
        x = np.arange(1, len(order) + 1)
        record_line = np.maximum.accumulate(sc)
        proc = ca[label]["procedures"].get("tenure_exact_certified") or ca[label]["procedures"]["spending_exact_certified"]
        cert_names = set(proc["certified"])
        cert_line, cur = [], sc[0]
        for i, k in enumerate(order):
            if b.names[k] in cert_names:
                cur = sc[i]
            cert_line.append(cur)
        ax.plot(x, sc, "o", ms=2.6, mfc=MUTED, mec=MUTED, alpha=0.55, zorder=1)
        ax.step(x, record_line, where="post", color=AXIS, lw=1.6, zorder=2, label="leaderboard record")
        ax.step(x, cert_line, where="post", color=BLUE, lw=2.2, zorder=3, label="certified record (tenure rule, Corollary 3)")
        cx = [i + 1 for i, k in enumerate(order) if b.names[k] in cert_names]
        cy = [sc[i] for i, k in enumerate(order) if b.names[k] in cert_names]
        ax.plot(cx, cy, "o", ms=6.5, mfc=BLUE, mec=SURFACE, mew=1.4, zorder=4)
        n_rec = ca[label]["procedures"]["lord_exact_leader"]["n_tests"]
        ax.set_title(f"{label}: {n_rec} leaderboard records, {len(cx)} certified", loc="left", fontsize=8.8, color=INK)
        ax.set_xlabel("Submission (date order)")
        ax.set_ylabel("Score (%)")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axes[0].legend(loc="lower right", frameon=False, fontsize=7.6, labelcolor=INK2)
    save(fig, "fig6_certified")


if __name__ == "__main__":
    which = sys.argv[1:] or ["forest", "resolution", "ranks", "bias", "ladders", "certified"]
    fns = {"forest": fig_forest, "resolution": fig_resolution, "ranks": fig_ranks, "bias": fig_bias, "ladders": fig_ladders, "certified": fig_certified}
    for w in which:
        fns[w]()
