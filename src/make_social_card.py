"""Social/preview card (1200 x 630) from the audit results: release/social_card.png."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "release")
os.makedirs(OUT, exist_ok=True)
a = json.load(open(os.path.join(ROOT, "results", "audit_summary.json")))
tot = a["totals"]
steps, sig = tot["steps"], tot["significant"]
sig_by_board = [(b["board"], b["records"]["steps"], b["records"]["significant"]) for b in a["boards"] if "records" in b]

INK, INK2, MUTED, BLUE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#2a78d6", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"]})
fig = plt.figure(figsize=(12.0, 6.3), dpi=100, facecolor=SURFACE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1200); ax.set_ylim(0, 630); ax.axis("off")
ax.text(60, 560, "HONEST LEADERBOARDS", fontsize=8.5, color=MUTED, fontweight="bold", va="center")
ax.text(60, 455, f"{sig} of {steps}", fontsize=62, color=BLUE, fontweight="bold", va="center")
ax.text(60, 345, "record claims on six agent leaderboards", fontsize=17, color=INK, va="center")
ax.text(60, 305, "can be told apart from the record they replaced.", fontsize=17, color=INK, va="center")
ax.text(60, 245, "The rest cannot be told apart from the last record\nwith the tasks the benchmark has.", fontsize=11.5, color=INK2, va="center", linespacing=1.4)
ax.text(60, 120, "Opoku and Banahene  ·  per-task results from SWE-bench\nand Terminal-Bench  ·  two-sided exact paired test, p ≤ 0.05", fontsize=8.6, color=MUTED, va="center", linespacing=1.5)
# waffle: one dot per record claim, filled when it beats the previous record
cols = 10
x0, y0, dx, dy = 690, 505, 46, 52
k = 0
flags = [True] * sig + [False] * (steps - sig)
for i, f in enumerate(flags):
    r, c = divmod(i, cols)
    x, y = x0 + c * dx, y0 - r * dy
    if f:
        ax.plot([x], [y], "o", ms=17, mfc=BLUE, mec=BLUE, mew=1.5)
    else:
        ax.plot([x], [y], "o", ms=17, mfc=SURFACE, mec=MUTED, mew=1.6)
ax.plot([700], [78], "o", ms=9, mfc=BLUE, mec=BLUE)
ax.text(722, 78, "distinguishable from the record it replaced", fontsize=8.2, color=INK2, va="center")
ax.plot([965], [78], "o", ms=9, mfc=SURFACE, mec=MUTED, mew=1.4)
ax.text(987, 78, "not distinguishable", fontsize=8.2, color=INK2, va="center")
fig.savefig(os.path.join(OUT, "social_card.png"), dpi=100, facecolor=SURFACE)
print("wrote release/social_card.png", steps, sig)

