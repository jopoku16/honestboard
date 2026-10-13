"""Compare the all-pairs (simultaneous) and Hsu-style (mcb) confidence sets on each board, and check inclusion of
each truly best system in the equal-leaders simulation."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "honestboard", "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from honestboard import Board  # noqa: E402
from validate_winners_curse import bunched_model  # noqa: E402

for fname in ("swebench_verified.json", "swebench_lite.json", "swebench_test.json", "swebench_multimodal.json",
              "swebench_multilingual.json", "terminalbench.json"):
    b = Board.from_json(os.path.join(ROOT, "data", "boards", fname))
    s = b.confidence_set_best(B=4000, method="simultaneous")
    m = b.confidence_set_best(B=4000, method="mcb")
    print(f"{fname:30s} K={b.K:3d}  simultaneous set {s['size']:3d} (c={s['critical_value']:.2f})   mcb set {m['size']:3d} (c={m['critical_value']:.2f})")

b0 = Board.from_json(os.path.join(ROOT, "data", "boards", "swebench_verified.json"))
Pm, leaders = bunched_model(b0.Y.astype(float), m=20)
lead = np.array(sorted(leaders))
rng = np.random.default_rng(3)
for n in (250, 500):
    inc, size, any_best = [], [], []
    for _ in range(300):
        idx = rng.integers(0, b0.T, n)
        b = Board(b0.names, rng.binomial(1, Pm[:, idx]).astype(np.uint8))
        cs = b.confidence_set_best(B=400, seed=int(rng.integers(1 << 30)))
        mem = set(cs["members"])
        inc.append(np.mean([b0.names[k] in mem for k in lead]))
        size.append(cs["size"])
        any_best.append(any(b0.names[k] in mem for k in lead))
    print(f"equal-leaders n={n}: mcb set contains each truly best system in {np.mean(inc):.3f} of runs, at least one in {np.mean(any_best):.3f}, median size {np.median(size):.0f}")
