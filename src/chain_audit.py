"""Run the sequential certification procedures on every board with dates. Writes results/chain_audit.json."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "honestboard", "src"))
from honestboard import Board, certify_chain  # noqa: E402

BOARDS = [("SWE-bench Verified", "swebench_verified.json"), ("SWE-bench Lite", "swebench_lite.json"),
          ("SWE-bench Test", "swebench_test.json"), ("SWE-bench Multimodal", "swebench_multimodal.json"),
          ("SWE-bench Multilingual", "swebench_multilingual.json"), ("Terminal-Bench 2", "terminalbench.json"),
          ("LiveBench", "livebench.json")]
NON_AGENT = {"LiveBench"}                  # reported separately; the totals cover the agent boards only
PROCS = [("unadjusted", "exact", "leader"), ("holm", "exact", "leader"), ("spending", "exact", "leader"), ("fallback", "exact", "leader"),
         ("lord", "exact", "leader"), ("lord", "polyhedral", "leader"), ("spending", "polyhedral", "leader"), ("lord", "exact", "certified"),
         ("spending", "exact", "certified")]

out = {"boards": []}
for label, fname in BOARDS:
    b = Board.from_json(os.path.join(ROOT, "data", "boards", fname))
    entry = {"board": label, "agent": label not in NON_AGENT, "procedures": {}}
    for rule, pv, basel in PROCS:
        if pv == "polyhedral" and not b.is_binary:
            pass                                       # rates boards use the same normal approximation; fine
        res = certify_chain(b, rule=rule, pvalue=pv, baseline=basel)
        key = f"{rule}_{pv}_{basel}"
        entry["procedures"][key] = {"n_tests": res["n_tests"], "n_certified": res["n_certified"], "certified": res["certified"],
                                    "final_certified_holder": res["final_certified_holder"],
                                    "steps": [{k: v for k, v in s.items()} for s in res["steps"]]}
    out["boards"].append(entry)
    line = "  ".join(f"{k}: {v['n_certified']}/{v['n_tests']}" for k, v in entry["procedures"].items())
    print(f"{label:24s} {line}")
tot = {}
for entry in out["boards"]:
    if not entry["agent"]:
        continue
    for k, v in entry["procedures"].items():
        t = tot.setdefault(k, {"tests": 0, "certified": 0})
        t["tests"] += v["n_tests"]; t["certified"] += v["n_certified"]
out["totals"] = tot
print("totals:", {k: f"{v['certified']}/{v['tests']}" for k, v in tot.items()})
with open(os.path.join(ROOT, "results", "chain_audit.json"), "w") as f:
    json.dump(out, f, indent=2, default=float)
