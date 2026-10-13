"""Merge results/validation_<tag>__<stage>.json files into results/validation_<tag>.json.

usage: python merge_validation.py <tag> [<tag> ...]
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

for tag in sys.argv[1:]:
    parts = sorted(glob.glob(os.path.join(RES, f"validation_{tag}__*.json")))
    merged = {}
    r_by_stage = {}
    for p in parts:
        d = json.load(open(p))
        r_by_stage[os.path.basename(p).split("__")[1].split(".")[0]] = d.get("R")
        for k, v in d.items():
            if k == "R":
                continue
            if k == "V1b" and k in merged:
                merged[k].update(v)              # keep n_equal_top from any stage
            elif k in ("V1", "V1b", "V2", "V3", "V4") and not v:
                continue                          # an empty placeholder from a stage that did not run it
            else:
                merged[k] = v
    merged["R_by_stage"] = r_by_stage
    merged["R"] = min(v for v in r_by_stage.values() if v)       # the smallest number of repetitions in any stage
    json.dump(merged, open(os.path.join(RES, f"validation_{tag}.json"), "w"), indent=2)
    print(tag, "merged", [os.path.basename(p) for p in parts], "keys", sorted(merged))
