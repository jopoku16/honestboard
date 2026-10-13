"""Fetch per-submission results.json + metadata.yaml from the public SWE-bench/experiments repo.

Only small text files are fetched (no trajectories). Output: data/raw_<split>.json
"""
import json, os, sys, time, requests

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(OUT, exist_ok=True)
RAW = "https://raw.githubusercontent.com/SWE-bench/experiments/main/"
UA = {"User-Agent": "honestboard-audit (jeffery.opoku01@utrgv.edu)"}

r = requests.get("https://api.github.com/repos/SWE-bench/experiments/git/trees/main?recursive=1", timeout=60, headers=UA)
r.raise_for_status()
paths = [t["path"] for t in r.json()["tree"] if t["type"] == "blob"]
sel = [p for p in paths if p.startswith("evaluation/") and (p.endswith("/results/results.json") or p.endswith("/metadata.yaml") or p.endswith("/per_instance_details.json"))]

splits = sys.argv[1].split(",") if len(sys.argv) > 1 else ["verified", "lite"]
for split in splits:
    files = [p for p in sel if p.split("/")[1] == split]
    out = {}
    for i, p in enumerate(files):
        sub = p.split("/")[2]
        kind = "results" if p.endswith("results.json") else ("perinst" if p.endswith("per_instance_details.json") else "metadata")
        for attempt in range(3):
            try:
                rr = requests.get(RAW + p, timeout=30, headers=UA)
                if rr.status_code == 200:
                    out.setdefault(sub, {})[kind] = rr.text
                else:
                    out.setdefault(sub, {})[kind + "_err"] = rr.status_code
                break
            except Exception as e:
                time.sleep(2)
        time.sleep(0.05)
        if i % 50 == 0:
            print(split, i, "/", len(files), flush=True)
    with open(os.path.join(OUT, f"raw_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("saved", split, len(out), "submissions", flush=True)
