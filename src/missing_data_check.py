"""What do the Verified submissions without per-task files look like, and could they change the record chain?

Each submission's metadata.yaml reports its aggregate score (info.resolved, in percent). For the submissions that have
per-task results we check that the metadata score matches the computed one. For the others we compare their aggregate
scores and dates with the included submissions, and ask which of them would have been record holders by aggregate score.

Reads data/raw_verified.json (not shipped in the review packet). Writes results/missing_data_verified.json.
"""
import json
import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw = json.load(open(os.path.join(ROOT, "data", "raw_verified.json"), encoding="utf-8"))
board = json.load(open(os.path.join(ROOT, "data", "boards", "swebench_verified.json"), encoding="utf-8"))
T = len(board["tasks"])
included = {s["name"]: len(s["resolved"]) / T for s in board["systems"]}

rows = []
for name, d in sorted(raw.items()):
    meta_score = None
    try:
        m = yaml.safe_load(d.get("metadata", "")) or {}
        meta_score = (m.get("info") or {}).get("resolved")
    except Exception:
        pass
    mm = re.match(r"(\d{8})_", name)
    rows.append({"name": name, "date": mm.group(1) if mm else "", "meta_score": meta_score / 100 if meta_score is not None else None,
                 "included": name in included, "computed": included.get(name)})

have_meta = [r for r in rows if r["meta_score"] is not None]
both = [r for r in rows if r["included"] and r["meta_score"] is not None]
mismatch = [r for r in both if abs(r["meta_score"] - r["computed"]) > 0.0051]
missing = [r for r in rows if not r["included"]]
miss_meta = [r for r in missing if r["meta_score"] is not None]


def chain(items, key):
    items = sorted(items, key=lambda r: (r["date"], -r[key]))
    cur, out = None, []
    for r in items:
        if cur is None or r[key] > cur[key] + 1e-12:
            out.append(r)
            cur = r
    return out


inc_rows = [dict(r, score=r["computed"]) for r in rows if r["included"]]
all_rows = [dict(r, score=r["computed"] if r["included"] else r["meta_score"]) for r in rows if (r["included"] or r["meta_score"] is not None)]
chain_inc = chain(inc_rows, "score")
chain_all = chain(all_rows, "score")
hidden_records = [r for r in chain_all if not r["included"]]
inc_scores = sorted(r["computed"] for r in rows if r["included"])
miss_scores = sorted(r["meta_score"] for r in miss_meta)
med = lambda v: v[len(v) // 2] if v else None
out = {
    "submissions_found": len(rows), "included": len(included), "missing_per_task": len(missing),
    "missing_with_metadata_score": len(miss_meta),
    "metadata_matches_computed": len(both) - len(mismatch), "metadata_checked": len(both),
    "median_score_included": med(inc_scores), "median_score_missing": med(miss_scores),
    "max_score_included": inc_scores[-1] if inc_scores else None, "max_score_missing": miss_scores[-1] if miss_scores else None,
    "record_steps_included_only": len(chain_inc) - 1, "record_steps_all_by_aggregate_score": len(chain_all) - 1,
    "record_holders_missing_per_task": [{"name": r["name"], "date": r["date"], "score": r["score"]} for r in hidden_records],
    "missing_dates": [r["date"] for r in missing][:60],
}
json.dump(out, open(os.path.join(ROOT, "results", "missing_data_verified.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "missing_dates"}, indent=1))
