"""Join each SWE-bench Verified record step to the submission's own write-up, from the leaderboard's metadata.

The public `experiments` repository stores a metadata.yaml per submission with the system's display name, its
organisation, a site link (usually the code repository) and, for some submissions, a report link (a paper or a
blog post). This script joins those fields to the record steps of the Verified board (same ordering and the same
per-date correction as the audit) and writes results/record_claims_verified.json, which make_numbers.py turns
into the worked-certification table of the audit section.

Needs data/raw_verified.json (the raw download, which is not redistributed); the result file it writes is shipped.
"""
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "honestboard", "src"))
from honestboard import Board  # noqa: E402

raw = json.load(open(os.path.join(ROOT, "data", "raw_verified.json"), encoding="utf-8"))
b = Board.from_json(os.path.join(ROOT, "data", "boards", "swebench_verified.json"))
audit = b.record_audit()
scores = {name: float(b.Y[k].mean()) for k, name in enumerate(b.names)}


def text(x):
    if x is None:
        return None
    if isinstance(x, list):
        return ", ".join(str(v) for v in x)
    return str(x)


def report_kind(url):
    """What the submitter's `report` link points to: a paper or technical report, or a blog or news post."""
    if not url:
        return None
    u = url.lower()
    if any(k in u for k in ("arxiv.org", "openreview.net", "aclanthology.org", "doi.org", "proceedings.")) or u.endswith(".pdf"):
        return "paper"
    return "post"


steps = []
for s in audit["steps"]:
    md = yaml.safe_load(raw[s["new"]]["metadata"]) if s["new"] in raw and raw[s["new"]].get("metadata") else {}
    info = (md or {}).get("info") or {}
    tags = (md or {}).get("tags") or {}
    steps.append({
        "new": s["new"], "previous": s["previous"], "date": s["date"],
        "name": text(info.get("name")), "org": text(tags.get("org")), "site": text(info.get("site")), "report": text(info.get("report")),
        "report_kind": report_kind(text(info.get("report"))),
        "checked": bool(tags.get("checked")) if tags.get("checked") is not None else None,
        "score": scores[s["new"]], "previous_score": scores[s["previous"]], "margin": s["margin"],
        "date_group": s.get("date_group", 1), "p": s["p"], "p_uncorrected": s.get("p_uncorrected", s["p"]),
        "significant": bool(s["significant_raw"]), "b_only": s["b_only"], "a_only": s["a_only"],
    })

out = {
    "board": "SWE-bench Verified", "n_steps": len(steps),
    "n_with_report": sum(1 for e in steps if e["report"]),
    "n_with_report_significant": sum(1 for e in steps if e["report"] and e["significant"]),
    "n_report_paper": sum(1 for e in steps if e["report_kind"] == "paper"),
    "n_report_post": sum(1 for e in steps if e["report_kind"] == "post"),
    "n_with_site": sum(1 for e in steps if e["site"]),
    "n_checked": sum(1 for e in steps if e["checked"]),
    "n_checked_significant": sum(1 for e in steps if e["checked"] and e["significant"]),
    "n_significant": sum(1 for e in steps if e["significant"]),
    "steps": steps,
}
with open(os.path.join(ROOT, "results", "record_claims_verified.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(f"{out['n_steps']} record steps; {out['n_with_report']} link a report ({out['n_report_paper']} papers, {out['n_report_post']} posts; {out['n_with_report_significant']} significant); "
      f"{out['n_with_site']} link a site; {out['n_checked']} checked by the maintainers ({out['n_checked_significant']} significant); "
      f"{out['n_significant']} significant in all")
for e in steps:
    print(f"  {e['date']} {e['name'][:48] if e['name'] else e['new']:48s} {100 * e['score']:5.1f}  +{100 * e['margin']:4.1f}  g={e['date_group']}  p={e['p']:.3f}"
          f"  {'report' if e['report'] else ('site' if e['site'] else '--'):6s} {'checked' if e['checked'] else ''}")
