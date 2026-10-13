"""Build site/index.html: a self-contained page showing each leaderboard with its uncertainty.

Reads data/boards/*.json, runs honestboard, inlines the results into site/template.html.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "honestboard", "src"))
from honestboard import Board  # noqa: E402

SITE = os.path.join(ROOT, "site")
os.makedirs(SITE, exist_ok=True)
cluster_fn = lambda t: t.rsplit("-", 1)[0]

BOARDS = [
    ("verified", "SWE-bench Verified", "swebench_verified.json", True, "500 human-validated tasks from 12 Python repositories"),
    ("lite", "SWE-bench Lite", "swebench_lite.json", True, "300 tasks, a smaller subset of the SWE-bench test set"),
    ("test", "SWE-bench Test", "swebench_test.json", True, "2,294 tasks, the full SWE-bench test set"),
    ("multimodal", "SWE-bench Multimodal", "swebench_multimodal.json", True, "510 tasks that include images"),
    ("multilingual", "SWE-bench Multilingual", "swebench_multilingual.json", True, "300 tasks in several programming languages"),
    ("tbench", "Terminal-Bench 2", "terminalbench.json", False, "89 command-line tasks, five trials each"),
    ("livebench", "LiveBench (language models)", "livebench.json", True, "318 questions with objective per-question credit; the non-agent comparison board"),
]


def short(name):
    s = name.split("_", 1)[1] if name[:8].isdigit() and "_" in name else name
    return s.replace("__", " / ")


out = []
for key, label, fname, uc, blurb in BOARDS:
    b = Board.from_json(os.path.join(ROOT, "data", "boards", fname))
    sc = b.scores()
    rs = b.rank_summary(B=4000, top=12)
    obs = [1 + int((sc > r["score"] + 1e-12).sum()) for r in rs]
    cs = b.confidence_set_best(B=4000)
    res = b.resolution()
    entry = {
        "mdd": res["mdd"], "tasks_for_1pt": res["tasks_for_1pt"],
        "key": key, "label": label, "blurb": blurb, "systems": b.K, "tasks": b.T,
        "ranks": [{"name": short(r["name"]), "score": r["score"], "lo": r["rank_lo"], "hi": r["rank_hi"], "pbest": r["p_best"], "rank": o}
                  for r, o in zip(rs, obs)],
        "set": {"size": cs["size"], "members": [short(m) for m in cs["members"]]},
    }
    if b.dates:
        a = b.record_audit()
        margins = [s["margin"] for s in a["steps"]]
        entry["records"] = {
            "steps": a["n_steps"], "sig": a["n_significant_raw"], "holm": a["n_significant_holm"],
            "median_margin": float(__import__("numpy").median(margins)) if margins else None,
            "list": [{"date": s["date"], "prev": short(s["previous"]), "new": short(s["new"]), "margin": s["margin"], "p": s["p"],
                      "sig": s["significant_raw"], "holm": s["significant_holm"],
                      "score": float(sc[b.names.index(s["new"])])} for s in a["steps"]],
        }
    w = b.winner_adjusted(B=4000)
    wi = b.winner_bootstrap(B=4000, iterate=True, w=w["winner"])
    entry["winner"] = {"name": short(w["name"]), "observed": w["observed"], "adjusted": wi["estimate"], "ci": list(wi["ci"])}
    out.append(entry)
    print(label, "ok")

tmpl = open(os.path.join(SITE, "template.html"), encoding="utf-8").read()
html = tmpl.replace("/*__DATA__*/null", json.dumps(out, separators=(",", ":")))
open(os.path.join(SITE, "index.html"), "w", encoding="utf-8").write(html)
print("wrote site/index.html", len(html) // 1024, "KB")
