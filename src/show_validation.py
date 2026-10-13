"""Print a validation JSON compactly: python show_validation.py <tag>"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
v = json.load(open(os.path.join(ROOT, "results", f"validation_{sys.argv[1]}.json")))
print("R", v.get("R"), v.get("R_by_stage"), "equal top", v.get("V1b", {}).get("n_equal_top"))


def show(name, d):
    print(name)
    for m, x in d.items():
        if m.startswith("_"):
            continue
        cov = "n/a" if x.get("coverage") is None else f"{x['coverage']:.2f}"
        print(f"   {m:12s} bias {x['bias_pts']:+.2f} med {x['median_err_pts']:+.2f} rmse {x['rmse_pts']:.2f} cov {cov} finite {x['finite']}/{x['n']}")


for n, d in v.get("V1", {}).items():
    show("V1 n=" + n, d)
for n, d in v.get("V1b", {}).items():
    if n != "n_equal_top":
        show("V1b n=" + n, d)
for key in ("V2", "V3"):
    if key in v:
        show(key, v[key])
for n, d in v.get("V4", {}).items():
    print("V4", n, d)
