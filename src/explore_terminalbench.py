"""List the Terminal-Bench 2.0 leaderboard submissions on Hugging Face and show the structure of one job result.

Read-only exploration; prints to stdout.
"""
import json
import requests

DS = "harborframework/terminal-bench-2-leaderboard"
API = f"https://huggingface.co/api/datasets/{DS}/tree/main"
RAW = f"https://huggingface.co/datasets/{DS}/resolve/main/"

r = requests.get(f"{API}/submissions/terminal-bench/2.0", timeout=60)
print(r.status_code)
dirs = [e["path"] for e in r.json() if e["type"] == "directory"]
print(len(dirs), "submission dirs; first 5:", [d.split("/")[-1] for d in dirs[:5]])
d0 = dirs[0]
r = requests.get(f"{API}/{d0}", timeout=60)
ents = r.json()
print("contents of first submission:", [(e["path"].split("/")[-1], e["type"], e.get("size")) for e in ents][:10])
jobs = [e["path"] for e in ents if e["type"] == "directory"]
if jobs:
    r = requests.get(f"{API}/{jobs[0]}", timeout=60)
    ents2 = r.json()
    print("contents of first job:", [(e["path"].split("/")[-1], e["type"], e.get("size")) for e in ents2][:12])
    res = requests.get(RAW + jobs[0] + "/result.json", timeout=60)
    print("result.json status", res.status_code, len(res.text))
    j = res.json()
    def show(o, depth=0, key=""):
        pad = "  " * depth
        if isinstance(o, dict):
            for k, v in list(o.items())[:12]:
                if isinstance(v, (dict, list)):
                    print(f"{pad}{k}: {type(v).__name__}({len(v)})")
                    if depth < 5:
                        show(v, depth + 1, k)
                else:
                    print(f"{pad}{k}: {str(v)[:70]}")
        elif isinstance(o, list) and o:
            print(f"{pad}[0]: {str(o[0])[:90]}")
    show(j)
