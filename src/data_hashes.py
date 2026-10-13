"""SHA-256 of every built board, raw data file and result file, with sizes, so that a rebuild can be checked
against the numbers in the paper. Writes results/data_hashes.json and prints a short table."""
import glob
import hashlib
import json
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS = {"boards": "data/boards/*.json", "raw": "data/raw_*.json", "livebench": "data/livebench/*.parquet",
          "results": "results/*.json", "tables": "paper/tab_*.tex", "numbers": "paper/numbers.tex"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


out = {"python": sys.version.split()[0], "platform": platform.platform(), "files": {}}
try:
    import numpy, scipy  # noqa: E401
    out["numpy"], out["scipy"] = numpy.__version__, scipy.__version__
except ImportError:
    pass
for group, pattern in GROUPS.items():
    for p in sorted(glob.glob(os.path.join(ROOT, *pattern.split("/")))):
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        out["files"][rel] = {"group": group, "bytes": os.path.getsize(p), "sha256": sha(p)}
with open(os.path.join(ROOT, "results", "data_hashes.json"), "w") as f:
    json.dump(out, f, indent=2)
for rel, v in out["files"].items():
    if v["group"] in ("boards", "raw", "livebench"):
        print(f"{v['sha256'][:16]}  {v['bytes']:>12,}  {rel}")
print(f"{len(out['files'])} files hashed; python {out['python']}, numpy {out.get('numpy')}, scipy {out.get('scipy')}")
