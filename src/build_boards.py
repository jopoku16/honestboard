"""Turn the raw SWE-bench submission files into honestboard JSON boards.

Reads : data/raw_<split>.json
Writes: data/boards/swebench_<split>.json  {"benchmark", "tasks", "systems": [{"name","date","resolved"}]}
        data/ids_<split>.json              official task ids (when the dataset listing could be fetched)

Official task ids come from the public Hugging Face dataset listings (instance_id column only).
If they cannot be fetched, tasks nobody solved are padded as anonymous always-failed tasks.
"""
import io
import json
import os
import re
import sys

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
os.makedirs(os.path.join(DATA, "boards"), exist_ok=True)

DATASETS = {"verified": ("princeton-nlp/SWE-bench_Verified", 500), "lite": ("princeton-nlp/SWE-bench_Lite", 300),
            "test": ("princeton-nlp/SWE-bench", 2294), "multimodal": ("princeton-nlp/SWE-bench_Multimodal", None),
            "multilingual": ("SWE-bench/SWE-bench_Multilingual", 300)}
MAX_BYTES = 40_000_000     # refuse to download a listing file larger than this


def official_ids(split):
    path = os.path.join(DATA, f"ids_{split}.json")
    if os.path.exists(path):
        return json.load(open(path))
    ds, n = DATASETS[split]
    try:
        import pyarrow.parquet as pq
        meta = requests.get(f"https://huggingface.co/api/datasets/{ds}/parquet/default/test", timeout=30).json()
        head = requests.head(meta[0], timeout=30, allow_redirects=True)
        size = int(head.headers.get("content-length", 0))
        if size > MAX_BYTES:
            raise RuntimeError(f"listing file is {size / 1e6:.0f} MB, over the limit")
        raw = requests.get(meta[0], timeout=300).content
        table = pq.read_table(io.BytesIO(raw), columns=["instance_id"])
        ids = sorted(table.column("instance_id").to_pylist())
        assert n is None or len(ids) == n, (len(ids), n)
        json.dump(ids, open(path, "w"))
        return ids
    except Exception as e:  # network, format or pyarrow problem
        print(f"[{split}] could not fetch official ids ({type(e).__name__}: {e}); padding instead", file=sys.stderr)
        return None


def build(split):
    raw = json.load(open(os.path.join(DATA, f"raw_{split}.json"), encoding="utf-8"))
    systems = []
    all_ids = set()
    for name, d in sorted(raw.items()):
        resolved = None
        if "results" in d:
            try:
                r = json.loads(d["results"])
                if isinstance(r, dict) and isinstance(r.get("resolved"), list):
                    resolved = set(r["resolved"])
            except Exception:
                pass
        elif "perinst" in d:                 # multilingual-style: {instance_id: {"resolved": bool, ...}}
            try:
                r = json.loads(d["perinst"])
                all_ids |= set(r)
                resolved = {k for k, v in r.items() if isinstance(v, dict) and v.get("resolved")}
            except Exception:
                pass
        if resolved is None:
            continue
        m = re.match(r"(\d{8})_", name)
        systems.append({"name": name, "date": m.group(1) if m else "", "resolved": sorted(resolved)})
    ids = official_ids(split) if split in DATASETS and DATASETS[split][1] is not None or split == "multimodal" else None
    if ids is None and all_ids:
        ids = sorted(all_ids)                # task list recovered from the per-instance files themselves
    seen = sorted(set().union(*[set(s["resolved"]) for s in systems]))
    if ids is None:
        n = DATASETS[split][1] or len(seen)
        ids = seen + [f"__never_solved_{i}-0" for i in range(max(0, n - len(seen)))]
        official = False
    else:
        extra = [t for t in seen if t not in set(ids)]
        if extra:
            print(f"[{split}] {len(extra)} resolved ids are not in the official list; they are ignored", file=sys.stderr)
        official = True
    clusters = {t: t.rsplit("-", 1)[0] for t in ids}          # repository, e.g. django__django
    out = {"benchmark": f"SWE-bench {split}", "official_ids": official, "tasks": ids, "systems": systems, "clusters": clusters}
    json.dump(out, open(os.path.join(DATA, "boards", f"swebench_{split}.json"), "w"))
    print(f"{split}: {len(systems)} systems, {len(ids)} tasks, official_ids={official}")


if __name__ == "__main__":
    for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["verified", "lite"]):
        build(s)
