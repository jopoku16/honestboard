import json, os, collections
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
raw = json.load(open(os.path.join(D, "raw_verified.json"), encoding="utf-8"))
print("submissions:", len(raw))
keysets = collections.Counter()
types = collections.Counter()
n_res = []
bad = []
for sub, d in sorted(raw.items()):
    if "results" not in d:
        bad.append((sub, [k for k in d])); continue
    try:
        r = json.loads(d["results"])
    except Exception as e:
        bad.append((sub, "json:" + str(e))); continue
    keysets[tuple(sorted(r.keys())) if isinstance(r, dict) else ("<" + type(r).__name__ + ">",)] += 1
    if isinstance(r, dict) and "resolved" in r:
        types[type(r["resolved"]).__name__] += 1
        n_res.append((sub, len(r["resolved"])))
print("missing/bad:", len(bad), bad[:5])
for k, v in keysets.most_common(8):
    print(v, k)
print(types)
print(n_res[:5], n_res[-5:])
first = sorted(raw)[-1]
print(first)
print(raw[first].get("metadata", "")[:600])
