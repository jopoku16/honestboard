"""How many systems appear on both SWE-bench Lite and the full Test split, and how many tasks are disjoint?"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "data", "boards")
lite = json.load(open(os.path.join(B, "swebench_lite.json")))
test = json.load(open(os.path.join(B, "swebench_test.json")))
ver = json.load(open(os.path.join(B, "swebench_verified.json")))
ln = {s["name"] for s in lite["systems"]}
tn = {s["name"] for s in test["systems"]}
vn = {s["name"] for s in ver["systems"]}
print("lite systems", len(ln), "test systems", len(tn), "verified systems", len(vn))
print("on both Lite and Test:", len(ln & tn), sorted(ln & tn))
print("on both Verified and Test:", len(vn & tn))
print("on both Lite and Verified:", len(ln & vn))
lt, tt, vt = set(lite["tasks"]), set(test["tasks"]), set(ver["tasks"])
print("tasks: lite", len(lt), "test", len(tt), "verified", len(vt))
print("Lite in Test:", len(lt & tt), "Verified in Test:", len(vt & tt), "Lite & Verified:", len(lt & vt))
print("Test minus Lite:", len(tt - lt), "Test minus (Lite or Verified):", len(tt - lt - vt))
