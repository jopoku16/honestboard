"""Build a honestboard board from the LiveBench per-question judgments (data/livebench/model_judgment.parquet).

LiveBench is a model leaderboard, not an agent leaderboard: a fixed prompt per question, objective scoring, credit in
[0, 1] per question. The judgments file (Hugging Face livebench/model_judgment) holds the per-question scores the
leaderboard is computed from, with a timestamp per judgment. Rules, the same as for Terminal-Bench:
  * the question set is the questions present for at least half of the models in the file;
  * a model is kept if it has every question in that set;
  * repeated judgments of one (model, question) pair are averaged;
  * a model's submission date is the day of its earliest judgment in the file.
Clusters: the task a question belongs to. Writes data/boards/livebench.json and results/livebench_build.json.
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_parquet(os.path.join(ROOT, "data", "livebench", "model_judgment.parquet"))
df = df[["question_id", "task", "category", "model", "score", "tstamp"]]
n_models = df.model.nunique()
cov = df.groupby("question_id").model.nunique()
qs = sorted(cov[cov >= 0.5 * n_models].index)
first = df.groupby("model").tstamp.min()
sub = df[df.question_id.isin(qs)]
agg = sub.groupby(["model", "question_id"]).score.agg(["mean", "count"]).reset_index()
have = agg.groupby("model").question_id.nunique()
keep = sorted(have[have == len(qs)].index)
task_of = df.drop_duplicates("question_id").set_index("question_id").task.to_dict()

systems = []
for m in keep:
    rates = agg[agg.model == m].set_index("question_id")["mean"]
    date = pd.to_datetime(first[m], unit="s").strftime("%Y%m%d")
    systems.append({"name": m, "date": date, "rates": {q: float(rates[q]) for q in qs}})
systems.sort(key=lambda s: (s["date"], s["name"]))

board = {"benchmark": "LiveBench (judgments file of 2025-04; coding, instruction following and language questions)",
         "tasks": qs, "systems": systems, "clusters": {q: task_of[q] for q in qs}}
os.makedirs(os.path.join(ROOT, "data", "boards"), exist_ok=True)
json.dump(board, open(os.path.join(ROOT, "data", "boards", "livebench.json"), "w"))

kept_rows = agg[agg.model.isin(keep)]
build = {"n_models_in_file": int(n_models), "n_questions_in_file": int(df.question_id.nunique()),
         "n_questions_kept": len(qs), "n_models_kept": len(keep), "n_models_dropped": int(n_models - len(keep)),
         "repeated_pairs_averaged": int((kept_rows["count"] > 1).sum()),
         "tasks": sorted({task_of[q] for q in qs}), "n_tasks": len({task_of[q] for q in qs}),
         "categories": sorted(sub.category.unique()),
         "share_binary_scores": float(sub[sub.model.isin(keep)].score.isin([0.0, 1.0]).mean()),
         "date_first": systems[0]["date"], "date_last": systems[-1]["date"],
         "n_on_first_date": sum(1 for s in systems if s["date"] == systems[0]["date"]),
         "dropped_models": sorted(set(df.model) - set(keep))}
json.dump(build, open(os.path.join(ROOT, "results", "livebench_build.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in build.items() if k != "dropped_models"}, indent=1))
print("dropped:", len(build["dropped_models"]))
