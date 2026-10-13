"""Fetch the LiveBench per-question judgments (the leaderboard's own scoring file) and summarise them.

Source: Hugging Face dataset livebench/model_judgment, file data/leaderboard-00000-of-00001.parquet (about 0.7 MB).
Columns: question_id, task, model, score (0 to 1), turn, tstamp, category. Writes data/livebench/model_judgment.parquet
and prints a summary used to decide the task set.
"""
import os
import sys

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "livebench")
URL = "https://huggingface.co/datasets/livebench/model_judgment/resolve/main/data/leaderboard-00000-of-00001.parquet"
path = os.path.join(OUT, "model_judgment.parquet")

os.makedirs(OUT, exist_ok=True)
if not os.path.exists(path) or "--refresh" in sys.argv:
    r = requests.get(URL, timeout=120, headers={"User-Agent": "research-pilot"})
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    print("downloaded", len(r.content), "bytes")

df = pd.read_parquet(path)
print(df.shape, "rows; models", df.model.nunique(), "questions", df.question_id.nunique(), "tasks", df.task.nunique(), "categories", df.category.nunique())
print("turns", df.turn.value_counts().to_dict())
print("score values:", np.round(np.quantile(df.score, [0, .1, .25, .5, .75, .9, 1]), 3), " share in {0,1}:", round(float(df.score.isin([0.0, 1.0]).mean()), 3))
print("duplicate (model, question):", int(df.duplicated(["model", "question_id"]).sum()))
per_model = df.groupby("model").agg(n_q=("question_id", "nunique"), first=("tstamp", "min"), last=("tstamp", "max"), score=("score", "mean"))
per_model["first"] = pd.to_datetime(per_model["first"], unit="s").dt.date
per_model["last"] = pd.to_datetime(per_model["last"], unit="s").dt.date
print(per_model.n_q.describe())
print(per_model.sort_values("first").to_string(max_rows=200))
cov = df.groupby("question_id").model.nunique()
print("questions by number of models that have them:", cov.value_counts().sort_index().to_dict())
print(df.groupby(["category", "task"]).question_id.nunique().to_string())
