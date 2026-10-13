"""Loaders for public leaderboard repositories."""
from __future__ import annotations

import json
import os
import re
from typing import Optional, Sequence

from .core import Board


def _swebench_repo(task_id: str) -> str:
    """'django__django-11099' -> 'django__django'"""
    return task_id.rsplit("-", 1)[0]


def _read_resolved(folder: str):
    """Resolved task ids (a set) and the other ids seen in a submission folder, or (None, set())."""
    p1 = os.path.join(folder, "results", "results.json")
    p2 = os.path.join(folder, "per_instance_details.json")
    try:
        if os.path.exists(p1):
            with open(p1, encoding="utf-8") as f:
                r = json.load(f)
            if isinstance(r.get("resolved"), list):
                return set(r["resolved"]), set()
        elif os.path.exists(p2):
            with open(p2, encoding="utf-8") as f:
                r = json.load(f)
            return {k for k, v in r.items() if isinstance(v, dict) and v.get("resolved")}, set(r)
    except (OSError, ValueError, AttributeError):
        pass
    return None, set()


def board_with_submission(evaluation_dir: str, new_dir: str, tasks: Optional[Sequence[str]] = None, cluster_fn=_swebench_repo):
    """The board of an existing leaderboard checkout plus one new submission folder.

    Returns (board, new_index). The new submission is appended last; if its folder name is already on the
    board it is replaced, so the check also works on a checkout that already contains the pull request."""
    name = os.path.basename(os.path.normpath(new_dir))
    new_resolved, new_seen = _read_resolved(new_dir)
    if new_resolved is None:
        raise ValueError(f"no results.json or per_instance_details.json found in {new_dir}")
    systems, seen = [], set(new_resolved) | new_seen
    for sub in sorted(os.listdir(evaluation_dir)):
        if sub == name:
            continue
        resolved, other = _read_resolved(os.path.join(evaluation_dir, sub))
        if resolved is None:
            continue
        seen |= resolved | other
        m = re.match(r"(\d{8})_", sub)
        systems.append({"name": sub, "date": m.group(1) if m else "", "resolved": sorted(resolved)})
    m = re.match(r"(\d{8})_", name)
    systems.append({"name": name, "date": m.group(1) if m else "", "resolved": sorted(new_resolved)})
    task_list = list(tasks) if tasks is not None else sorted(seen)
    return Board.from_resolved(task_list, systems, cluster_fn=cluster_fn), len(systems) - 1


def from_swebench_experiments(evaluation_dir: str, tasks: Optional[Sequence[str]] = None, cluster_fn=_swebench_repo) -> Board:
    """Build a Board from a checkout of github.com/SWE-bench/experiments, for example
    `evaluation_dir = ".../experiments/evaluation/verified"`.

    Each submission folder holds `results/results.json` with a `resolved` list (or a
    `per_instance_details.json` mapping task id -> {"resolved": bool}). If `tasks` (the full task id
    list of the benchmark) is not given, the tasks are the union of ids seen in any file, so tasks that
    no system solved are missing: pass the official id list for exact scores. Tasks are grouped by
    repository (`cluster_fn`) so that `cluster=True` works; pass `cluster_fn=None` to turn that off."""
    systems, seen = [], set()
    for name in sorted(os.listdir(evaluation_dir)):
        folder = os.path.join(evaluation_dir, name)
        resolved = None
        p1 = os.path.join(folder, "results", "results.json")
        p2 = os.path.join(folder, "per_instance_details.json")
        try:
            if os.path.exists(p1):
                with open(p1, encoding="utf-8") as f:
                    r = json.load(f)
                if isinstance(r.get("resolved"), list):
                    resolved = set(r["resolved"])
            elif os.path.exists(p2):
                with open(p2, encoding="utf-8") as f:
                    r = json.load(f)
                seen |= set(r)
                resolved = {k for k, v in r.items() if isinstance(v, dict) and v.get("resolved")}
        except (OSError, ValueError, AttributeError):
            continue
        if resolved is None:
            continue
        seen |= resolved
        m = re.match(r"(\d{8})_", name)
        systems.append({"name": name, "date": m.group(1) if m else "", "resolved": sorted(resolved)})
    task_list = list(tasks) if tasks is not None else sorted(seen)
    return Board.from_resolved(task_list, systems, cluster_fn=cluster_fn)
