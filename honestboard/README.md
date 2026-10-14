# honestboard

Honest uncertainty for agent leaderboards. Give it the per-task results of every system on a benchmark and it tells you:

- **Is this SOTA claim real?** An exact paired test of a new record against the record it replaced.
- **What can this benchmark resolve?** The smallest difference it detects with 80% power, and how many tasks a one-point gap would need.
- **How sure is the ranking?** Rank intervals and the probability each system is the best.
- **Who could still be the best?** A 95% confidence set of the best system.
- **What is the winner's score, corrected?** The top score on a leaderboard is biased upward because it was picked for being highest. `honestboard` removes most of that selection bias.
- **How many record claims survive?** An audit of every record step in a leaderboard's history, with a Holm correction.
- **Who is really on the cost-accuracy frontier?** Given a cost per system (or per task), `pareto_frontier` gives the observed Pareto frontier, the systems that cannot be excluded from it at a stated level (a multiple-comparisons-with-the-best construction with each system's rivals restricted to the systems that cost no more and the margin taken jointly, so the set contains the whole true frontier with probability about 1 - alpha; per-system variants are returned beside it), each system's undominated frequency, and the cheapest plausible best (the cheapest system that cannot be told apart from the most accurate). `frontier_curse` gives the selection-adjusted accuracy of every system on the observed frontier; `exact_frontier_set` is a finite-sample (Bonferroni, exact paired tests) version of the set, and `entrant_pvalues` certifies a new frontier entrant against its whole cheaper pool (intersection-union of polyhedral selective p-values conditional on the frontier event).
- **Certify records as they arrive.** `certify_chain` runs a sequential rule along the submission history: the exact sign-test p-value conditional on the record event (with a per-date factor for submissions that share a date), at levels set by alpha-spending or by the online fallback rule (family-wise error over the whole chain, any number of records, no independence assumption), by tenure-indexed spending on the certified baseline, or by LORD++ (online FDR; certified records earn back budget), against the leaderboard holder or against the last certified record.

It needs per-task outcomes (which tasks each system solved), not just the headline score. A checkout of the SWE-bench `experiments` repository loads with one call (`from_swebench_experiments`); any other leaderboard works once you write a board file (format below).

## Quick start

```python
from honestboard import Board

b = Board.from_json("swebench_verified.json")   # board files may carry a "clusters" map (task -> repository)

b.compare("20250928_trae_doubao_seed_code", "20251205_sonar-foundation-agent_claude-opus-4-5")
b.rank_summary(top=10)              # 95% rank intervals and top-rank frequency
b.confidence_set_best()             # systems that cannot be ruled out as best
b.record_audit()                    # every record step: selective exact test (with the per-date factor), Holm-adjusted
b.record_lags()                     # how far back a record is distinguishable
from honestboard import certify_chain
certify_chain(b, rule="lord", baseline="certified")   # sequential certification in date order
b.resolution()                      # minimum detectable difference, tasks needed for 1 and 2 points
b.winner_adjusted()                 # selection-corrected score for the leader (iterated bootstrap)
b.check_claim("some-system-name")   # the main facts about one system in one call
from honestboard import pareto_frontier, frontier_curse
pareto_frontier(b, costs)           # costs: {system: cost per task} or {system: array of per-task costs}
frontier_curse(b, costs)            # selection-adjusted accuracy for the systems on the observed frontier
```

Command line:

```
python -m honestboard report swebench_verified.json
python -m honestboard check swebench_verified.json 20251205_sonar-foundation-agent_claude-opus-4-5
python -m honestboard pr-check experiments/evaluation/verified experiments/evaluation/verified/NEW_SUBMISSION --tasks verified_ids.json
python -m honestboard frontier board.json costs.json --curse   # costs.json: {system: run total} or {system: {task: cost}}
```

`pr-check` reads a checkout of a leaderboard repository and prints a Markdown report on one submission (rank interval, margin over the current record, a paired test of the claim, what the board can resolve). `examples/leaderboard-pr-check.yml` shows how a leaderboard repository can post that report on every pull request.

Board file (JSON): `{"tasks": [...ids...], "systems": [{"name": "...", "date": "20251205", "resolved": [...ids...]}], "clusters": {task: group}}`.
For repeated trials use `"rates": {task: share_passed}` instead of `"resolved"`. A CSV with header `name,date,<task ids>` and 0/1 or rate cells also works. `clusters` is optional; `cluster=True` needs it and raises an error without it.

## Methods

| Question | Method |
|---|---|
| Is B better than A? | Exact McNemar test (0/1 outcomes) or paired sign-flip permutation test (rates): one-sided, two-sided and the selective p-value `p_selective` (the one-sided tail conditional on B having scored higher, the record test) |
| Rank uncertainty | Task bootstrap; tied systems share a rank; top-rank frequency counts ties fractionally. `cluster=True` resamples repositories instead of tasks |
| Confidence set of the best | Paired-bootstrap multiple comparisons with the best over all systems (max studentised difference); coverage is approximate |
| Record audit | Selective exact paired test of each new record against the previous record (`sided="selective"`, default; `"two"` and `"one"` as sensitivities), Holm-adjusted over the steps. `tie_order` handles same-date submissions best-first (default, fewest steps) or worst-first; under best-first the challenger is the best of its date, so its p-value is multiplied by the size of that date group (`date_correction=True`, default), which the guarantees need |
| Resolution | Normal approximation from the paired standard deviation between neighbouring top systems; checked by simulation |
| Winner's score | Iterated plug-in bootstrap bias correction (default). The exact conditional (polyhedral) method is `winner_conditional`; two published baselines are `winner_hybrid` (the hybrid of Andrews, Kitagawa and McCloskey, 2024: conditional, with the truncation set bounded by a simultaneous projection interval, so never unbounded) and `winner_zoom` (the zoom interval of Zrnic and Fithian, 2025: simultaneous over the plausible winners only; corrects the interval, not the estimate) |
| Cost-accuracy frontier | `pareto_frontier`: keep a system unless a cheaper-or-equal system beats it by more than a simultaneous bootstrap margin (MCB with the rival set restricted to cheaper systems and a joint margin); per-task costs are resampled with the outcomes so the rival set varies. The set contains the whole true frontier with probability about 1 - alpha (proved for fixed costs); `clusters=` resamples task groups instead of tasks. `frontier_curse`: conditional and iterated-bootstrap corrections with the pool of cheaper-or-equal systems in place of the whole board |
| Record chains over time | `certify_chain`: a selective p-value given the record event, at levels from an alpha-spending series, the online fallback rule (`rule="fallback"`: a certified record passes its level to the next one) or, for `baseline="certified"` (the leader moves only on certification), the tenure-indexed rule (`rule="tenure"`); all three with family-wise error control over the chain proved under the task-wise null, any dependence. LORD++ (online FDR) is calibrated by simulation. See the paper |

## Limits you should know about

- It measures **task-sampling luck only**. Developers tuning on a public benchmark and run-to-run randomness of agents are outside the model, so the real overstatement of a winner is larger than the correction removes.
- Tasks are treated as independent draws. Tasks from the same repository are correlated; `cluster=True` is a rough sensitivity check (with few clusters it is not reliable and the paired tests still ignore clusters).
- The record test conditions on the record event (the selective p-value); it is exact for a fixed holder under the task-wise null and conservative when the previous record is a running maximum (under a global null it rejects well under 5% of record steps). A record that does not pass is not necessarily worse. It cannot be told apart from the previous record with these tasks. The task-level test treats tasks as independent; where tasks share a repository, `cluster=True` is the cautious count.
- The winner correction removes most of the bias in simulations but not all of it, much less under a shift to other repositories, and its interval is approximate (below nominal at small sample sizes).
- The conditional method is exact under normality but gives an unbounded answer when the winner ties or nearly ties the runner-up.
- The record chain depends on how submissions with the same date are ordered.

## Install and test

```
pip install honestboard   # from PyPI (numpy and scipy only); or, from a checkout of this folder:
pip install -e .
python -m pytest tests
```

MIT licensed. If you use `honestboard`, please cite the paper (see `CITATION.cff`).
