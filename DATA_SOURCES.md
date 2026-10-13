# Data sources

All data are public per-task leaderboard results. Fetched 2026-10-12.

## SWE-bench submissions (per-task results)
- Source: https://github.com/SWE-bench/experiments, files under `evaluation/<split>/<submission>/results/results.json` (or `per_instance_details.json` for Multilingual) and `metadata.yaml`.
- Fetched with `src/fetch_swebench_results.py` (small text files only; no trajectories).
- Stored raw: `data/raw_verified.json`, `raw_lite.json`, `raw_test.json`, `raw_multimodal.json`, `raw_multilingual.json`. Built into `data/boards/swebench_<split>.json` by `src/build_boards.py`.
- Usable submissions: Verified 135 of 182 found (47 have no per-task file at the expected path), Lite 84, Test 24, Multimodal 12 (of 22 directories), Multilingual 14.
- Official task lists (instance ids only) from the public Hugging Face dataset listings: `princeton-nlp/SWE-bench_Verified`, `princeton-nlp/SWE-bench_Lite`, `princeton-nlp/SWE-bench`, `princeton-nlp/SWE-bench_Multimodal`. Stored as `data/ids_<split>.json` for Verified and Lite. Multilingual task ids come from its per-instance files.
- License: the repository page showed no license text when checked. Do not redistribute the raw files publicly until you have checked; cite the repository.
- Policy note found during research: since 2025-11-18, Verified and Multilingual accept only submissions with an open publication and an academic or established-lab author; Multimodal is open to anyone.

## Terminal-Bench 2 leaderboard
- Source: https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard, `submissions/terminal-bench/2.0/<Agent>__<Model>/<job>/result.json` (job-level files, about 30 KB each).
- Fetched with `src/fetch_terminalbench.py`. Raw: `data/raw_terminalbench.json`; board: `data/boards/terminalbench.json`; build summary: `results/tb_build.json`.
- Scoring (revised 2026-10-12 after review): every trial named in a job file is counted once; trials under reward 1.0 pass, trials under reward 0.0 fail, trials that appear only in the exception lists (agent errors, timeouts) count as failures. Per-task score = passes / trials on that task, with the trial count the files report (pooled multi-job submissions have 4 to 10 trials on some tasks; 60 of the kept submissions have exactly five everywhere). The board stores per-task trial counts (`trials`).
- Task set: the 89 tasks that at least half of the submissions ran; one extra task name in one file is ignored.
- Dropped: 2 submissions whose job files hold no per-task results (Crux__Claude-Opus-4.6, WozCode__Claude-Opus-4.6), 1 submission with a job labelled benchmark version 2.1 (most job files carry no version label; the dataset path is the 2.0 leaderboard), and 5 submissions that lack trials on some tasks. 67 of 75 kept (recorded in `results/tb_build.json` as `dropped_no_results`, `dropped_version_21`, `dropped_missing_tasks`).
- The dataset card states Apache-2.0 and says submissions are closed pending a new process.

## LiveBench (the non-agent comparison board)
- Source: https://huggingface.co/datasets/livebench/model_judgment, file `data/leaderboard-00000-of-00001.parquet` (737 KB; last modified 2025-04-07): the per-question judgments the LiveBench leaderboard is computed from. Columns: question_id, task, model, score (0 to 1), turn, tstamp, category. The file holds three of LiveBench's six categories (coding, instruction following, language; 7 tasks).
- Fetched with `src/fetch_livebench.py` into `data/livebench/` (a folder of its own; nothing is executed from it). Board built by `src/build_livebench_board.py` into `data/boards/livebench.json`; build summary in `results/livebench_build.json`.
- Rules: questions present for at least half of the 195 models in the file (318 of 494); models that have every one of them (94 kept, 101 dropped, most of them partial re-judgings of a few questions); 571 repeated (model, question) judgments averaged; a model's date is the day of its earliest judgment (8 models share 2024-06-07). Clusters: the 6 tasks that remain.
- License: the dataset card shows no license field; the LiveBench code is on GitHub under Apache-2.0 and the paper (arXiv:2406.19314) describes the questions as released. Cite the paper; do not redistribute the parquet file until checked.

## OpenAlex (citation landscape)
- Source: https://api.openalex.org, queried with `src/scan_openalex_landscape.py` on 2026-10-12. 14 of 19 planned sub-topics completed before rate limiting (HTTP 429). See `results/landscape_openalex_2026-10-12.md`.

## Leads not yet used (from the data-discovery pass)
- Multi-SWE-bench (github.com/multi-swe-bench/experiments, Apache-2.0, `python/verified` 39 submissions; check task overlap with SWE-bench Verified before use).
- BFCL v4 per-model score files (function calling, not agentic).
- SWE-bench Pro (only 9 runs on GitHub), tau2-bench (trajectories on S3, per-task rewards inside trajectory JSON), OSWorld-Verified (very large archives), HAL traces (encrypted).
