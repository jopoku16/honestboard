# Reproducing every number, table and figure

Everything in `paper/main.pdf` is generated: `src/make_numbers.py` writes `paper/numbers.tex` (the `\Num...` macros the text uses) and every `paper/tab_*.tex` from the JSON files in `results/`, and `src/make_figures.py` writes `figures/`. Nothing numeric is typed into `paper/main.tex`.

## Environment
- Python 3.13.7 with the versions in `requirements.txt` (`pip install -r requirements.txt`); the package itself needs only numpy and scipy (`pip install -e honestboard`).
- LaTeX with `latexmk` (MiKTeX or TeX Live); `pdftoppm`/`pdftotext` only for the page checks.
- Seeds are fixed in every script (`numpy.random.default_rng` with a constant). Results are bit-for-bit repeatable on the same numpy version; across numpy versions Monte Carlo numbers can move in the last digit.
- For speed set `OPENBLAS_NUM_THREADS=1` and run the long validations as parallel single-thread processes (the multi-threaded BLAS makes the small matrix products slower, not faster).

## Data provenance and exclusions
`DATA_SOURCES.md` lists every source, the fetch script, the scoring rules and every exclusion with the names of the excluded submissions. `python src/data_hashes.py` writes `results/data_hashes.json` with the SHA-256 and size of every raw file, built board, result file and generated table, and prints the board and raw-file hashes; compare against the committed file after a refetch. Raw files are not redistributed (licence check pending, see `DATA_SOURCES.md`); the built boards in `data/boards/` are.

## One command
`python run_all.py` rebuilds everything from the cached raw data in about an hour sequentially (`--quick` for a few-minute smoke test with small repetition counts; `--refetch` re-downloads the public results first). The order and the per-step commands are below so that any single table can be rebuilt alone.

## Boards
| Step | Command | Writes |
|---|---|---|
| SWE-bench boards | `python src/fetch_swebench_results.py verified,lite,test,multimodal,multilingual` then `python src/build_boards.py verified,lite,test,multimodal,multilingual` | `data/raw_*.json`, `data/boards/swebench_*.json` |
| Terminal-Bench 2 | `python src/fetch_terminalbench.py` | `data/raw_terminalbench.json`, `data/boards/terminalbench.json`, `results/tb_build.json` |
| LiveBench | `python src/fetch_livebench.py` then `python src/build_livebench_board.py` | `data/livebench/model_judgment.parquet`, `data/boards/livebench.json`, `results/livebench_build.json` |

## Analyses, with the tables and figures each one feeds
| Command (default repetitions) | Result file | Tables / figures / macros |
|---|---|---|
| `python src/audit_all_boards.py` | `results/audit_summary.json` | Tables: audit, sensitivity, lags, non-agent (with chain audit); Figures 2 and 4 (with make_figures); board macros, resolution intervals, adjusted winners |
| `python src/chain_audit.py` | `results/chain_audit.json` | Table: chain audit; Figure 6; certified counts |
| `python src/validate_record_test.py 600` | `results/validation_recordtest.json` | Tables: record-test size, by position, power; simulation design macros |
| `python src/validate_record_test.py 600 data/boards/swebench_lite.json lite` (also `terminalbench.json terminalbench`, `livebench.json livebench`) | `results/validation_recordtest_<tag>.json` | Appendix table: the calibration fitted to other boards |
| `python src/split_half_replication.py 200 4000` | `results/split_half.json` | Table: split-half replication |
| `python src/validate_cluster_effects.py 20000` | `results/validation_cluster.json` | Table: repository dependence |
| `python src/validate_private_tries.py 20000` | `results/validation_tries.json` | Table: private tries |
| `python src/validate_chain_test.py 300 500` | `results/validation_chain.json` | Table: sequential certification simulation (null, staircase, late, variant) |
| `python src/validate_winners_curse.py verified 400 V1`, `... 400 V1b`, `... 500 V23`, `... 600 V4`, then `python src/merge_validation.py verified` | `results/validation_verified.json` | Tables: validation (main and full), lottery; winner's-curse macros |
| `python src/validate_winners_curse.py lite 300 all`, `terminalbench 600 all`, `livebench 400 all` | `results/validation_{lite,terminalbench,livebench}.json` | Tables: Terminal-Bench and LiveBench validation; Lite macros |
| `python src/validate_rank_intervals.py verified 300`, `lite 300` | `results/validation_rankintervals_*.json` | Table: rank and set coverage |
| `python src/tb_variance_decomposition.py` | `results/tb_variance.json` | run-randomness macros |
| `python src/tb_job_selection.py` | `results/tb_job_selection.json` | best-of-runs macros |
| `python src/missing_data_check.py` (needs `data/raw_verified.json`) | `results/missing_data_verified.json` | missing-submission macros |
| `python src/make_figures.py` | `figures/fig1..fig6.{pdf,png}` | all figures |
| `python src/make_numbers.py` | `paper/numbers.tex`, `paper/tab_*.tex` | every macro and table |
| `cd paper && latexmk -pdf main.tex` | `paper/main.pdf` | the manuscript |
| `python src/render_abstract.py` | stdout | the abstract as plain text with the numbers filled in |
| `python src/build_arxiv_bundle.py`, `python src/build_review_packet.py` | `release/*.zip` | arXiv source and review packet |

Runtimes on one core: the audit a few minutes; the record-test calibration about a minute; the chain simulation about a minute; the split-half check two minutes; the cluster and private-tries simulations under a minute each; the winner's-curse validations 10 to 20 minutes per stage.

## Tests
`python -m pytest honestboard/tests` (47 tests, under a minute): exact test counts and the selective p-value's level for even discordant counts, the sequential rules' error rate under a global null, coverage of the confidence set and of the conditional, hybrid and zoom intervals, brute-force checks of the truncation limits and of the sign-flip enumeration, tie handling and input validation.
