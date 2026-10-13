# Phase B, Statistics Paper 1

**Statistical Inference for AI-Agent Leaderboards: Record Claims, Rank Uncertainty and the Winner's Curse**
Jeffery Opoku (UTRGV) and David Banahene (FIU). Software: `honestboard`.

Status (2026-10-12): manuscript draft (`paper/main.pdf`, 33 pages), analysis, validation, package and figures; everything compiles and the 57 package tests pass. Preprint in preparation. This repository holds the code, the built boards, every result file, the figures and the manuscript source; raw downloads are not redistributed (see `DATA_SOURCES.md`). Live page: https://jopoku16.github.io/honestboard/.

## Headline results
All numbers are generated into `paper/numbers.tex` from `results/*.json`.
- Across six public agent leaderboards there are 74 record steps; 20 (27%) can be told apart from the record they replaced with the selective exact paired test (the sign test conditional on the record event), 13 after Holm correction, 10 of 65 SWE-bench steps with repositories as the sampling unit (the cautious count: with system-by-repository effects of the size the board suggests, the task-level test rejects 10 to 25% of null record steps). The test's size was checked by simulation from a model fitted to the real board and by split-half replication on the real boards.
- The reason is resolution: SWE-bench Verified detects differences of about 4.3 points with 80% power, its median record margin is 1.5, and a one-point gap would need about 9,300 tasks; the 2,294-task SWE-bench Test resolves 9 of its 13 records.
- The top two systems on Verified tie exactly; a confidence set of the best (MCB-style) holds 14 of 135 systems.
- The leader's reported score is overstated by selection (2.2 points with equal leaders at 500 tasks, 2.0 on held-out halves); an iterated bootstrap removes about 84 to 85% of that (Monte Carlo error about 8 points), about half under a shift to other repositories, and overshoots on the 89-task Terminal-Bench.
- Terminal-Bench's repeated trials show run randomness is about 36% of the variance of a single-run score.
- A sequential certification procedure for record chains: a selective p-value given the record event, proved valid for any holder event that is increasing in the holder's outcomes, which covers a running-maximum holder and a holder chosen by an earlier certification (Proposition 1 and Lemma 1, also checked by exhaustive enumeration); alpha-spending and an online fallback rule (a certified record passes its level to the next), both with family-wise error control over the adaptively growing chain proved without independence assumptions (Corollaries 1 and 2); the same guarantee for the procedure in which the leader moves only on certification, with tenure-indexed levels (Corollary 3); LORD++ as a simulation-calibrated alternative. Run in date order the guaranteed rules certify 11 of the 74 agent-board records (fallback certifies all 4 LiveBench records against 2 for spending); on the certified baseline the tenure rule certifies 19 of 153 tested submissions.
- Development selection, where it can be seen: 24 Terminal-Bench submissions ran the benchmark several times; reporting the best run instead of the mean would add a median 1.9 points, and the between-run spread implies about 2 points for picking the best of five variants.
- Split-half replication on the real boards: steps significant on one random half of the tasks beat the old record on the other half in 91% of splits (null reference 48%); non-significant steps 79% (null 74%). A perturbed-variant simulation (half the submissions are variants of earlier ones with the same true score, the case the proof does not cover) keeps every rule at or below its level.
- LiveBench (language models, 318 questions) run through the same pipeline as the non-agent comparison: 4 of 4 records pass the unadjusted test, 1 survives Holm; too few records to support a contrast with the agent boards.
- Baselines for the winner's curse: the hybrid of Andrews, Kitagawa and McCloskey (2024) and the zoom interval of Zrnic and Fithian (2025). The hybrid estimate coincides with the conditional one and its interval is shorter (10.6 vs 15.0 points with equal leaders, coverage 0.97); the zoom interval covers 0.99 and corrects no point estimate; the iterated bootstrap remains the least biased estimate that is defined at a tie.
Limits are stated in the paper: task-sampling noise only, tuning on public tasks not modelled (one form bounded), single runs on SWE-bench, tasks treated as independent outside the sensitivity checks.

## Layout
- `paper/` manuscript (`main.tex`, `main.pdf`, `refs.bib`), generated `numbers.tex` and `tab_*.tex`
- `honestboard/` the Python package (numpy and scipy only): tests, README, licence, CI workflow, example PR-check workflow
- `src/` data fetch and build, audit, validation, figures, numbers, web page, release bundles
- `data/` raw public results and built boards (see `DATA_SOURCES.md`)
- `results/` audit, validation, calibration and landscape outputs
- `figures/` paper figures (PDF and PNG)
- `site/` self-contained web page (`index.html`)
- `docs/` the web page served by GitHub Pages
- Planning notes, review packets, outreach drafts and raw downloads stay out of the public repository.

## Reproduce
```
pip install -r requirements.txt
python run_all.py --quick     # smoke test
python run_all.py             # full run (about an hour in sequence; see the docstring for the parallel recipe)
cd paper && latexmk -pdf main.tex
python -m pytest honestboard/tests
```
Seeds are fixed, so results are repeatable. `REPRODUCE.md` gives the command behind every table and figure, and `python src/data_hashes.py` writes the SHA-256 of every data and result file (`results/data_hashes.json`).
