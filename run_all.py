"""Rebuild everything: boards, audit, validation, figures, paper numbers, web page.

    python run_all.py              analysis from the stored data (validation takes about 30 minutes in sequence)
    python run_all.py --refetch    re-download the public results first
    python run_all.py --quick      validation with 40 repetitions per stage (smoke test, a few minutes)

For speed the validation stages can instead be run in parallel, one thread each:
    set OPENBLAS_NUM_THREADS=1
    python src/validate_winners_curse.py verified 400 V1   (also V1b, V23, V4; and `lite 300 all`, `terminalbench 600 all`)
    python src/merge_validation.py verified
The manuscript is compiled separately: cd paper && latexmk -pdf main.tex ; then python src/build_arxiv_bundle.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
QUICK = "--quick" in sys.argv


def run(script, *args):
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, os.path.join(SRC, script), *args], check=True)


if "--refetch" in sys.argv:
    run("fetch_swebench_results.py", "verified,lite,test,multimodal,multilingual")
    run("fetch_terminalbench.py")
    run("fetch_livebench.py", "--refresh")
run("build_boards.py", "verified,lite,test,multimodal,multilingual")
run("fetch_terminalbench.py")            # builds the board from the cached raw file if present
if not os.path.exists(os.path.join(ROOT, "data", "livebench", "model_judgment.parquet")):
    run("fetch_livebench.py")
run("build_livebench_board.py")          # the non-agent comparison board
run("tb_job_selection.py")               # best-of-runs bound from the Terminal-Bench submissions with several jobs
run("pilot_winners_curse.py")
run("audit_all_boards.py")
for stage in ("V1", "V1b", "V23", "V4"):
    run("validate_winners_curse.py", "verified", "40" if QUICK else "400", stage)
run("merge_validation.py", "verified")
run("validate_winners_curse.py", "lite", "40" if QUICK else "300", "all")
run("validate_winners_curse.py", "terminalbench", "40" if QUICK else "600", "all")
run("validate_winners_curse.py", "livebench", "40" if QUICK else "400", "all")
run("validate_record_test.py", "60" if QUICK else "600")
for tag in ("lite", "terminalbench", "livebench"):          # the calibration fitted to the other boards (appendix)
    fname = {"lite": "swebench_lite.json", "terminalbench": "terminalbench.json", "livebench": "livebench.json"}[tag]
    run("validate_record_test.py", "60" if QUICK else "600", os.path.join(ROOT, "data", "boards", fname), tag)
run("validate_rank_intervals.py", "verified", "40" if QUICK else "300")
run("validate_rank_intervals.py", "lite", "40" if QUICK else "300")
run("tb_variance_decomposition.py")
run("chain_audit.py")
run("validate_chain_test.py", "40" if QUICK else "300", "500")
run("split_half_replication.py", "40" if QUICK else "200", "1000" if QUICK else "4000")
run("validate_cluster_effects.py", "2000" if QUICK else "20000")
run("validate_private_tries.py", "2000" if QUICK else "20000")
if os.path.exists(os.path.join(ROOT, "data", "raw_verified.json")):
    run("missing_data_check.py")              # needs the raw download
run("make_figures.py")
run("make_numbers.py")
run("build_site.py")
run("make_social_card.py")
run("data_hashes.py")
