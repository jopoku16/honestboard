"""Generate paper/numbers.tex and paper/tab_*.tex from the result JSON files, so the manuscript can never
disagree with the analysis. Run after audit_all_boards.py, validate_winners_curse.py, validate_record_test.py,
validate_rank_intervals.py, missing_data_check.py and tb_variance_decomposition.py."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PAP = os.path.join(ROOT, "paper")
os.makedirs(PAP, exist_ok=True)


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


audit = load("audit_summary.json")
val = load("validation_verified.json")
val_lite = load("validation_lite.json")
val_tb = load("validation_terminalbench.json")
rtest = load("validation_recordtest.json")
rank_v = load("validation_rankintervals_verified.json")
rank_l = load("validation_rankintervals_lite.json")
miss = load("missing_data_verified.json")
tbvar = load("tb_variance.json")

NUM = {}


def put(name, value):
    NUM[name] = value


def pct(x, d=0):
    return f"{100 * x:.{d}f}"


def signed(x, d=1):
    """Signed number with a typographic minus for LaTeX; values that round to zero print as 0.0."""
    if abs(x) < 0.5 * 10 ** (-d):
        return f"{0:.{d}f}"
    return f"{x:+.{d}f}".replace("-", "$-$")


def pts(x, d=1):
    return signed(x, d)


def thousands(x):
    return f"{x:,.0f}".replace(",", "{,}")


KEYS = {"Verified": "SWE-bench Verified", "Lite": "SWE-bench Lite", "Test": "SWE-bench Test", "Multimodal": "SWE-bench Multimodal",
        "Multilingual": "SWE-bench Multilingual", "TB": "Terminal-Bench 2", "LB": "LiveBench"}
boards = {b["board"]: b for b in audit["boards"]}
agent_boards = [b for b in audit["boards"] if b.get("agent", True)]        # LiveBench is the non-agent comparison, kept out of the totals

# ---- audit totals (primary: two-sided exact paired test, best-first ordering) ----
tot = audit["totals"]
put("TotalSteps", tot["steps"]); put("TotalSig", tot["significant"]); put("TotalHolm", tot["holm"])
put("TotalSigPct", pct(tot["significant"] / tot["steps"])); put("TotalNotSigPct", pct(1 - tot["significant"] / tot["steps"]))
put("TotalHolmPct", pct(tot["holm"] / tot["steps"]))
t2 = audit.get("totals_two_sided")
put("TotalTwoSided", t2["significant"] if t2 else "--"); put("TotalTwoSidedPct", pct(t2["significant"] / tot["steps"]) if t2 else "--")
t1 = audit.get("totals_one_sided", {"significant": None})
put("TotalOneSided", t1["significant"]); put("TotalOneSidedPct", pct(t1["significant"] / tot["steps"]) if t1["significant"] is not None else "--")
tw = audit["totals_worst_first"]
put("WorstSteps", tw["steps"]); put("WorstSig", tw["significant"]); put("WorstSigPct", pct(tw["significant"] / tw["steps"]))
small = [b for b in agent_boards if b["tasks"] <= 510 and "records" in b]
ss = sum(b["records"]["steps"] for b in small); sg = sum(b["records"]["significant"] for b in small); sh = sum(b["records"]["holm"] for b in small)
put("SmallSteps", ss); put("SmallSig", sg); put("SmallHolm", sh); put("SmallSigPct", pct(sg / ss)); put("SmallHolmPct", pct(sh / ss))
put("SmallBoards", len(small))
big = [b for b in agent_boards if b["tasks"] > 510 and "records" in b]
put("LargeSteps", sum(b["records"]["steps"] for b in big)); put("LargeSig", sum(b["records"]["significant"] for b in big))
put("SmallMarginSteps", sum(b["records"]["small_margin_steps(<2pts)"] for b in agent_boards if "records" in b))
put("SmallMarginSig", sum(b["records"]["small_margin_significant"] for b in agent_boards if "records" in b))
cl_tot = audit.get("cluster_robust_total")
if cl_tot:
    put("ClusterSig", cl_tot["significant"]); put("ClusterSteps", cl_tot["steps"]); put("ClusterSigPct", pct(cl_tot["significant"] / cl_tot["steps"]))
    put("SWEIidSig", sum(b["records"]["significant"] for b in agent_boards if b["board"].startswith("SWE-bench") and "records" in b))
put("BoardsWithSteps", sum(1 for b in agent_boards if "records" in b and b["records"]["steps"] > 0))
_steps = [s for b in agent_boards if "records" in b for s in b["records"]["details"]]
put("TotalDateGroupSteps", sum(1 for s in _steps if s.get("date_group", 1) > 1))
put("TotalSigUncorrected", sum(1 for s in _steps if s.get("p_uncorrected", s["p"]) <= 0.05))
put("TotalDateGroupMax", max((s.get("date_group", 1) for s in _steps), default=1))

# ---- per board ----
for key, label in KEYS.items():
    if label not in boards:
        continue
    b = boards[label]
    K = b["systems"]
    put(f"{key}K", K); put(f"{key}T", thousands(b["tasks"]))
    put(f"{key}Leader", pct(b["leader_score"], 1)); put(f"{key}PBest", f"{b['leader_p_best']:.2f}")
    put(f"{key}LeaderTied", b.get("leader_tied", 1))
    put(f"{key}Set", b["confidence_set_size_95"]); put(f"{key}SetSim", b.get("confidence_set_size_95_simultaneous", "--"))
    put(f"{key}NoiseRecords", f"{sum(1.0 / k for k in range(1, K + 1)) - 1:.1f}")
    w = b["winner"]
    put(f"{key}WinnerObs", pct(w["observed"], 1)); put(f"{key}WinnerAdj", pct(w["adjusted_iterated"], 1))
    put(f"{key}WinnerDrop", f"{100 * (w['observed'] - w['adjusted_iterated']):.1f}")
    ce = w.get("conditional_estimate")
    put(f"{key}CondAdj", pct(ce, 1) if (ce is not None and np.isfinite(ce)) else "undefined")
    put(f"{key}CoLeaders", len(w.get("co_leaders", [])))
    co = [r for r in b.get("top3", []) if r["name"] in w.get("co_leaders", [])]
    others = sorted((r["p_best"] for r in co if r["name"] != b["leader"]), reverse=True)
    put(f"{key}PBestOther", f"{others[0]:.2f}" if others else "--")
    if "records" in b:
        r = b["records"]
        put(f"{key}Steps", r["steps"]); put(f"{key}Sig", r["significant"]); put(f"{key}Holm", r["holm"])
        put(f"{key}TwoSided", r.get("two_sided", {}).get("significant", "--")); put(f"{key}OneSided", r.get("one_sided", {}).get("significant", "--"))
        put(f"{key}MedianMargin", f"{100 * r['median_margin']:.1f}" if r["median_margin"] is not None else "--")
        put(f"{key}MedianLag", f"{r['median_lag']:.0f}" if r["median_lag"] is not None else "--")
        put(f"{key}ShareAdjacent", pct(r["share_adjacent"]) if r["share_adjacent"] is not None else "--")
        put(f"{key}ShareNever", pct(r["share_never"]) if r["share_never"] is not None else "--")
        ub = r.get("median_upper975_nonsig")
        put(f"{key}UpperNonSig", f"{100 * ub:.1f}" if ub is not None else "--")
        put(f"{key}WorstSteps", r["worst_first"]["steps"]); put(f"{key}WorstSig", r["worst_first"]["significant"])
        if r.get("cluster_robust"):
            put(f"{key}ClusterSig", r["cluster_robust"]["significant"])
    rs = b["resolution"]
    put(f"{key}CritMargin", f"{100 * 1.95996 * rs['paired_sd'] / np.sqrt(b['tasks']):.1f}")    # smallest margin the two-sided 5% test calls significant for a typical pair
    put(f"{key}MDD", f"{100 * rs['mdd']:.1f}"); put(f"{key}TasksOnePt", thousands(round(rs["tasks_for_1pt"], -2)))
    put(f"{key}TasksTwoPt", thousands(round(rs["tasks_for_2pt"], -2)))
    if "mdd_ci95" in rs:
        put(f"{key}MDDLo", f"{100 * rs['mdd_ci95'][0]:.1f}"); put(f"{key}MDDHi", f"{100 * rs['mdd_ci95'][1]:.1f}")
        put(f"{key}TasksOnePtLo", thousands(round(rs["tasks_for_1pt_ci95"][0], -2))); put(f"{key}TasksOnePtHi", thousands(round(rs["tasks_for_1pt_ci95"][1], -2)))
    if w.get("ci_iterated"):
        put(f"{key}WinnerCiLo", pct(w["ci_iterated"][0], 1)); put(f"{key}WinnerCiHi", pct(w["ci_iterated"][1], 1))
    if "resolution_cluster" in b:
        put(f"{key}MDDCluster", f"{100 * b['resolution_cluster']['mdd']:.1f}")
    if "confidence_set_size_95_cluster" in b:
        put(f"{key}SetCluster", b["confidence_set_size_95_cluster"]); put(f"{key}PBestCluster", f"{b['leader_p_best_cluster']:.2f}")

# ---- validation macros ----
METHODS = (("naive", "Naive"), ("bootstrap", "Boot"), ("iterated", "Iter"), ("conditional", "Cond"), ("hybrid", "Hyb"), ("zoom", "Zoom"))
METHOD_NAMES = (("naive", "reported"), ("bootstrap", "bootstrap"), ("iterated", "iterated bootstrap"), ("conditional", "conditional"),
                ("hybrid", "hybrid (AKM)"), ("zoom", "zoom interval (ZF)"))
NTAG = {"100": "NA", "250": "NB", "500": "NC"}


def put_cell(prefix, d):
    for m, mt in METHODS:
        x = d.get(m)
        if x is None:
            continue
        put(f"{prefix}{mt}Bias", f"{x['bias_pts']:.1f}" if m == "naive" else pts(x["bias_pts"]))
        put(f"{prefix}{mt}Se", f"{x['se_bias_pts']:.2f}" if x.get("se_bias_pts") is not None else "--")
        put(f"{prefix}{mt}Med", pts(x["median_err_pts"])); put(f"{prefix}{mt}Rmse", f"{x['rmse_pts']:.1f}")
        put(f"{prefix}{mt}Cov", f"{x['coverage']:.2f}" if x.get("coverage") is not None else "--")
        put(f"{prefix}{mt}Len", f"{x['median_len_pts']:.1f}" if x.get("median_len_pts") is not None else "--")
        put(f"{prefix}{mt}Fin", pct(x["finite"] / x["n"]))
        put(f"{prefix}{mt}BiasC", pts(x["bias_common_pts"]) if x.get("bias_common_pts") is not None else "--")
        put(f"{prefix}{mt}MedC", pts(x["median_common_pts"]) if x.get("median_common_pts") is not None else "--")
    nb, ib, pb = d["naive"]["bias_pts"], d["iterated"]["bias_pts"], d["bootstrap"]["bias_pts"]
    put(f"{prefix}Reduction", pct(1 - ib / nb) if nb else "--"); put(f"{prefix}ReductionPlain", pct(1 - pb / nb) if nb else "--")
    # the reduction carries Monte Carlo error: delta method on the two means (treated as independent, conservative)
    se_n, se_i = d["naive"].get("se_bias_pts"), d["iterated"].get("se_bias_pts")
    if nb and se_n is not None and se_i is not None:
        se_red = np.sqrt((se_i / nb) ** 2 + (ib * se_n / nb ** 2) ** 2)
        put(f"{prefix}ReductionSe", pct(se_red))
    put(f"{prefix}NCommon", d.get("_n_common", "--"))


if val:
    for n, tag in NTAG.items():
        if n in val.get("V1", {}):
            put_cell(f"VOne{tag}", val["V1"][n])
    for n, tag in NTAG.items():
        if n in val.get("V1b", {}):
            put_cell(f"VBunch{tag}", val["V1b"][n])
    if "V2" in val:
        put_cell("VTwo", val["V2"])
    for k, tag in (("V3", "VThree"), ("V3_raw", "VThreeRaw"), ("V3_iid", "VThreeIid")):
        if k in val:
            put_cell(tag, val[k])
    put("BunchEqualTop", val["V1b"]["n_equal_top"])
    put("ValR", val["R"])
    if "V3_split" in val:
        put("VThreeSelMin", val["V3_split"]["min"]); put("VThreeSelMax", val["V3_split"]["max"]); put("VThreeSplits", val["V3_split"]["n_candidate_splits"])
    for n, tag in NTAG.items():
        if n in val.get("V4", {}):
            d = val["V4"][n]
            put(f"VFour{tag}Win", f"{d['p_winner_is_true_best']:.2f}")
            put(f"VFour{tag}Cover", f"{d['set_covers_true_best_when_winner_wrong']:.2f}" if d["set_covers_true_best_when_winner_wrong"] is not None else "--")
            put(f"VFour{tag}Size", f"{d['median_set_size']:.0f}")
if val_tb:
    n = str(val_tb["T"])
    put_cell("TBOne", val_tb["V1"][n]); put_cell("TBBunch", val_tb["V1b"][n]); put_cell("TBTwo", val_tb["V2"])
if val_lite:
    put_cell("LiteTwo", val_lite["V2"])
    if "V3" in val_lite:
        put_cell("LiteThree", val_lite["V3"])
    put_cell("LiteOneNB", val_lite["V1"]["250"]); put_cell("LiteBunchNB", val_lite["V1b"]["250"])
val_lb = load("validation_livebench.json")
if val_lb:
    n = str(val_lb["T"])
    put_cell("LBOne", val_lb["V1"][n]); put_cell("LBBunch", val_lb["V1b"][n]); put_cell("LBTwo", val_lb["V2"])
    put("LBValR", val_lb["R"])

# ---- rank intervals, missing data, Terminal-Bench variance ----
if rank_v:
    for n, tag in (("250", "A"), ("500", "B")):
        d = rank_v["by_n"][n]
        put(f"RankCovAll{tag}", f"{d['coverage_all_systems']:.2f}"); put(f"RankCovTop{tag}", f"{d['coverage_true_top10']:.2f}")
        put(f"RankCovWin{tag}", f"{d['coverage_selected_winner']:.2f}")
        b = rank_v["bunched"][n]
        put(f"RankBunchLead{tag}", f"{b['coverage_equal_leaders']:.2f}"); put(f"RankBunchAll{tag}", f"{b['coverage_all_systems']:.2f}")
        put(f"RankBunchLeadAvg{tag}", f"{b.get('coverage_equal_leaders_avg', float('nan')):.2f}")
        put(f"RankBunchBlock{tag}", f"{b.get('coverage_equal_leaders_block', float('nan')):.2f}")
        put(f"SetBunchIn{tag}", f"{b['confidence_set_contains_each_equal_leader']:.2f}"); put(f"SetBunchSize{tag}", f"{b['median_set_size']:.0f}")
    put("RankCovR", rank_v["R"])
if rank_l:
    d = rank_l["by_n"]["300"]; b = rank_l["bunched"]["300"]
    put("RankCovLiteAll", f"{d['coverage_all_systems']:.2f}"); put("RankBunchLiteLead", f"{b['coverage_equal_leaders']:.2f}")
if miss:
    put("MissingN", miss["missing_per_task"]); put("MissingFound", miss["submissions_found"])
    put("MissingMedian", pct(miss["median_score_missing"], 1)); put("IncludedMedian", pct(miss["median_score_included"], 1))
    put("MissingMax", pct(miss["max_score_missing"], 1)); put("IncludedMax", pct(miss["max_score_included"], 1))
    put("MissingHidden", len(miss["record_holders_missing_per_task"]))
    dts = sorted(d for d in miss["missing_dates"] if d)
    put("MissingFrom", f"{dts[0][:4]}-{dts[0][4:6]}"); put("MissingTo", f"{dts[-1][:4]}-{dts[-1][4:6]}")
tbb = load("tb_build.json")
if tbb:
    put("TBFound", tbb["n_submissions_found"]); put("TBKept", tbb["kept"]); put("TBDroppedVersion", len(tbb["dropped_version_21"]))
    put("TBDroppedNoResults", len(tbb.get("dropped_no_results", [])))
    put("TBDroppedMissing", len(tbb["dropped_missing_tasks"])); put("TBTrialsMin", tbb["trials_per_task_min"]); put("TBTrialsMax", tbb["trials_per_task_max"])
    put("TBExactlyFive", tbb["n_exactly_five_everywhere"]); put("TBPooled", tbb["n_multi_job"]); put("TBTasks", tbb["n_tasks"])
if tbvar:
    put("TBRunShareSingle", pct(tbvar["run_share_single_median"])); put("TBRunShareSingleMin", pct(tbvar["run_share_single_min"]))
    put("TBRunShareSingleMax", pct(tbvar["run_share_single_max"])); put("TBRunShareFive", pct(tbvar["run_share_five_median"]))
    put("TBPairRunShareSingle", pct(tbvar["paired_run_share_single_median"])); put("TBPairRunShareFive", pct(tbvar["paired_run_share_five_median"]))

# ---- development selection seen through repeated full runs (Terminal-Bench jobs) ----
tbjob = load("tb_job_selection.json")
if tbjob:
    put("TBJobSubs", tbjob["n_submissions_with_full_jobs"]); put("TBJobMedianJobs", f"{tbjob['jobs_per_submission_median']:.0f}")
    put("TBJobInflMedian", f"{tbjob['best_minus_mean_median_pts']:.1f}"); put("TBJobInflMean", f"{tbjob['best_minus_mean_mean_pts']:.1f}")
    put("TBJobInflMax", f"{tbjob['best_minus_mean_max_pts']:.1f}"); put("TBJobSd", f"{tbjob['between_job_sd_median_pts']:.1f}")
    put("TBJobSdLo", f"{tbjob['between_job_sd_iqr_pts'][0]:.1f}"); put("TBJobSdHi", f"{tbjob['between_job_sd_iqr_pts'][1]:.1f}")
    for v, word in ((2, "Two"), (3, "Three"), (5, "Five"), (10, "Ten")):          # macro names cannot hold digits
        put(f"TBJobInflV{word}", f"{tbjob['expected_inflation_pts_by_V'][str(v)]:.1f}")

# ---- the non-agent comparison board (LiveBench) ----
lbb = load("livebench_build.json")
if lbb:
    put("LBModelsFile", lbb["n_models_in_file"]); put("LBQuestionsFile", lbb["n_questions_in_file"])
    put("LBQuestionsKept", lbb["n_questions_kept"]); put("LBModelsKept", lbb["n_models_kept"]); put("LBModelsDropped", lbb["n_models_dropped"])
    put("LBRepeated", lbb["repeated_pairs_averaged"]); put("LBNTasks", lbb["n_tasks"]); put("LBShareBinary", pct(lbb["share_binary_scores"]))
    put("LBFirstDate", f"{lbb['date_first'][:4]}-{lbb['date_first'][4:6]}"); put("LBLastDate", f"{lbb['date_last'][:4]}-{lbb['date_last'][4:6]}")
    put("LBOnFirstDate", lbb["n_on_first_date"])

# ---- sequential certification ----
chain_sim = load("validation_chain.json")
chain_audit = load("chain_audit.json")
RULE_TAG = {"unadjusted": "Unadj", "holm": "Holm", "spending": "Spend", "fallback": "Fall", "tenure": "Ten", "lord": "Lord"}
PV_TAG = {"exact": "", "polyhedral": "Poly"}
BASE_TAG = {"leader": "", "certified": "Cert"}


def proc_tag(key):
    rule, pv, basel = key.split("_")
    return RULE_TAG[rule] + PV_TAG[pv] + BASE_TAG[basel]


if chain_sim:
    SC = {"null": "Null", "staircase": "Stair", "late": "Late", "variant": "Variant"}
    for scen, procs in chain_sim["scenarios"].items():
        for key, v in procs.items():
            tg = SC[scen] + proc_tag(key)
            put(f"Chain{tg}Fwer", pct(v["fwer"])); put(f"Chain{tg}Fdr", pct(v["fdr"]))
            put(f"Chain{tg}Power", pct(v["power"]) if v["power"] is not None else "--")
            put(f"Chain{tg}Tests", f"{v['mean_tests']:.1f}"); put(f"Chain{tg}Cert", f"{v['mean_certified']:.2f}")
    dz = chain_sim["design"]
    put("ChainR", dz["R"]); put("ChainT", dz["T"]); put("ChainK", dz["K"]); put("ChainStep", dz["step_points"]); put("ChainPeriod", dz["period"])
if chain_audit:
    for key, v in chain_audit["totals"].items():
        put(f"CertTotal{proc_tag(key)}", v["certified"]); put(f"CertTests{proc_tag(key)}", v["tests"])
    CB = {"SWE-bench Verified": "Verified", "SWE-bench Lite": "Lite", "SWE-bench Test": "Test", "SWE-bench Multimodal": "Multimodal",
          "SWE-bench Multilingual": "Multilingual", "Terminal-Bench 2": "TB", "LiveBench": "LB"}
    for entry in chain_audit["boards"]:
        kb = CB[entry["board"]]
        for key, v in entry["procedures"].items():
            put(f"{kb}Cert{proc_tag(key)}", v["n_certified"]); put(f"{kb}CertTests{proc_tag(key)}", v["n_tests"])

# ---- record-test calibration ----
if rtest:
    tagof = {300: "A", 500: "B", 2000: "C"}
    for blk in rtest["null_global"]:
        t = tagof[blk["T"]]
        put(f"NullOne{t}", f"{100 * blk['per_step_rejection_one_sided']:.1f}"); put(f"NullTwo{t}", f"{100 * blk['per_step_rejection_two_sided']:.1f}")
        put(f"NullSelective{t}", f"{100 * blk['per_step_rejection_selective']:.1f}" if "per_step_rejection_selective" in blk else "--")
        put(f"NullHolm{t}", f"{100 * blk['holm_any_rejection_per_board']:.1f}"); put(f"NullSteps{t}", f"{blk['mean_steps_per_board']:.1f}")
        for pos, ptag in (("1", "First"), ("2", "Second"), ("3+", "Later")):
            bp = blk.get("by_position", {}).get(pos)
            if bp and bp["steps"]:
                put(f"NullPos{ptag}One{t}", f"{100 * bp['one'] / bp['steps']:.1f}"); put(f"NullPos{ptag}Two{t}", f"{100 * bp['two'] / bp['steps']:.1f}")
                put(f"NullPos{ptag}Sel{t}", f"{100 * bp['sel'] / bp['steps']:.1f}" if "sel" in bp else "--")
    for blk in rtest["null_selected"]:
        t = tagof[blk["T"]]
        put(f"NullSelOne{t}", f"{100 * blk['rejection_one_sided']:.1f}"); put(f"NullSelTwo{t}", f"{100 * blk['rejection_two_sided']:.1f}")
        put(f"NullSelSel{t}", f"{100 * blk['rejection_selective']:.1f}" if "rejection_selective" in blk else "--")
    for blk in rtest["power"]:
        t = tagof[blk["T"]]
        put(f"FormulaMdd{t}", f"{100 * blk['formula_mdd']:.1f}"); put(f"PowerAtMdd{t}", f"{blk['power_at_formula_mdd']:.2f}")
        put(f"PowerAtMddSel{t}", f"{blk['power_at_formula_mdd_selective']:.2f}" if "power_at_formula_mdd_selective" in blk else "--")

# ---- the record-test calibration repeated with the model fitted to other boards (appendix) ----
others = [(tag, lab, load(f"validation_recordtest_{tag}.json")) for tag, lab in
          (("lite", "SWE-bench Lite"), ("terminalbench", "Terminal-Bench 2"), ("livebench", "LiveBench"))]
others = [(t, l, d) for t, l, d in others if d]
if rtest and others:                 # the other boards' rows go into the merged record-test table below
    for tag, lab, d in others:
        key = {"lite": "Lite", "terminalbench": "TB", "livebench": "LB"}[tag]
        s = {blk["T"]: blk for blk in d["null_selected"]}
        put(f"RtestRho{key}", f"{d['design']['rho']:.2f}")
        put(f"RtestSelSel{key}", f"{100 * s[500]['rejection_selective']:.1f}" if 500 in s and "rejection_selective" in s[500] else "--")
        put(f"RtestSelOne{key}", f"{100 * s[500]['rejection_one_sided']:.1f}" if 500 in s else "--")

# ---- private tries: the submission is the best of V variants on the same tasks ----
tries = load("validation_tries.json")
if tries:
    VT = {1: "One", 2: "Two", 3: "Three", 5: "Five", 10: "Ten"}
    gl = {g["V"]: g for g in tries["global"]}
    for s in tries["selected"]:
        tg = VT[s["V"]]
        put(f"TriesSize{tg}", f"{100 * s['size_selective']:.1f}"); put(f"TriesCorr{tg}", f"{100 * s['size_corrected']:.1f}")
        put(f"TriesRec{tg}", pct(s["record_share"]))
        if s["V"] in gl:
            put(f"TriesFwer{tg}", pct(gl[s["V"]]["fwer_selective"], 1)); put(f"TriesFwerCorr{tg}", pct(gl[s["V"]]["fwer_corrected"], 1))
    with open(os.path.join(PAP, "tab_tries.tex"), "w") as f:
        f.write("\\begin{tabular}{rrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{Selected-step null} & \\multicolumn{2}{c}{Global null, chain of 40} \\\\\n"
                "$V$ & sets a record & size, selective & size, $V$-corrected & FWER, selective & FWER, $V$-corrected \\\\\n\\midrule\n")
        for s in tries["selected"]:
            g = gl.get(s["V"], {})
            f.write(f"{s['V']} & {pct(s['record_share'])}\\% & {100 * s['size_selective']:.1f}\\% & {100 * s['size_corrected']:.1f}\\% & "
                    f"{pct(g.get('fwer_selective', float('nan')), 1)}\\% & {pct(g.get('fwer_corrected', float('nan')), 1)}\\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

# ---- repository dependence: size of the record test with system-by-repository random effects ----
clu = load("validation_cluster.json")
if clu:
    put("ClusterTauPts", f"{clu['design']['tau_est_pts']:.1f}"); put("ClusterTauLogit", f"{clu['design']['tau_est_logit']:.2f}")
    put("ClusterG", clu["design"]["G"]); put("ClusterPairs", thousands(clu["design"]["pairs"]))
    put("ClusterRealSe", f"{clu['design']['real_se_ratio_top10']:.2f}")
    MT = {0.0: "Zero", 0.25: "Quarter", 0.5: "Half", 1.0: "One"}
    for lv in clu["levels"]:
        tg = MT[lv["tau_logit"]]
        put(f"ClusterTask{tg}", f"{100 * lv['reject_task_selective']:.1f}"); put(f"ClusterClu{tg}", f"{100 * lv['reject_cluster_selective']:.1f}")
        put(f"ClusterSe{tg}", f"{lv['se_ratio_cluster_over_task']:.2f}")
    with open(os.path.join(PAP, "tab_cluster_sim.tex"), "w") as f:
        f.write("\\begin{tabular}{rrrr}\n\\toprule\n$\\tau$ (logit) & Task-level selective test & Repository-level test & SE ratio, repository / task \\\\\n\\midrule\n")
        for lv in clu["levels"]:
            f.write(f"{lv['tau_logit']:.2f} & {100 * lv['reject_task_selective']:.1f}\\% & {100 * lv['reject_cluster_selective']:.1f}\\% & {lv['se_ratio_cluster_over_task']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    dz = rtest["design"]
    put("SimRho", f"{dz['rho']:.2f}"); put("SimSdLogit", f"{dz['difficulty_sd_logit']:.2f}"); put("SimBase", pct(dz["base_score"]))
    put("SimSdReal", f"{dz['paired_sd_real']:.2f}"); put("SimSdSim", f"{dz['paired_sd_sim']:.2f}"); put("SimK", dz["K_null"])
    put("NullR", rtest["null_global"][0]["R"]); put("NullSelPairs", thousands(rtest["null_selected"][0]["pairs"]))

# (numbers.tex is written at the end of the script, after the table section, which also defines macros)


# ---- tables ----
def row_audit(b):
    r = b.get("records")
    sig = f"{r['significant']}/{r['steps']}" if r else "--"
    holm = f"{r['holm']}" if r else "--"
    w = b["winner"]
    cr = r.get("cluster_robust") if r else None
    clu = f"{cr['significant']}/{cr['steps']}" if cr else "--"
    return (f"{b['board']} & {b['systems']} & {b['tasks']:,} & {b['leader_p_best']:.2f} & {b['confidence_set_size_95']} & "
            f"{sig} & {holm} & {clu} & {100 * b['resolution']['mdd']:.1f} & {pct(w['observed'], 1)} $\\to$ {pct(w['adjusted_iterated'], 1)} \\\\")


with open(os.path.join(PAP, "tab_audit.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
            "Board & Systems & Tasks & Top-rank freq. & 95\\% set & Records sig. & Holm & Repository-level & MDD (pts) & Leader (\\%): reported $\\to$ adjusted \\\\\n\\midrule\n")
    for b in agent_boards:
        f.write(row_audit(b) + "\n")
    f.write(f"\\midrule\nAll boards & & & & & {tot['significant']}/{tot['steps']} & {tot['holm']} & {cl_tot['significant'] if cl_tot else '--'}/{cl_tot['steps'] if cl_tot else '--'} & & \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

with open(os.path.join(PAP, "tab_sensitivity.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\nBoard & Steps & Selective (primary) & Two-sided & One-sided & Repository-level & Worst-first ordering \\\\\n\\midrule\n")
    for b in agent_boards:
        r = b.get("records")
        if not r:
            continue
        cr = r.get("cluster_robust")
        f.write(f"{b['board']} & {r['steps']} & {r['significant']} & {r.get('two_sided', {}).get('significant', '--')} & {r.get('one_sided', {}).get('significant', '--')} & "
                f"{cr['significant'] if cr else '--'} & {r['worst_first']['significant']}/{r['worst_first']['steps']} \\\\\n")
    f.write(f"\\midrule\nAll boards & {tot['steps']} & {tot['significant']} & {t2['significant'] if t2 else '--'} & {t1['significant']} & "
            f"{cl_tot['significant'] if cl_tot else '--'} (of {cl_tot['steps'] if cl_tot else '--'}) & {tw['significant']}/{tw['steps']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

with open(os.path.join(PAP, "tab_lags.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\nBoard & Median margin (pts) & Median lag & Beats previous record & Beats no earlier record \\\\\n\\midrule\n")
    for b in agent_boards:
        r = b.get("records")
        if not r or not r["steps"]:
            continue
        adj = f"{pct(r['share_adjacent'])}\\%" if r["share_adjacent"] is not None else "--"
        nev = f"{pct(r['share_never'])}\\%" if r["share_never"] is not None else "--"
        lag = f"{r['median_lag']:.0f}" if r["median_lag"] is not None else "--"
        f.write(f"{b['board']} & {100 * r['median_margin']:.1f} & {lag} & {adj} & {nev} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")


# ---- the non-agent comparison: LiveBench beside the largest agent board and the smallest ----
def nonagent_table():
    cols = [("LiveBench", "LiveBench"), ("SWE-bench Verified", "SWE-bench Verified"), ("Terminal-Bench 2", "Terminal-Bench 2")]
    if any(lab not in boards for lab, _ in cols):
        return
    ca = {e["board"]: e["procedures"] for e in chain_audit["boards"]} if chain_audit else {}

    def cell(fn):
        return " & ".join(fn(boards[lab]) for lab, _ in cols)

    def rec(b, k, fmt=str):
        r = b.get("records")
        return fmt(r[k]) if r and r.get(k) is not None else "--"

    rows = [
        ("Type of system", " & ".join(("language model" if lab == "LiveBench" else "agent") for lab, _ in cols)),
        ("Per-task score", " & ".join(("credit in $[0,1]$" if lab == "LiveBench" else ("mean of 1 to 10 trials" if lab == "Terminal-Bench 2" else "pass or fail")) for lab, _ in cols)),
        ("Systems", cell(lambda b: str(b["systems"]))), ("Tasks", cell(lambda b: f"{b['tasks']:,}")),
        ("Record steps", cell(lambda b: rec(b, "steps"))),
        ("Distinguishable (two-sided exact)", cell(lambda b: rec(b, "significant"))),
        ("After Holm", cell(lambda b: rec(b, "holm"))),
        ("Certified, LORD++", " & ".join(str(ca[lab]["lord_exact_leader"]["n_certified"]) if lab in ca else "--" for lab, _ in cols)),
        ("Certified, alpha-spending", " & ".join(str(ca[lab]["spending_exact_leader"]["n_certified"]) if lab in ca else "--" for lab, _ in cols)),
        ("Certified, online fallback", " & ".join(str(ca[lab]["fallback_exact_leader"]["n_certified"]) if lab in ca and "fallback_exact_leader" in ca[lab] else "--" for lab, _ in cols)),
        ("Median record margin (pts)", cell(lambda b: rec(b, "median_margin", lambda v: f"{100 * v:.1f}"))),
        ("Minimum detectable difference (pts)", cell(lambda b: f"{100 * b['resolution']['mdd']:.1f}")),
        ("Tasks to resolve one point", cell(lambda b: thousands(round(b["resolution"]["tasks_for_1pt"], -2)))),
        ("Leader's top-rank frequency", cell(lambda b: f"{b['leader_p_best']:.2f}")),
        ("Systems tied at the top", cell(lambda b: str(b.get("leader_tied", 1)))),
        ("95\\% confidence set of the best", cell(lambda b: str(b["confidence_set_size_95"]))),
        ("Leader's score, reported $\\to$ corrected (\\%)", cell(lambda b: f"{pct(b['winner']['observed'], 1)} $\\to$ {pct(b['winner']['adjusted_iterated'], 1)}")),
    ]
    with open(os.path.join(PAP, "tab_nonagent.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrr}\n\\toprule\n & " + " & ".join(name for _, name in cols) + " \\\\\n\\midrule\n")
        for lab, val_ in rows:
            f.write(f"{lab} & {val_} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def val_rows(d, label):
    cm = d.get("_coverage_meaningful", True)
    out = []
    for m, name in METHOD_NAMES:
        x = d.get(m)
        if x is None:
            continue
        cov = f"{x['coverage']:.2f}" if (cm and x.get("coverage") is not None) else "--"
        se = f" ({x['se_bias_pts']:.2f})" if x.get("se_bias_pts") is not None else ""
        bc = signed(x["bias_common_pts"], 2) if x.get("bias_common_pts") is not None else "--"
        mc = signed(x["median_common_pts"], 2) if x.get("median_common_pts") is not None else "--"
        ln = f"{x['median_len_pts']:.1f}" if x.get("median_len_pts") is not None else "--"
        out.append(f"{label if m == 'naive' else ''} & {name} & {signed(x['bias_pts'], 2)}{se} & {signed(x['median_err_pts'], 2)} & "
                   f"{x['rmse_pts']:.2f} & {cov} & {ln} & {pct(x['finite'] / x['n'])}\\% & {bc} & {mc} \\\\")
    return out


def val_table(path, settings):
    with open(path, "w") as f:
        f.write("\\begin{tabular}{llrrrrrrrr}\n\\toprule\nSetting & Estimator & Mean error (SE) & Median error & RMSE & Coverage & Length & Defined & Mean, common & Median, common \\\\\n\\midrule\n")
        for i, (lab, d) in enumerate(settings):
            f.write("\n".join(val_rows(d, lab)) + "\n")
            if i < len(settings) - 1:
                f.write("\\addlinespace\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


if val:
    main_rows = []
    if "500" in val.get("V1", {}):
        main_rows.append(("Known truth, 500 tasks", val["V1"]["500"]))
    if "500" in val.get("V1b", {}):
        main_rows.append(("Equal leaders, 500 tasks", val["V1b"]["500"]))
    if "V2" in val:
        main_rows.append(("Held-out tasks", val["V2"]))
    if "V3" in val:
        main_rows.append(("Held-out repositories (cluster)", val["V3"]))
    val_table(os.path.join(PAP, "tab_validation.tex"), main_rows)
    full = [(f"Known truth, {n} tasks", val["V1"][n]) for n in ("100", "250", "500") if n in val.get("V1", {})]
    full += [(f"Equal leaders, {n} tasks", val["V1b"][n]) for n in ("250", "500") if n in val.get("V1b", {})]
    full += [(lab, val[k]) for k, lab in (("V2", "Held-out tasks"), ("V3", "Held-out repositories (cluster, shift-adjusted)"),
                                          ("V3_raw", "Held-out repositories (cluster, unadjusted)"), ("V3_iid", "Held-out repositories (iid, shift-adjusted)")) if k in val]
    val_table(os.path.join(PAP, "tab_validation_full.tex"), full)
    small_rows = []                  # Terminal-Bench and LiveBench share one table in the supplement
    if val_tb:
        n = str(val_tb["T"])
        small_rows += [(f"Terminal-Bench 2: known truth, {n} tasks", val_tb["V1"][n]), (f"Terminal-Bench 2: equal leaders, {n} tasks", val_tb["V1b"][n]),
                       ("Terminal-Bench 2: held-out tasks", val_tb["V2"])]
    if val_lb:
        n = str(val_lb["T"])
        small_rows += [(f"LiveBench: known truth, {n} questions", val_lb["V1"][n]), (f"LiveBench: equal leaders, {n} questions", val_lb["V1b"][n]),
                       ("LiveBench: held-out questions", val_lb["V2"])]
    if small_rows:
        val_table(os.path.join(PAP, "tab_validation_small.tex"), small_rows)
    with open(os.path.join(PAP, "tab_lottery.tex"), "w") as f:
        f.write("\\begin{tabular}{rrrr}\n\\toprule\nTasks per benchmark & Winner is a truly best system (share of runs) & Set covers a truly best system when the winner is not & Median set size \\\\\n\\midrule\n")
        for n in ("100", "250", "500"):
            if n in val.get("V4", {}):
                d = val["V4"][n]
                c = f"{d['set_covers_true_best_when_winner_wrong']:.2f}" if d["set_covers_true_best_when_winner_wrong"] is not None else "--"
                f.write(f"{n} & {d['p_winner_is_true_best']:.2f} & {c} & {d['median_set_size']:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

if rank_v:
    with open(os.path.join(PAP, "tab_rankcov.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{Known truth: interval covers true rank} & \\multicolumn{4}{c}{Equal leaders: interval covers} & \\multicolumn{2}{c}{Equal leaders: confidence set} \\\\\n"
                "Tasks & all & true top 10 & selected winner & rank 1 (min) & average rank & tied block & all & contains each leader & median size \\\\\n\\midrule\n")
        for n in ("250", "500"):
            d, b = rank_v["by_n"][n], rank_v["bunched"][n]
            f.write(f"{n} & {d['coverage_all_systems']:.2f} & {d['coverage_true_top10']:.2f} & {d['coverage_selected_winner']:.2f} & "
                    f"{b['coverage_equal_leaders']:.2f} & {b.get('coverage_equal_leaders_avg', float('nan')):.2f} & {b.get('coverage_equal_leaders_block', float('nan')):.2f} & "
                    f"{b['coverage_all_systems']:.2f} & {b['confidence_set_contains_each_equal_leader']:.2f} & {b['median_set_size']:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

if chain_sim:
    LAB = {"unadjusted_exact_leader": "unadjusted, exact", "holm_exact_leader": "Holm over the realised chain", "spending_exact_leader": "alpha-spending, exact",
           "fallback_exact_leader": "online fallback, exact", "spending_exact_certified": "alpha-spending, exact, certified baseline",
           "tenure_exact_certified": "tenure-indexed spending, exact, certified baseline",
           "lord_exact_leader": "LORD++, exact", "spending_polyhedral_leader": "alpha-spending, polyhedral", "lord_polyhedral_leader": "LORD++, polyhedral",
           "lord_exact_certified": "LORD++, exact, certified baseline"}
    sc_ = chain_sim["scenarios"]
    has_var = "variant" in sc_
    with open(os.path.join(PAP, "tab_chain_sim.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrrr" + ("rr" if has_var else "") + "}\n\\toprule\n & \\multicolumn{2}{c}{Global null} & \\multicolumn{3}{c}{Staircase} & \\multicolumn{3}{c}{One late jump}"
                + (" & \\multicolumn{2}{c}{Variants, global null}" if has_var else "") + " \\\\\n"
                "Procedure & FWER & certified & FDR & power & certified & FDR & power & certified" + (" & FWER & certified" if has_var else "") + " \\\\\n\\midrule\n")
        for key in sc_["null"]:
            n, s, l = sc_["null"][key], sc_["staircase"][key], sc_["late"][key]
            pw = lambda v: f"{100 * v['power']:.0f}\\%" if v["power"] is not None else "--"
            f.write(f"{LAB.get(key, key)} & {100 * n['fwer']:.1f}\\% & {n['mean_certified']:.2f} & {100 * s['fdr']:.1f}\\% & {pw(s)} & {s['mean_certified']:.1f} & "
                    f"{100 * l['fdr']:.1f}\\% & {pw(l)} & {l['mean_certified']:.2f}"
                    + (f" & {100 * sc_['variant'][key]['fwer']:.1f}\\% & {sc_['variant'][key]['mean_certified']:.2f}" if has_var else "") + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    if has_var:
        put("ChainVariantShare", pct(chain_sim["design"]["variant_share"])); put("ChainVariantFlip", pct(chain_sim["design"]["variant_flip_pass_to_fail"]))

# ---- split-half replication of the record test on the real boards ----
split = load("split_half.json")
if split:
    p = split["pooled"]
    put("SplitR", split["R"])
    put("SplitSigRep", f"{p['real']['sig1']['rep']:.2f}"); put("SplitSigRepSig", f"{p['real']['sig1']['repsig']:.2f}")
    put("SplitNonRep", f"{p['real']['nonsig1']['rep']:.2f}"); put("SplitNonRepSig", f"{p['real']['nonsig1']['repsig']:.2f}")
    put("SplitNullSigRep", f"{p['null']['sig1']['rep']:.2f}"); put("SplitNullNonRep", f"{p['null']['nonsig1']['rep']:.2f}")
    put("SplitNullSigRepSig", f"{p['null']['sig1']['repsig']:.2f}"); put("SplitNullNonRepSig", f"{p['null']['nonsig1']['repsig']:.2f}")
    put("SplitFullSigRep", f"{p['full_sig']['rep_any_half']:.2f}"); put("SplitFullNonRep", f"{p['full_nonsig']['rep_any_half']:.2f}")
    put("SplitSigShare", pct(p["real"]["sig1"]["n"] / (p["real"]["sig1"]["n"] + p["real"]["nonsig1"]["n"])))
    put("SplitNullSigShare", pct(p["null"]["sig1"]["n"] / (p["null"]["sig1"]["n"] + p["null"]["nonsig1"]["n"])))
    with open(os.path.join(PAP, "tab_split_half.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n & & \\multicolumn{3}{c}{Half 1 significant} & \\multicolumn{3}{c}{Half 1 not significant} \\\\\n"
                "Board & Steps & share of splits & replicates & null reference & share of splits & replicates & null reference \\\\\n\\midrule\n")
        fmt = lambda x: "--" if x is None else f"{x:.2f}"
        for b in split["boards"] + [{"board": "All boards", "n_steps": sum(x["n_steps"] for x in split["boards"]), **split["pooled"]}]:
            r, nl = b["real"], b["null"]
            tot = r["sig1"]["n"] + r["nonsig1"]["n"]
            f.write(f"{b['board']} & {b['n_steps']} & {pct(r['sig1']['n'] / tot) if tot else '--'}\\% & {fmt(r['sig1']['rep'])} & {fmt(nl['sig1']['rep'])} & "
                    f"{pct(r['nonsig1']['n'] / tot) if tot else '--'}\\% & {fmt(r['nonsig1']['rep'])} & {fmt(nl['nonsig1']['rep'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
if chain_audit:
    cols = ["unadjusted_exact_leader", "holm_exact_leader", "spending_exact_leader", "fallback_exact_leader", "lord_exact_leader", "lord_polyhedral_leader",
            "spending_exact_certified", "tenure_exact_certified"]
    cols = [c for c in cols if c in chain_audit["totals"]]
    HEAD = {"unadjusted_exact_leader": "Unadjusted", "holm_exact_leader": "Holm", "spending_exact_leader": "Spending", "fallback_exact_leader": "Fallback",
            "lord_exact_leader": "LORD++", "lord_polyhedral_leader": "LORD++ (polyhedral)", "spending_exact_certified": "Spending, certified baseline",
            "tenure_exact_certified": "Tenure, certified baseline"}
    with open(os.path.join(PAP, "tab_chain_audit.tex"), "w") as f:
        f.write("\\begin{tabular}{lr" + "r" * len(cols) + "}\n\\toprule\nBoard & Records & " + " & ".join(HEAD[c] for c in cols) + " \\\\\n\\midrule\n")
        for entry in chain_audit["boards"]:
            if not entry.get("agent", True):
                continue
            pr = entry["procedures"]
            f.write(f"{entry['board']} & {pr['lord_exact_leader']['n_tests']} & " + " & ".join(
                f"{pr[c]['n_certified']}" + (f"/{pr[c]['n_tests']}" if c.endswith("certified") else "") for c in cols) + " \\\\\n")
        tot = chain_audit["totals"]
        f.write("\\midrule\nAll boards & " + f"{tot['lord_exact_leader']['tests']} & " + " & ".join(
            f"{tot[c]['certified']}" + (f"/{tot[c]['tests']}" if c.endswith("certified") else "") for c in cols) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
# ---- the worked certification of SWE-bench Verified: every record step with its write-up, p-value, fallback level and verdict ----
claims = load("record_claims_verified.json")
if chain_audit and claims:
    ver = next((x for x in chain_audit["boards"] if x["board"] == "SWE-bench Verified"), None)
    fb = {st["new"]: st for st in ver["procedures"]["fallback_exact_leader"]["steps"]} if ver and "fallback_exact_leader" in ver["procedures"] else {}
    put("ClaimsSteps", claims["n_steps"]); put("ClaimsReport", claims["n_with_report"]); put("ClaimsReportSig", claims["n_with_report_significant"])
    put("ClaimsChecked", claims["n_checked"]); put("ClaimsCheckedSig", claims["n_checked_significant"]); put("ClaimsSig", claims["n_significant"])
    put("ClaimsFallbackCert", sum(1 for st in fb.values() if st["certified"]))
    put("ClaimsReportNotSig", claims["n_with_report"] - claims["n_with_report_significant"])
    put("ClaimsSigNotCert", claims["n_significant"] - sum(1 for st in fb.values() if st["certified"]))
    put("ClaimsFirstLevel", f"{100 * min((st['level'] for st in fb.values()), default=float('nan')):.1f}" if fb else "--")

    def short(name, n=44):
        name = (name or "").replace("&", "\\&").replace("_", "\\_").replace("%", "\\%").replace("#", "\\#")
        return name if len(name) <= n else name[: n - 1].rstrip() + "\\ldots"

    def pfmt(p):
        return "$<$0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.9995 else "1.000")

    with open(os.path.join(PAP, "tab_verified_chain.tex"), "w") as f:
        f.write("\\begin{tabular}{rllrrrrrll}\n\\toprule\n\\# & Date & System, as named on the leaderboard & Score & Margin & $g$ & $p_{\\mathrm{sel}}$ & Level & Certified & Write-up \\\\\n\\midrule\n")
        for i, e in enumerate(claims["steps"], 1):
            st = fb.get(e["new"], {})
            if "level" in st:
                lvl = f"{st['level']:.4f}" if st["level"] >= 0.001 else ("$%s\\times10^{%d}$" % (f"{st['level']:.1e}".split("e")[0], int(f"{st['level']:.1e}".split("e")[1])))
            else:
                lvl = "--"
            cert = "yes" if st.get("certified") else "no"
            wu = "report" if e["report"] else ("site" if e["site"] else "--")
            d = e["date"]
            f.write(f"{i} & {d[:4]}-{d[4:6]}-{d[6:]} & {short(e['name'] or e['new'])} & {100 * e['score']:.1f} & {100 * e['margin']:+.1f} & {e['date_group']} & "
                    f"{pfmt(e['p'])} & {lvl} & {cert} & {wu} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

# ---- run-split replication of the record test on Terminal-Bench (trials split into two parts) ----
runsplit = load("tb_run_split.json")
if runsplit:
    p = runsplit["pooled"]
    fmt2 = lambda x: "--" if x is None else f"{x:.2f}"
    put("RunSplitR", runsplit["R"]); put("RunSplitSteps", runsplit["n_steps_used"]); put("RunSplitStepsAll", runsplit["n_steps"])
    put("RunSplitNullDraws", runsplit["null_draws"])
    put("RunSplitSigShare", pct(p["real"]["sig1"]["n"] / (p["real"]["sig1"]["n"] + p["real"]["nonsig1"]["n"])))
    put("RunSplitNullSigShare", pct(p["null"]["sig1"]["n"] / (p["null"]["sig1"]["n"] + p["null"]["nonsig1"]["n"])))
    put("RunSplitSigRep", fmt2(p["real"]["sig1"]["rep"])); put("RunSplitSigRepSig", fmt2(p["real"]["sig1"]["repsig"]))
    put("RunSplitNonRep", fmt2(p["real"]["nonsig1"]["rep"])); put("RunSplitNonRepSig", fmt2(p["real"]["nonsig1"]["repsig"]))
    put("RunSplitNullSigRep", fmt2(p["null"]["sig1"]["rep"])); put("RunSplitNullNonRep", fmt2(p["null"]["nonsig1"]["rep"]))
    put("RunSplitFullSigRep", fmt2(p["full_sig"]["rep_any_half"])); put("RunSplitFullNonRep", fmt2(p["full_nonsig"]["rep_any_half"]))
    put("RunSplitFullSigSteps", p["full_sig"]["steps"]); put("RunSplitFullNonSteps", p["full_nonsig"]["steps"])

if rtest:
    # one table for the record-test calibration: the model fitted to Verified (all three tests) and to the other boards
    pc = lambda x: f"{100 * x:.1f}\\%" if x is not None else "--"
    fitted = [("SWE-bench Verified", rtest)] + [(lab, d) for _, lab, d in others]
    with open(os.path.join(PAP, "tab_recordtest.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrrrrr}\n\\toprule\n & & & \\multicolumn{3}{c}{Global null (holder is the best so far)} & \\multicolumn{3}{c}{Selected-step null (holder not inflated)} & & \\\\\n"
                "Fitted board & Tasks & Steps per board & selective & two-sided & one-sided & selective & two-sided & one-sided & Holm, any (selective) & Power at MDD \\\\\n\\midrule\n")
        for bi, (lab, d) in enumerate(fitted):
            sel = {blk["T"]: blk for blk in d["null_selected"]}
            pw = {blk["T"]: blk for blk in d["power"]}
            for k, blk in enumerate(d["null_global"]):
                s, pw_ = sel[blk["T"]], pw[blk["T"]]
                f.write(f"{lab if k == 0 else ''} & {blk['T']:,} & {blk['mean_steps_per_board']:.1f} & {pc(blk.get('per_step_rejection_selective'))} & {pc(blk['per_step_rejection_two_sided'])} & "
                        f"{pc(blk['per_step_rejection_one_sided'])} & {pc(s.get('rejection_selective'))} & {pc(s['rejection_two_sided'])} & {pc(s['rejection_one_sided'])} & "
                        f"{pc(blk['holm_any_rejection_per_board'])} & {pw_.get('power_at_formula_mdd_selective', pw_['power_at_formula_mdd']):.2f} \\\\\n")
            if bi < len(fitted) - 1:
                f.write("\\addlinespace\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(PAP, "tab_position.tex"), "w") as f:
        f.write("\\begin{tabular}{rlrrrr}\n\\toprule\nTasks & Step in the chain & Steps & selective & two-sided & one-sided \\\\\n\\midrule\n")
        for blk in rtest["null_global"]:
            for pos, lab in (("1", "first (holder is the first submission)"), ("2", "second"), ("3+", "third and later")):
                bp = blk.get("by_position", {}).get(pos)
                if bp and bp["steps"]:
                    sv = pc(bp["sel"] / bp["steps"]) if "sel" in bp else "--"
                    f.write(f"{blk['T']:,} & {lab} & {bp['steps']:,} & {sv} & {pc(bp['two'] / bp['steps'])} & {pc(bp['one'] / bp['steps'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(PAP, "tab_power.tex"), "w") as f:
        f.write("\\begin{tabular}{rrrrrr}\n\\toprule\nTasks & 1 pt & 2 pts & 3 pts & 5 pts & Power at the formula's MDD \\\\\n\\midrule\n")
        for blk in rtest["power"]:
            p = blk["power"]
            key_p = "power_selective" if all("power_selective" in v for v in p.values()) else "power_two_sided"
            cells = " & ".join(f"{p[k][key_p]:.2f}" for k in sorted(p, key=float))
            at = blk.get("power_at_formula_mdd_selective", blk["power_at_formula_mdd"])
            f.write(f"{blk['T']:,} & {cells} & {at:.2f} (MDD {100 * blk['formula_mdd']:.1f} pts) \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
nonagent_table()
with open(os.path.join(PAP, "numbers.tex"), "w") as f:
    f.write("% generated by src/make_numbers.py; do not edit\n")
    for k, v in NUM.items():
        f.write(f"\\newcommand{{\\Num{k}}}{{{v}}}\n")
print("wrote", len(NUM), "macros and tables to paper/")
