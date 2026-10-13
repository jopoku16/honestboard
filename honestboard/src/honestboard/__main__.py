"""Command line: python -m honestboard report BOARD.json | check BOARD.json SYSTEM"""
import argparse
import sys

from .core import Board


def _load(path):
    return Board.from_csv(path) if path.endswith(".csv") else Board.from_json(path)


def _pct(x):
    return f"{100 * x:.1f}%"


def cmd_report(args):
    b = _load(args.board)
    print(f"{b.K} systems, {b.T} tasks\n")
    rs = b.rank_summary(top=args.top, cluster=args.cluster)
    print("Leaders with 95% rank intervals and top-rank frequency:")
    for r in rs:
        print(f"  {r['name'][:52]:52s} {_pct(r['score']):>6s}  rank [{r['rank_lo']:.0f}, {r['rank_hi']:.0f}]  top-rank freq.={r['p_best']:.2f}")
    cs = b.confidence_set_best(cluster=args.cluster)
    print(f"\nConfidence set of the best (95%): {cs['size']} of {b.K} systems cannot be ruled out as best.")
    if b.dates:
        a = b.record_audit()
        print(f"\nRecord claims: {a['n_steps']} steps, {a['n_significant_raw']} beat the previous record at the 5% level "
              f"(selective exact paired test, conditional on the record event), {a['n_significant_holm']} after Holm correction.")
        r = b.resolution()
        print(f"Resolution: this board detects differences of about {100 * r['mdd']:.1f} points (80% power); "
              f"resolving a 1-point gap would take about {r['tasks_for_1pt']:,.0f} tasks.")
    w = b.winner_adjusted(cluster=args.cluster)
    print(f"\nWinner {w['name']}: reported {_pct(w['observed'])}, selection-adjusted estimate {_pct(w['estimate'])}, "
          f"95% interval [{_pct(w['ci'][0])}, {_pct(w['ci'][1])}] (naive [{_pct(w['naive_ci'][0])}, {_pct(w['naive_ci'][1])}]).")
    c = w["conditional"]
    print(f"  conditional method: estimate {_pct(c['estimate'])}, interval [{_pct(c['ci'][0])}, {_pct(c['ci'][1])}] "
          f"(unbounded when the winner's margin is about zero)")
    h = b.winner_hybrid(cluster=args.cluster, w=w["winner"])
    z = b.winner_zoom(cluster=args.cluster, w=w["winner"])
    print(f"  hybrid (Andrews, Kitagawa and McCloskey): estimate {_pct(h['estimate'])}, interval [{_pct(h['ci'][0])}, {_pct(h['ci'][1])}]"
          + (" (undefined at an exact tie for first)" if h["degenerate"] else ""))
    print(f"  zoom interval (Zrnic and Fithian): [{_pct(z['ci'][0])}, {_pct(z['ci'][1])}] over {z['n_plausible']} plausible winners")


def cmd_check(args):
    b = _load(args.board)
    r = b.check_claim(args.system, cluster=args.cluster)
    print(f"{r['system']}: score {_pct(r['score'])}, observed rank {r['rank_observed']}, "
          f"95% rank interval {r['rank_interval_95']}, top-rank freq.={r['p_best']:.2f}")
    print(f"Inside the 95% confidence set of the best: {r['in_confidence_set_of_best']} (set size {r['confidence_set_size']})")
    c = r["vs_best_other"]
    print(f"Paired test, {c['b']} vs {c['a']}: difference {100 * c['diff']:+.1f} pts, one-sided exact p={c['p_one_sided']:.3f}")
    if "winner_adjusted" in r:
        w = r["winner_adjusted"]
        print(f"Selection-adjusted score: {_pct(w['estimate'])}, 95% interval [{_pct(w['ci'][0])}, {_pct(w['ci'][1])}] "
              f"(reported {_pct(w['observed'])})")


def cmd_pr_check(args):
    """Markdown report on one new submission, suitable for a pull-request comment."""
    import json
    import numpy as np
    from .loaders import board_with_submission
    tasks = None
    if args.tasks:
        with open(args.tasks, encoding="utf-8") as f:
            tasks = json.load(f)
    b, k = board_with_submission(args.eval_dir, args.new_dir, tasks=tasks)
    if tasks is None:
        print("_Warning: no --tasks file was given, so tasks that no submission solved are not counted and every score is overstated._\n")
    if b.K < 2:
        print(f"### honestboard check: `{b.names[k]}`\n\n- First submission on this board: score {_pct(b.scores()[k])} on {b.T:,} tasks; nothing to compare with yet.")
        return
    sc = b.scores()
    others = [j for j in np.argsort(-sc, kind="stable") if j != k]
    prev = others[0]
    res = b.resolution()
    rs = {r["name"]: r for r in b.rank_summary(B=4000)}
    me = rs[b.names[k]]
    margin = sc[k] - sc[prev]
    lines = [f"### honestboard check: `{b.names[k]}`", ""]
    lines.append(f"- Score: **{_pct(sc[k])}** on {b.T:,} tasks; observed rank {1 + int((sc > sc[k]).sum())} of {b.K}; "
                 f"95% rank interval {me['rank_lo']:.0f} to {me['rank_hi']:.0f}; top-rank frequency {me['p_best']:.2f}.")
    if margin > 1e-12:
        c = b.compare(b.names[prev], b.names[k], B=3000)
        verdict = "beats" if c["p_selective"] <= 0.05 else "cannot be told apart from"
        lines.append(f"- Current record: `{b.names[prev]}` at {_pct(sc[prev])}. This submission is {100 * margin:+.1f} points above it "
                     f"(selective exact paired p, conditional on it scoring higher = {c['p_selective']:.3f}), so it **{verdict}** the current record at the 5% level.")
        if abs(100 * margin) < 100 * res["mdd"]:
            lines.append(f"- This board detects differences of about {100 * res['mdd']:.1f} points with 80% power, so a margin of {100 * margin:.1f} points is below its resolution.")
        w = b.winner_adjusted(B=3000)
        lines.append(f"- Selection-adjusted score for the new leader: {_pct(w['estimate'])} (95% interval {_pct(w['ci'][0])} to {_pct(w['ci'][1])}).")
    elif abs(margin) <= 1e-12:
        lines.append(f"- Current record: `{b.names[prev]}` at {_pct(sc[prev])}. This submission ties it exactly and does not set a new record.")
    else:
        c = b.compare(b.names[k], b.names[prev], B=3000)
        lines.append(f"- Current record: `{b.names[prev]}` at {_pct(sc[prev])}. This submission is {100 * -margin:.1f} points below it and does not set a new record "
                     f"(one-sided p that the record is better = {c['p_one_sided']:.3f}).")
    lines.append("")
    lines.append("_Task sampling only; single runs; see the honestboard documentation for limits._")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="honestboard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pr-check", help="check one new submission against a leaderboard checkout and print Markdown")
    p.add_argument("eval_dir", help="evaluation folder of the leaderboard checkout, for example experiments/evaluation/verified")
    p.add_argument("new_dir", help="the new submission folder")
    p.add_argument("--tasks", help="JSON file with the benchmark's full task id list (recommended)")
    p.set_defaults(fn=cmd_pr_check)
    for name, fn in (("report", cmd_report), ("check", cmd_check)):
        p = sub.add_parser(name)
        p.add_argument("board")
        if name == "report":
            p.add_argument("--top", type=int, default=10)
        else:
            p.add_argument("system")
        p.add_argument("--cluster", action="store_true",
                       help="resample task clusters (needs a 'clusters' map in the board file) instead of tasks")
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
