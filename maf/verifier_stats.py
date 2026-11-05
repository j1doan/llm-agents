"""verify_all detection statistics per stage and per seed, parsed from an experiment's run.log.

  python -m maf.verifier_stats logs/experiment_<time>_<label>/run.log [more run.log ...] [--csv out.csv] [--strategy verify_hybrid]

Pass every log of a grid that was run in pieces: the Phase 1 v2 dev grid (verify_all and verify_hybrid, 10 seeds) is in
logs/experiment_20261101_013239_* (seed 0) and logs/experiment_20261101_012916_* (seeds 1-9); the counts are summed.

A verdict on the fault-injected run (the first run of the faulty stage, `<stage>#1`) is a true positive when
rejected and a false negative when accepted; every other verdict is a false positive when rejected and a true
negative when accepted (including re-verification after a rerun). Verdicts are deterministic at temperature 0,
so false positives cluster in a few world seeds: read the per-seed table, not just the totals.

For `--strategy verify_hybrid` a rule-checker rejection (`rules <run> -> REJECT`) counts the same way as a verdict, and
a final table splits the counts by checker: rules (only rejections are logged, so TP/FP) and llm (verdicts on outputs
the rules accepted).
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict

EPISODE = re.compile(r"maf\.experiment: episode (\S+)$")
VERDICT = re.compile(r"verify (\w+)#(\d+) -> (accept|REJECT)")
RULES = re.compile(r"rules (\w+)#(\d+) -> REJECT")
FORCED = re.compile(r"(\w+)#\d+: rejected by verifier but force-committed")
STAGES = ["research_pricing", "research_policy", "plan", "execute", "notify"]


def parse(paths, strategy="verify_all"):
    """paths: one run.log or a list of them (counts are summed).
    Return (per_stage, per_seed, forced, per_checker) where counts are keyed by 'tp','fn','fp','tn'.
    per_checker maps 'rules'/'llm' to counts and is only filled for verify_hybrid."""
    per_stage = defaultdict(Counter)
    per_seed = defaultdict(Counter)
    forced = Counter()
    per_checker = defaultdict(Counter)
    for path in [paths] if isinstance(paths, str) else paths:
        key = None
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = EPISODE.search(line)
                if m:
                    key = m.group(1)
                    continue
                if key is None or not key.endswith("|" + strategy):
                    continue
                m = VERDICT.search(line)
                r = RULES.search(line)
                if m or r:
                    checker = "llm" if m else "rules"
                    g = m or r
                    stage, attempt, verdict = g.group(1), int(g.group(2)), (m.group(3) if m else "REJECT")
                    seed, fault = int(key.split("|")[1]), key.split("|")[2]
                    fault_stage = fault.split("@")[1] if "@" in fault else None
                    injected = stage == fault_stage and attempt == 1
                    outcome = ("tp" if verdict == "REJECT" else "fn") if injected else \
                              ("fp" if verdict == "REJECT" else "tn")
                    per_stage[stage][outcome] += 1
                    per_seed[seed][outcome] += 1
                    per_checker[checker][outcome] += 1
                    continue
                m = FORCED.search(line)
                if m:
                    forced[int(key.split("|")[1])] += 1
    return per_stage, per_seed, forced, per_checker


def rate(n, d):
    return f"{n / d:.2f}" if d else "n/a"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", nargs="+", help="one or more run.log files; counts are summed")
    p.add_argument("--strategy", default="verify_all", choices=["verify_all", "verify_hybrid"])
    p.add_argument("--csv", help="write the per-stage and per-seed tables to this CSV")
    args = p.parse_args(argv)
    per_stage, per_seed, forced, per_checker = parse(args.log, args.strategy)
    rows = []
    print("| stage | TP | FN | FP | TN | recall | false-positive rate |\n|---|---|---|---|---|---|---|")
    tot = Counter()
    for stage in STAGES:
        c = per_stage.get(stage, Counter())
        tot.update(c)
        print(f"| {stage} | {c['tp']} | {c['fn']} | {c['fp']} | {c['tn']} | {rate(c['tp'], c['tp'] + c['fn'])} | "
              f"{rate(c['fp'], c['fp'] + c['tn'])} |")
        rows.append(["stage", stage, c["tp"], c["fn"], c["fp"], c["tn"], 0])
    print(f"| all | {tot['tp']} | {tot['fn']} | {tot['fp']} | {tot['tn']} | {rate(tot['tp'], tot['tp'] + tot['fn'])} | "
          f"{rate(tot['fp'], tot['fp'] + tot['tn'])} |")
    print(f"\nprecision {rate(tot['tp'], tot['tp'] + tot['fp'])}, forced commits (rejected twice, committed anyway): "
          f"{sum(forced.values())}")
    print("\n| seed | TP | FN | FP | TN | forced commits |\n|---|---|---|---|---|---|")
    for seed in sorted(per_seed):
        c = per_seed[seed]
        print(f"| {seed} | {c['tp']} | {c['fn']} | {c['fp']} | {c['tn']} | {forced[seed]} |")
        rows.append(["seed", seed, c["tp"], c["fn"], c["fp"], c["tn"], forced[seed]])
    if args.strategy == "verify_hybrid":
        print("\n| checker | TP | FN | FP | TN |\n|---|---|---|---|---|")
        for checker in ("rules", "llm"):
            c = per_checker[checker]
            print(f"| {checker} | {c['tp']} | {c['fn']} | {c['fp']} | {c['tn']} |")
            rows.append(["checker", checker, c["tp"], c["fn"], c["fp"], c["tn"], 0])
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["table", "group", "tp", "fn", "fp", "tn", "forced_commits"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
