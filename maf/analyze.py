"""Summarise results JSONL into markdown tables (mean +/- sample std over episodes).

  python -m maf.analyze results/mock.jsonl                 # by fault kind x strategy
  python -m maf.analyze results/mock.jsonl --by stage      # by fault stage x strategy
  python -m maf.analyze results/mock.jsonl --kind malicious --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict

COLUMNS = ["task_success", "clean_success", "propagated_invalid", "unauthorized_final", "wrong_irreversible",
           "duplicate_irreversible", "recomputed_stages", "wasted_stage_runs", "retained_valid",
           "retained_invalid", "llm_calls", "latency_s", "tracking_overhead_s"]
STRATEGY_ORDER = ["none", "full_restart", "full_restart_gated", "verify_all", "verify_hybrid", "selective_msg",
                  "selective", "selective_gated"]
DETECTORS = ["verify_all", "verify_hybrid"]


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "error" not in rec:
                rows.append(rec)
    return rows


def fmt(vals):
    vals = [float(v) for v in vals]
    mean = statistics.fmean(vals)
    if len(vals) > 1:
        return f"{mean:.2f}+/-{statistics.stdev(vals):.2f}"
    return f"{mean:.2f}"


def table(rows, by, columns):
    groups = defaultdict(list)
    for r in rows:
        group = r["fault_kind"] if by == "kind" else f"{r['fault_kind']}@{r['fault_stage']}"
        groups[(group, r["strategy"])].append(r)
    order = lambda k: (k[0], STRATEGY_ORDER.index(k[1]) if k[1] in STRATEGY_ORDER else 99)
    header = ["group", "strategy", "n"] + columns
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    flat = []
    for key in sorted(groups, key=order):
        rs = groups[key]
        cells = [fmt([r[c] for r in rs]) for c in columns]
        lines.append("| " + " | ".join([key[0], key[1], str(len(rs))] + cells) + " |")
        flat.append([key[0], key[1], len(rs)] + [statistics.fmean(float(r[c]) for r in rs) for c in columns])
    return "\n".join(lines), header, flat


def detection(rows):
    for strat in DETECTORS:
        tp, fn, fp, tn = (sum(r.get(f"det_{k}", 0) for r in rows if r["strategy"] == strat)
                          for k in ("tp", "fn", "fp", "tn"))
        if tp + fn + fp + tn:
            print(f"\n{strat} detection: TP={tp} FN={fn} FP={fp} TN={tn} "
                  f"recall={tp / max(1, tp + fn):.2f} precision={tp / max(1, tp + fp):.2f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results")
    p.add_argument("--by", choices=["kind", "stage"], default="kind")
    p.add_argument("--kind", help="restrict to one fault kind (or 'clean')")
    p.add_argument("--columns", default=",".join(COLUMNS))
    p.add_argument("--csv", help="also write the aggregated table to this CSV path")
    args = p.parse_args(argv)
    rows = load(args.results)
    if args.kind:
        rows = [r for r in rows if r["fault_kind"] == args.kind]
    columns = args.columns.split(",")
    md, header, flat = table(rows, args.by, columns)
    print(md)
    detection(rows)
    noop = [r for r in rows if r["fault_fired"] and r["fault_noop"]]
    if noop:
        print(f"\nWARNING: {len(noop)} episodes injected a fault that did not corrupt the artifact (noop).")
    masked = [r for r in rows if r["strategy"] == "none" and r["fault_fired"] and r["task_success"]
              and r["fault_kind"] in ("wrong_result", "stale")]
    if masked:
        print(f"NOTE: {len(masked)} 'none' episodes with a wrong/stale fault still succeeded (fault masked by the "
              f"model); the workload did not depend on that result there.")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(flat)


if __name__ == "__main__":
    main()
