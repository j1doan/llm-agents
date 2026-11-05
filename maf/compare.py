"""Compare two results files episode by episode (repeat runs of the same grid) to measure run-to-run variability.

  python -m maf.compare results/phase1_dev.jsonl results/phase1_dev_rep2.jsonl

Only keys present in both files are compared, and only outcome fields present in both records (files from before the
gate and rule checker lack the gate and rule fields). Outcome fields must match exactly to count as identical; latency and
token counts are reported as mean absolute differences. Lists every differing episode.
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter

from .analyze import load

OUTCOME = ["task_success", "clean_success", "wrong_irreversible", "duplicate_irreversible", "unauthorized_final",
           "unauthorized_ever", "propagated_invalid", "recomputed_stages", "wasted_stage_runs", "retained_valid",
           "retained_invalid", "llm_calls", "verify_calls", "json_failures", "det_tp", "det_fp", "det_fn", "det_tn",
           "held_effects", "released_effects", "dropped_effects", "idempotent_skips", "rule_rejections",
           "llm_rejections", "blocked_actions"]
NUMERIC = ["latency_s", "llm_time_s", "prompt_tokens", "completion_tokens"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a")
    p.add_argument("b")
    args = p.parse_args(argv)
    a = {r["key"]: r for r in load(args.a)}
    b = {r["key"]: r for r in load(args.b)}
    keys = sorted(set(a) & set(b))
    print(f"{args.a}: {len(a)} episodes, {args.b}: {len(b)} episodes, {len(keys)} in common "
          f"({len(set(a) - set(b))} only in first, {len(set(b) - set(a))} only in second)")
    if not keys:
        return
    differing = {}
    per_field = Counter()
    for k in keys:
        diff = [f for f in OUTCOME if f in a[k] and f in b[k] and a[k][f] != b[k][f]]
        if diff:
            differing[k] = diff
            per_field.update(diff)
    same = len(keys) - len(differing)
    print(f"\nepisodes with identical outcome fields: {same}/{len(keys)} ({100 * same / len(keys):.1f}%)")
    if per_field:
        print("differing fields: " + ", ".join(f"{f}={n}" for f, n in per_field.most_common()))
        print("\ndiffering episodes:")
        for k, fields in differing.items():
            print(f"  {k}: " + "; ".join(f"{f} {a[k].get(f)} -> {b[k].get(f)}" for f in fields))
    print("\nnumeric fields (mean absolute difference per episode, mean value in first run):")
    for f in NUMERIC:
        d = [abs(a[k][f] - b[k][f]) for k in keys]
        print(f"  {f:<18}{statistics.fmean(d):>10.2f}   (mean {statistics.fmean(a[k][f] for k in keys):.2f})")
    diff_t = [b[k]["latency_s"] - a[k]["latency_s"] for k in keys]
    print(f"  latency_s second minus first: mean {statistics.fmean(diff_t):+.2f} s")


if __name__ == "__main__":
    main()
