"""Scaling micro-benchmark for the tracker (D25, D27 item 3), on synthetic dependency graphs.

    python -m maf.microbench [--sizes 5,10,50,100,500,1000] [--repeats R] [--out CSV]
    python -m maf.microbench --trace [--seeds 0,1,2] [--repeats R] [--out CSV]

Default: no model and no Episode, only Tracker and StageRun. --trace (D22): real mock episodes with the
event trace on (temp file) and off; reports trace and tracker overhead.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import tempfile
import time

from maf.experiment import run_episode
from maf.faults import ALL_STAGES, FaultSpec
from maf.llm import MockLLM
from maf.tracker import StageRun, Tracker
from maf.workflow import Config

DEFAULT_SIZES = (5, 10, 50, 100, 500, 1000)
WIDTH = 3   # runs per layer in the layered DAG
LIVE_EPISODE_S = 16.0   # estimate: first attempt's measured wall time per clean live episode (docs/REFERENCE.md)
TRACE_STRATEGIES = ("none", "selective", "selective_gated", "verify_hybrid")
TRACE_FIELDS = ("trace", "episodes", "events_per_ep", "trace_ms_ep_med", "trace_ms_ep_p95",
                "trace_us_event_med", "trace_us_event_p95", "tracker_us_ep_med", "wall_ms_ep_med")
FIELDS = ("shape", "n", "op", "median_us", "p95_us", "closure")


def make_runs(shape: str, n: int):
    """Return (tracker, reads): runs registered and live, reads = pending (kind, run, args) calls."""
    stages = [f"s{i}" for i in range(n)]
    tr = Tracker(stages)
    reads = []
    for i, st in enumerate(stages):
        run = StageRun(f"r{i}", st, 1)
        tr.runs[run.run_id] = run
        tr.current[st] = run.run_id
        if shape == "chain":
            if i:
                reads.append(("artifact", run, f"r{i - 1}"))
        else:
            first = (i // WIDTH - 1) * WIDTH   # first run of the previous layer
            if first >= 0:
                for j in range(2):             # two message reads
                    reads.append(("artifact", run, f"r{first + (i + j) % WIDTH}"))
                reads.append(("state", run, f"r{first + (i + 2) % WIDTH}"))   # one shared-state read
    return tr, reads


def record(tr: Tracker, reads) -> None:
    for kind, run, src in reads:
        if kind == "artifact":
            tr.note_artifact_read(run, src)
        else:
            tr.note_state_read(run, f"key_{src}", src)


def p95(xs):
    return sorted(xs)[min(len(xs) - 1, int(0.95 * len(xs)))]


def bench(shape: str, n: int, repeats: int):
    """Rows for one graph: read recording, then downstream() from first/middle/last run."""
    rec_us = []
    for _ in range(repeats):
        tr, reads = make_runs(shape, n)
        t0 = time.perf_counter_ns()
        record(tr, reads)
        rec_us.append((time.perf_counter_ns() - t0) / 1e3 / max(1, len(reads)))
    rows = [(shape, n, "record_read", statistics.median(rec_us), p95(rec_us), "")]

    tr, reads = make_runs(shape, n)
    record(tr, reads)
    for where, idx in (("first", 0), ("middle", n // 2), ("last", n - 1)):
        for track_state in (True, False):
            us = []
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                closure = tr.downstream(f"r{idx}", track_state)
                us.append((time.perf_counter_ns() - t0) / 1e3)
            op = f"downstream_{where}_state{'On' if track_state else 'Off'}"
            rows.append((shape, n, op, statistics.median(us), p95(us), len(closure)))
    return rows


def trace_bench(seeds, repeats):
    """Rows (trace on, trace off) over clean + wrong_result at each stage x TRACE_STRATEGIES x seeds."""
    faults = [None] + [FaultSpec("wrong_result", st) for st in ALL_STAGES]
    cfg = Config()
    out = {}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "events.jsonl")
        for on in (True, False):
            tr_ep, tr_ev, trk, wall, evs = [], [], [], [], []
            for _ in range(repeats):
                for seed in seeds:
                    for fault in faults:
                        for strat in TRACE_STRATEGIES:
                            rec, _ = run_episode(MockLLM(seed=seed), "microbench", seed, fault, strat, cfg,
                                                 events_path=path if on else None)
                            n = 0
                            if on:
                                with open(path, "rb") as f:
                                    n = sum(1 for _ in f)
                                os.remove(path)
                            tr_ep.append(rec["trace_overhead_s"])
                            tr_ev.append(rec["trace_overhead_s"] / n if n else 0.0)
                            trk.append(rec["tracking_overhead_s"])
                            wall.append(rec["wall_time_s"])
                            evs.append(n)
            out[on] = (len(wall), statistics.mean(evs), tr_ep, tr_ev, trk, wall)
    rows = []
    for on in (True, False):
        n, ev, tr_ep, tr_ev, trk, wall = out[on]
        rows.append(("on" if on else "off", n, ev, statistics.median(tr_ep) * 1e3, p95(tr_ep) * 1e3,
                     statistics.median(tr_ev) * 1e6, p95(tr_ev) * 1e6,
                     statistics.median(trk) * 1e6, statistics.median(wall) * 1e3))
    return rows


def print_trace(rows):
    print(f"{'trace':<6}{'episodes':>9}{'events/ep':>10}{'trace ms/ep med':>16}{'p95':>8}"
          f"{'us/event med':>14}{'p95':>8}{'tracker us/ep':>14}{'wall ms/ep':>11}")
    for t, n, ev, a, b, c, d, e, w in rows:
        print(f"{t:<6}{n:>9}{ev:>10.1f}{a:>16.3f}{b:>8.3f}{c:>14.2f}{d:>8.2f}{e:>14.2f}{w:>11.3f}")
    on = rows[0]
    pct = lambda secs: 100 * secs / LIVE_EPISODE_S
    print(f"ESTIMATE vs ~{LIVE_EPISODE_S:.0f} s live episode (first attempt, clean): trace {pct(on[3] / 1e3):.4f}%, "
          f"tracker {pct(on[7] / 1e6):.6f}%")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    ap.add_argument("--repeats", type=int, help="default 200 (tracker) or 3 (--trace)")
    ap.add_argument("--out", help="optional CSV path")
    ap.add_argument("--trace", action="store_true", help="trace-overhead section (mock episodes, trace on vs off)")
    ap.add_argument("--seeds", default="0,1,2", help="--trace only")
    args = ap.parse_args(argv)
    if args.trace:
        rows = trace_bench([int(x) for x in args.seeds.split(",")], args.repeats or 3)
        print_trace(rows)
        if args.out:
            with open(args.out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(TRACE_FIELDS)
                w.writerows(rows)
        return rows
    sizes = [int(s) for s in args.sizes.split(",")]   # N=5 (the real workflow) first if listed first

    rows = [r for n in sizes for shape in ("chain", "dag") for r in bench(shape, n, args.repeats or 200)]
    print(f"{'shape':<6}{'N':>6}  {'op':<28}{'median_us':>11}{'p95_us':>11}{'closure':>9}")
    for shape, n, op, med, hi, cl in rows:
        print(f"{shape:<6}{n:>6}  {op:<28}{med:>11.2f}{hi:>11.2f}{cl:>9}")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            w.writerows((s, n, op, f"{m:.3f}", f"{h:.3f}", c) for s, n, op, m, h, c in rows)
    return rows


if __name__ == "__main__":
    main()
