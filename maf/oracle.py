r"""Dependency oracle (D26, D29): how right is the tracker's dependency closure?

At temperature 0 replay is exact, so a fault episode under `none` differs from the clean episode of the same seed
only where the fault's effect really reaches. That is the TRUE affected set. The PREDICTED set is what
Tracker.downstream() returns for the fault, rebuilt from the same episode's events.

  python -m maf.oracle analyze results/dev.jsonl [--csv out.csv] [--changes]
  python -m maf.oracle probe --backend mock --seeds 0-9 --out results/probes.jsonl     # "should not matter" probes
  python -m maf.oracle analyze results/probes.jsonl

"Changed" is reported two ways (D29): strict = the normalized artifact differs at all; decision = the task-relevant
part differs (plan: supplier, qty; execute: tool and args of every action; notify: supplier and cc). The research
stages only copy tool output and read nothing, so they count as affected only when they are the faulted stage.
The faulted stage is left out of both the true and the predicted set (D36): only downstream stages are scored.
An empty set is still counted; precision (empty predicted set) or recall (empty true set) is then n/a. precision = |pred & true| / |pred| (below 1: over-invalidation,
wasted recompute); recall = |pred & true| / |true| (below 1: under-invalidation, invalid results reused).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys

from .experiment import episode_key, make_llm, parse_seeds, pin_run, resolve_ledger, run_episode
from .faults import PROBE_STAGES, FaultSpec
from .log import DEFAULT_LOG_DIR, run_dir_for, setup_logging
from .trace import load_attempts
from .tracker import StageRun, Tracker
from .workflow import STAGES, Config

log = logging.getLogger("maf.oracle")

NOTIONS = ("strict", "decision")
TRACKS = {True: "selective", False: "selective_msg"}   # track_state -> the strategy that uses it


# ---------------- reading events ----------------

def split_key(key: str):
    """model|seed|fault|strategy -> (model, seed, fault, strategy)."""
    parts = key.split("|")
    return "|".join(parts[:-3]), int(parts[-3]), parts[-2], parts[-1]


def load_episodes(path):
    """key -> events of its latest complete attempt. Returns (episodes, incomplete attempt count)."""
    attempts: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            attempts.setdefault((e["episode"], e["attempt"]), []).append(e)
    best, incomplete = {}, 0
    for (key, att), evs in attempts.items():
        if evs[-1]["kind"] != "episode_end":
            incomplete += 1
        elif att > best.get(key, (0, None))[0]:
            best[key] = (att, evs)
    return {k: v[1] for k, v in best.items()}, incomplete


def count_errors(results_path) -> int:
    """Error records whose episode never succeeded (a retried episode is counted once, as done)."""
    ok, bad = set(), set()
    try:
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                (bad if "error" in rec else ok).add(rec["key"])
    except FileNotFoundError:
        pass
    return len(bad - ok)


def artifacts(events) -> dict:
    """stage -> committed artifact content (last one per stage)."""
    return {e["stage"]: e["payload"]["content"] for e in events if e["kind"] == "artifact"}


def fault_of(events):
    """(kind, stage, run_id) of the injected fault, or None."""
    for e in events:
        if e["kind"] == "fault":
            return e["payload"]["fault"], e["stage"], e["run_id"]
    return None


def rebuild_tracker(events) -> Tracker:
    """Replay what workflow.py recorded into a real Tracker: artifact reads from the artifact event's parents,
    shared-state reads from state_read events (key, writer)."""
    run_of = {e["id"]: e["run_id"] for e in events if e["kind"] == "artifact"}
    tr = Tracker(STAGES)

    def run(e):
        rid = e["run_id"]
        if rid not in tr.runs:
            tr.runs[rid] = StageRun(rid, e["stage"], int(rid.split("#")[1]))
        return tr.runs[rid]

    for e in events:
        if e["kind"] == "state_read":
            tr.note_state_read(run(e), e["payload"]["key"], e["payload"]["writer"])
        elif e["kind"] == "artifact":
            r = run(e)
            for p in e["parents"]:
                if p in run_of:
                    tr.note_artifact_read(r, run_of[p])
            tr.current[e["stage"]] = r.run_id
    return tr


def after(fault_stage: str) -> list:
    """Stages that run after fault_stage in workflow order (D38)."""
    return STAGES[STAGES.index(fault_stage) + 1:]


def predicted(tracker: Tracker, fault_run: str, track_state: bool) -> set:
    """Stages the tracker says depend on the fault, restricted to stages after it (same scope as true_set)."""
    stages = {tracker.runs[r].stage for r in tracker.downstream(fault_run, track_state)}
    return stages & set(after(fault_run.split("#")[0]))


# ---------------- strict vs decision ----------------

def signature(stage, content, notion):
    """Comparable value of an artifact. None for stages with no decision content (see module doc)."""
    if notion == "strict":
        return json.dumps(content, sort_keys=True, default=str)
    if stage == "plan":
        return (content.get("supplier"), content.get("qty"))
    if stage == "execute":
        return sorted(json.dumps([a.get("tool"), a.get("args")], sort_keys=True, default=str)
                      for a in content.get("actions", []) if isinstance(a, dict))
    if stage == "notify":
        return (content.get("supplier"), content.get("cc"))
    return None


def true_set(base: dict, faulty: dict, fault_stage: str, notion: str) -> set:
    """Stages AFTER the faulted stage whose artifact differs from the clean baseline (D36, D38). The faulted stage
    and earlier stages are never scored: earlier ones saw identical inputs, so a difference there is model noise."""
    return {s for s in after(fault_stage) if s in base and s in faulty
            and signature(s, base[s], notion) != signature(s, faulty[s], notion)}


# ---------------- analysis ----------------

def observe(base_events, fault_events):
    """One observation: the fault, the tracker's prediction for both modes, and the true set for both notions."""
    kind, stage, run_id = fault_of(fault_events)
    tr = rebuild_tracker(fault_events)
    base, faulty = artifacts(base_events), artifacts(fault_events)
    return {"kind": kind, "stage": stage,
            "pred": {t: predicted(tr, run_id, t) for t in TRACKS},
            "true": {n: true_set(base, faulty, stage, n) for n in NOTIONS}}


def cross_check(tr, fault_run, rollback_runs, track_state) -> bool:
    """Does a recovery episode's rollback event invalidate exactly the stages the replayed tracker predicts?"""
    return {r.split("#")[0] for r in rollback_runs} - {fault_run.split("#")[0]} == predicted(tr, fault_run, track_state)


def collect(path):
    """Read <path>.events.jsonl and build observations. Returns (obs, cross, skipped)."""
    eps, incomplete = load_episodes(path + ".events.jsonl")
    skipped = {"error_records": count_errors(path), "incomplete_attempts": incomplete,
               "timeout": 0, "no_baseline": 0, "no_fault_event": 0}
    obs, cross = [], {t: [0, 0, 0] for t in TRACKS}      # per track: match, mismatch, no recovery episode
    for key, evs in sorted(eps.items()):
        model, seed, fault, strategy = split_key(key)
        if fault == "clean" or strategy != "none":
            continue
        if fault.startswith("timeout@"):
            skipped["timeout"] += 1
            continue
        base = eps.get(episode_key(model, seed, None, "none"))
        info = fault_of(evs)
        if base is None:
            skipped["no_baseline"] += 1
            continue
        if info is None:
            skipped["no_fault_event"] += 1
            continue
        o = observe(base, evs)
        o["seed"] = seed
        obs.append(o)
        tr = rebuild_tracker(evs)
        for t, name in TRACKS.items():
            rec = eps.get(f"{model}|{seed}|{fault}|{name}")
            rb = next((e for e in rec or [] if e["kind"] == "rollback"), None)
            if rb is None:
                cross[t][2] += 1
            else:
                cross[t][0 if cross_check(tr, info[2], rb["payload"]["runs"], t) else 1] += 1
    return obs, cross, skipped


def score(group, track, notion) -> dict:
    pred = sum(len(o["pred"][track]) for o in group)
    true = sum(len(o["true"][notion]) for o in group)
    both = sum(len(o["pred"][track] & o["true"][notion]) for o in group)
    return {"n": len(group), "tracker": TRACKS[track], "notion": notion, "pred": pred, "true": true, "both": both,
            "over": pred - both, "under": true - both,
            "precision": both / pred if pred else float("nan"), "recall": both / true if true else float("nan")}


def groupings(obs):
    """(level, label, observations): per fault kind, per kind@stage, pooled."""
    out = []
    for level, keyf in (("kind", lambda o: o["kind"]), ("stage", lambda o: f"{o['kind']}@{o['stage']}")):
        for label in sorted({keyf(o) for o in obs}):
            out.append((level, label, [o for o in obs if keyf(o) == label]))
    return out + [("pooled", "ALL", obs)]


def score_table(obs) -> list:
    return [{"level": lv, "group": lab, **score(g, t, n)}
            for lv, lab, g in groupings(obs) for t in TRACKS for n in NOTIONS]


def change_table(obs) -> list:
    """Per (group, stage): how often the stage really changed (strict / decision) and how often it was predicted."""
    rows = []
    for lv, lab, g in groupings(obs):
        if lv != "stage":
            continue
        for s in STAGES:
            rows.append({"group": lab, "stage": s, "n": len(g),
                         "strict": sum(s in o["true"]["strict"] for o in g),
                         "decision": sum(s in o["true"]["decision"] for o in g),
                         "pred_selective": sum(s in o["pred"][True] for o in g),
                         "pred_selective_msg": sum(s in o["pred"][False] for o in g)})
    return rows


def fmt(x):
    return "  n/a" if x != x else f"{x:5.2f}"


def print_scores(rows):
    print(f"{'level':<7}{'group':<30}{'tracker':<14}{'notion':<9}{'n':>4}{'pred':>6}{'true':>6}{'both':>6}"
          f"{'over':>6}{'under':>6}{'prec':>7}{'recall':>7}")
    for r in rows:
        print(f"{r['level']:<7}{r['group']:<30}{r['tracker']:<14}{r['notion']:<9}{r['n']:>4}{r['pred']:>6}"
              f"{r['true']:>6}{r['both']:>6}{r['over']:>6}{r['under']:>6} {fmt(r['precision'])} {fmt(r['recall'])}")


def print_changes(rows):
    print(f"\n{'group':<30}{'stage':<18}{'n':>4}{'strict':>7}{'decision':>9}{'pred_sel':>9}{'pred_msg':>9}")
    for r in rows:
        print(f"{r['group']:<30}{r['stage']:<18}{r['n']:>4}{r['strict']:>7}{r['decision']:>9}"
              f"{r['pred_selective']:>9}{r['pred_selective_msg']:>9}")


def analyze(args):
    obs, cross, skipped = collect(args.results)
    print(f"{len(obs)} fault episodes analysed; skipped: " + ", ".join(f"{k}={v}" for k, v in skipped.items()))
    for t, name in TRACKS.items():
        m, mm, none = cross[t]
        print(f"cross-check vs {name} rollback events: match={m} mismatch={mm} no_{name}_episode={none}")
    rows = score_table(obs)
    print_scores(rows)
    changes = change_table(obs)
    shown = changes if args.changes else [r for r in changes if r["group"].startswith("probe_")]
    if shown:
        print_changes(shown)
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["level"])
            w.writeheader()
            w.writerows(rows)
        print(f"csv -> {args.csv}")


# ---------------- probes ----------------

def probe(args):
    """Strategy none only: one clean episode per seed plus one per probe, with events on."""
    run_dir = run_dir_for(args.log_dir, "oracle", args.model_label) if args.log_dir else None
    setup_logging(run_dir, "WARNING")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    label = args.model_label if args.backend == "llama" else "mock"
    faults = [None] + [FaultSpec(k, s) for k, ss in PROBE_STAGES.items() for s in ss]
    grid = [(seed, f) for seed in parse_seeds(args.seeds) for f in faults]
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {r["key"] for r in map(json.loads, f) if "error" not in r}
    events_path = args.out + ".events.jsonl"
    seen = load_attempts(events_path)
    pin_run(args, label)
    ledger = resolve_ledger(args)
    llm, cfg = make_llm(args), Config()
    todo = [(s, f) for s, f in grid if episode_key(label, s, f, "none") not in done]
    print(f"{len(grid)} episodes, {len(todo)} to run -> {os.path.abspath(args.out)}")
    for seed, fault in todo:
        key = episode_key(label, seed, fault, "none")
        try:
            rec, _ = run_episode(llm, label, seed, fault, "none", cfg, events_path=events_path,
                                 attempt=seen.get(key, 0) + 1, ledger=ledger)
        except Exception as exc:
            log.exception("episode failed: %s", key)
            rec = {"key": key, "error": repr(exc)}
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"done. next: python -m maf.oracle analyze {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="tracker precision/recall against true effects, from <results>.events.jsonl")
    a.add_argument("results")
    a.add_argument("--csv")
    a.add_argument("--changes", action="store_true", help="also print per-stage change rates for every group")
    a.set_defaults(fn=analyze)
    b = sub.add_parser("probe", help="run clean + 'should not matter' probe episodes under strategy none")
    b.add_argument("--backend", choices=["mock", "llama"], default="mock")
    b.add_argument("--base-url", default="http://127.0.0.1:8080")
    b.add_argument("--model-label", default="local")
    b.add_argument("--seeds", default="0-9")
    b.add_argument("--out", required=True)
    b.add_argument("--temperature", type=float, default=0.0)
    b.add_argument("--max-tokens", type=int, default=512)
    b.add_argument("--seed-offset", type=int, default=0)
    b.add_argument("--fold-system", action="store_true")
    b.add_argument("--obey-prob", type=float, default=0.8)
    b.add_argument("--verifier-error", type=float, default=0.0)
    b.add_argument("--pp-tps", type=float, default=200.0)
    b.add_argument("--tg-tps", type=float, default=15.0)
    b.add_argument("--ledger", choices=["sqlite", "memory"],
                   help="orders and ledger backend (D37); default: sqlite for llama, memory for mock")
    b.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    b.set_defaults(fn=probe)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
