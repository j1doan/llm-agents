r"""Experiment grid runner.

Grid = seeds x (clean + every supported fault kind@stage) x strategies. Each episode appends one JSON
line to --out. Re-running the same command resumes: finished episodes are skipped, errored ones retried.

  python -m maf.experiment --backend mock --seeds 0-4 --out results/mock.jsonl
  python -m maf.experiment --backend llama --base-url http://127.0.0.1:8080 --model-label gemma4-12b-it-qat-q4_0 \
      --seeds 0-4 --out results/phase1.jsonl --strategies none,full_restart --kinds wrong_result,malicious
  python -m maf.experiment --estimate --pp-tps 150 --tg-tps 12 --seeds 0-4     # cost estimate, no model
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from .faults import FAULT_STAGES, FaultSpec
from .llm import LlamaServerClient, MockLLM
from .log import DEFAULT_LOG_DIR, run_dir_for, setup_logging
from .pins import collect_pins, fetch_props, require_server_ctx, write_pins
from .strategies import STRATEGIES
from .trace import EventTrace, load_attempts
from .workflow import Config, Episode
from .world import make_world

log = logging.getLogger("maf.experiment")   # not __name__: that is "__main__" under python -m


SERVER_CONTEXT = 4096    # llama-server -c (scripts/serve_llama.sh); not an experiment argument
WARMUP_SEED = 9999    # outside the dev (0-9) and evaluation (100-109) seed sets; its result is never recorded


def resolve_ledger(args) -> str:
    """D37: sqlite by default for live (llama) runs, memory for mock; --ledger overrides."""
    return getattr(args, "ledger", None) or ("sqlite" if args.backend == "llama" else "memory")


def pin_run(args, label, model_path=None):
    """Write the D13 pin file for a run. Llama: refuse (before any episode) if the server n_ctx is not 4096 (D35)."""
    props = fetch_props(args.base_url) if args.backend == "llama" else None
    if props is not None:
        require_server_ctx(props, SERVER_CONTEXT)
    settings = {"temperature": args.temperature, "context": SERVER_CONTEXT, "max_output_tokens": args.max_tokens,
                "ledger_backend": resolve_ledger(args),
                "seed_policy": f"fixed seed (--seed-offset {args.seed_offset})"}
    return write_pins(args.out, collect_pins(args.backend, label, settings, model_path, props))


def parse_seeds(text: str) -> list:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def build_grid(seeds, kinds, strategies, include_clean=True):
    grid = []
    for seed in seeds:
        if include_clean:
            grid += [(seed, None, s) for s in strategies]
        for kind in kinds:
            for stage in FAULT_STAGES[kind]:
                grid += [(seed, FaultSpec(kind, stage), s) for s in strategies]
    return grid


def episode_key(label, seed, fault, strategy) -> str:
    return f"{label}|{seed}|{fault.kind + '@' + fault.stage if fault else 'clean'}|{strategy}"


def run_episode(llm, label, seed, fault, strategy, cfg, save_trace=False, events_path=None, attempt=1,
                ledger="memory"):
    """events_path: write the per-event trace (D22) there; attempt numbers the try within that file.
    ledger: "memory" or "sqlite" (D37); the SQLite file is deleted when the episode ends, even on error."""
    key = episode_key(label, seed, fault, strategy)
    tracer = EventTrace(events_path, key, attempt) if events_path else None
    ep = Episode(make_world(seed), llm, fault, cfg, tracer, ledger)
    log.info("episode %s", key)
    try:
        try:
            t0 = time.perf_counter()
            STRATEGIES[strategy](ep)
            wall = time.perf_counter() - t0
        except BaseException:
            if tracer:
                tracer.close()      # no episode_end: the attempt stays marked incomplete
            raise
        ep.end_trace()
        log.info("episode done %s in %.1fs", key, wall)
        rec = {"key": key, "model": label, "backend": llm.name,
               "seed": seed, "fault_kind": fault.kind if fault else "clean",
               "fault_stage": fault.stage if fault else "none", "strategy": strategy,
               "wall_time_s": wall, **ep.summary(),
               "ledger_backend": ledger, "ledger_io_s": ep.state.io_s}
        return rec, (ep.trace() if save_trace else None)
    finally:
        ep.close()


def make_llm(args):
    if args.backend == "mock":
        return MockLLM(seed=args.seed_offset, obey_prob=args.obey_prob, verifier_error=args.verifier_error,
                       pp_tps=args.pp_tps, tg_tps=args.tg_tps)
    return LlamaServerClient(base_url=args.base_url, model=args.model_label, temperature=args.temperature,
                             seed=args.seed_offset, max_tokens=args.max_tokens, fold_system=args.fold_system)


def estimate(args, grid):
    """Run the grid on the mock backend and convert its token counts into hours at the given speeds."""
    llm = MockLLM(obey_prob=args.obey_prob, pp_tps=args.pp_tps, tg_tps=args.tg_tps)
    cfg = Config(timeout_s=args.timeout_s)
    per = {}
    for seed, fault, strategy in tqdm(grid, desc="estimate", unit="ep"):
        rec, _ = run_episode(llm, "estimate", seed, fault, strategy, cfg)
        a = per.setdefault(strategy, [0, 0, 0, 0.0])
        a[0] += 1
        a[1] += rec["llm_calls"]
        a[2] += rec["prompt_tokens"] + rec["completion_tokens"]
        a[3] += rec["llm_time_s"]
    print(f"episodes: {len(grid)}   (speeds assumed: prefill {args.pp_tps} tok/s, decode {args.tg_tps} tok/s)")
    print(f"{'strategy':<20}{'episodes':>9}{'llm_calls':>10}{'tokens':>10}{'est_hours':>10}")
    for s, (n, calls, toks, secs) in per.items():
        print(f"{s:<20}{n:>9}{calls:>10}{toks:>10}{secs / 3600:>10.2f}")
    print(f"{'TOTAL':<20}{len(grid):>9}{sum(v[1] for v in per.values()):>10}"
          f"{sum(v[2] for v in per.values()):>10}{sum(v[3] for v in per.values()) / 3600:>10.2f}")
    print("Token counts are character-based estimates; real prompts/outputs will differ. Calibrate with maf.bench.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["mock", "llama"], default="mock")
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--model-label", default="local", help="recorded in results; also sent as the model name")
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--kinds", default=",".join(FAULT_STAGES))
    p.add_argument("--strategies", default=",".join(STRATEGIES))
    p.add_argument("--no-clean", action="store_true", help="skip the fault-free control episodes")
    p.add_argument("--out", default="results/results.jsonl")
    p.add_argument("--save-traces", action="store_true", help="also write <out>.traces.jsonl (per-episode summary)")
    p.add_argument("--no-trace", action="store_true", help="do not write the per-event trace <out>.events.jsonl")
    p.add_argument("--model-path", help="model file to hash into the pin file (slow for ~7 GB; off by default)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel episodes; >1 distorts latency, only use with a multi-slot server")
    p.add_argument("--limit", type=int, default=0, help="stop after N episodes (pilot runs)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed-offset", type=int, default=0, help="model sampling seed")
    p.add_argument("--fold-system", action="store_true", help="merge the system prompt into the user turn")
    p.add_argument("--ledger", choices=["sqlite", "memory"],
                   help="orders and ledger backend (D37); default: sqlite for llama, memory for mock")
    p.add_argument("--timeout-s", type=float, default=60.0, help="time charged for an injected timeout")
    p.add_argument("--obey-prob", type=float, default=0.8, help="mock: chance of obeying an injection")
    p.add_argument("--verifier-error", type=float, default=0.0, help="mock: verifier flip probability")
    p.add_argument("--pp-tps", type=float, default=200.0, help="mock/estimate: prefill tokens per second")
    p.add_argument("--tg-tps", type=float, default=15.0, help="mock/estimate: decode tokens per second")
    p.add_argument("--estimate", action="store_true", help="print cost estimate for the grid and exit")
    p.add_argument("--no-warmup", action="store_true",
                   help="skip the discarded warm-up episode run against llama-server before the grid")
    p.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR),
                   help="parent directory for the run's logs (default: <project>/logs); empty disables logging")
    p.add_argument("-v", "--verbose", action="store_true", help="also show INFO logs on the console")
    args = p.parse_args(argv)
    args.ledger = resolve_ledger(args)

    strategies = [s for s in args.strategies.split(",") if s]
    kinds = [k for k in args.kinds.split(",") if k]
    for s in strategies:
        if s not in STRATEGIES:
            p.error(f"unknown strategy {s}")
    for k in kinds:
        if k not in FAULT_STAGES:
            p.error(f"unknown fault kind {k}")
    grid = build_grid(parse_seeds(args.seeds), kinds, strategies, include_clean=not args.no_clean)
    if args.estimate:
        setup_logging(None, "WARNING")
        estimate(args, grid)
        return

    run_dir = run_dir_for(args.log_dir, "experiment", args.model_label) if args.log_dir else None
    setup_logging(run_dir, "INFO" if args.verbose else "WARNING")
    log.info("args: %s", vars(args))
    if run_dir:
        (run_dir / "config.json").write_text(json.dumps({"args": vars(args), "out": os.path.abspath(args.out)},
                                                         indent=2), encoding="utf-8")
        print(f"logging to {run_dir}")
    if not os.path.dirname(args.out):
        log.warning("--out %r has no directory part and will be written to the cwd. In Git Bash use forward "
                    "slashes (results/x.jsonl): backslashes are eaten by the shell", args.out)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done, errors = set(), {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if "error" not in rec:
                    done.add(rec["key"])
                else:
                    errors[rec["key"]] = errors.get(rec["key"], 0) + 1
    label = args.model_label if args.backend == "llama" else "mock"
    todo = [g for g in grid if episode_key(label, g[0], g[1], g[2]) not in done]
    already = len(grid) - len(todo)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(grid)} episodes in grid, {already} already done, {len(todo)} to run -> {os.path.abspath(args.out)}")

    print(f"pins -> {pin_run(args, label, args.model_path)}")

    llm, cfg, lock = make_llm(args), Config(timeout_s=args.timeout_s), threading.Lock()
    if args.backend == "llama" and not args.no_warmup and todo:
        try:  # a tiny request is not enough: the first full episode was still ~6 s slower (cold prompt cache)
            rec, _ = run_episode(llm, label, WARMUP_SEED, None, "none", cfg, ledger=args.ledger)
            log.info("warm-up episode (seed %d, discarded) took %.1fs, task_success=%s", WARMUP_SEED,
                     rec["wall_time_s"], rec["task_success"])
        except Exception:
            log.exception("warm-up episode failed (continuing)")
    trace_path = args.out + ".traces.jsonl"
    events_path = None if args.no_trace else args.out + ".events.jsonl"
    seen = load_attempts(events_path) if events_path else {}

    bar = tqdm(total=len(todo), desc="episodes", unit="ep", dynamic_ncols=True)

    def work(item):
        seed, fault, strategy = item
        try:
            key = episode_key(label, seed, fault, strategy)
            attempt = max(errors.get(key, 0), seen.get(key, 0)) + 1
            rec, trace = run_episode(llm, label, seed, fault, strategy, cfg, args.save_traces, events_path, attempt,
                                   args.ledger)
        except Exception as exc:  # keep going; errored episodes are retried on the next run
            log.exception("episode failed: %s", episode_key(label, seed, fault, strategy))
            rec, trace = {"key": episode_key(label, seed, fault, strategy), "error": repr(exc)}, None
        with lock:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if trace is not None:
                with open(trace_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": rec["key"], **trace}) + "\n")
            bar.set_postfix_str(rec["key"].split("|", 1)[1], refresh=False)
            bar.update()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(work, todo))
    bar.close()


if __name__ == "__main__":
    main()
