"""Benchmark the model server on the real workflow prompts and check clean-task quality.

Runs N fault-free episodes with strategy `none` and `verify_all`, then prints per-role latency, token counts,
decode speed and success rate. Run this first: it tells you whether the model can do the clean task at
all and what a grid will cost.

  python -m maf.bench --base-url http://127.0.0.1:8080 --model-label gemma4-12b-it-qat-q4_0 -n 5
  python -m maf.bench --model-label gemma4-12b-it-qat-q4_0 -n 1 --strategies none
  python -m maf.bench --backend mock -n 5 --log-dir ""

Each run writes logs/bench_<time>_<label>/ with config.json (args, server /props), summary.txt (the printed
tables), run.log (DEBUG log of everything: prompts, raw replies, faults, rollbacks), calls.jsonl (every LLM call with raw text, finish reason and timings) and episodes.jsonl (per-episode
metrics).
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import urllib.request
from collections import defaultdict

from tqdm import tqdm

from .llm import LlamaServerClient, MockLLM
from .log import DEFAULT_LOG_DIR, run_dir_for, setup_logging
from .strategies import STRATEGIES
from .workflow import Config, Episode
from .world import make_world


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["mock", "llama"], default="llama")
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--model-label", default="local")
    p.add_argument("-n", type=int, default=3, help="number of seeds")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--fold-system", action="store_true")
    p.add_argument("--strategies", default="none,verify_all")
    p.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR),
                   help="parent directory for the run's logs (default: <project>/logs); empty disables logging")
    p.add_argument("-v", "--verbose", action="store_true", help="also show INFO logs (every LLM call, stage run) on the console")
    args = p.parse_args(argv)
    llm = (MockLLM() if args.backend == "mock" else
           LlamaServerClient(args.base_url, args.model_label, args.temperature, 0, args.max_tokens,
                             fold_system=args.fold_system))
    run_dir = None
    if args.log_dir:
        run_dir = run_dir_for(args.log_dir, "bench", args.model_label)
        props = None
        if args.backend == "llama":
            try:
                with urllib.request.urlopen(args.base_url.rstrip("/") + "/props", timeout=5) as r:
                    props = json.load(r)
            except Exception as e:
                props = {"error": repr(e)}
        (run_dir / "config.json").write_text(json.dumps({"args": vars(args), "server_props": props}, indent=2),
                                             encoding="utf-8")
    setup_logging(run_dir, "INFO" if args.verbose else "WARNING")
    logger = logging.getLogger("maf.bench")
    if run_dir:
        tqdm.write(f"logging to {run_dir}")
        logger.info("args: %s", vars(args))

    def log(name, rec):
        if run_dir:
            with open(run_dir / name, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def say(msg):
        tqdm.write(msg)
        if run_dir:
            with open(run_dir / "summary.txt", "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    for strategy in tqdm(args.strategies.split(","), desc="strategies", unit="strategy"):
        per_role = defaultdict(list)
        ok = clean = 0
        json_fail = 0
        for seed in tqdm(range(args.n), desc=f"  {strategy}", unit="seed", leave=False):
            ep = Episode(make_world(seed), llm, None, Config())
            logger.info("episode strategy=%s seed=%d", strategy, seed)
            STRATEGIES[strategy](ep)
            s = ep.summary()
            logger.info("episode done strategy=%s seed=%d task_success=%s json_failures=%s llm_calls=%s", strategy, seed,
                     s["task_success"], s["json_failures"], s["llm_calls"])
            ok += s["task_success"]
            clean += s["clean_success"]
            json_fail += s["json_failures"]
            log("episodes.jsonl", {"strategy": strategy, "seed": seed, **s})
            for c in ep.calls:
                per_role[c["role"]].append(c)
                log("calls.jsonl", {"strategy": strategy, "seed": seed, **c})
        calls = [c for cs in per_role.values() for c in cs]
        truncated = sum(c.get("finish") == "length" for c in calls)
        reasoning = sum(c.get("reasoning_chars", 0) > 0 for c in calls)
        say(f"\n== strategy={strategy}  seeds={args.n}  task_success={ok}/{args.n}  "
              f"clean_success={clean}/{args.n}  json_failures={json_fail}  "
              f"truncated={truncated}/{len(calls)}  with_reasoning={reasoning}/{len(calls)}")
        say(f"{'role':<18}{'calls':>6}{'prompt_tok':>11}{'out_tok':>9}{'lat_s':>8}{'decode_tok/s':>14}{'prefill_tok/s':>15}")
        for role, cs in per_role.items():
            lat = statistics.fmean(c["latency_s"] for c in cs)
            pt = statistics.fmean(c["prompt_tokens"] for c in cs)
            ct = statistics.fmean(c["completion_tokens"] for c in cs)
            dec = [c["timings"].get("predicted_per_second") for c in cs if c["timings"].get("predicted_per_second")]
            pre = [c["timings"].get("prompt_per_second") for c in cs if c["timings"].get("prompt_per_second")]
            dec_s = f"{statistics.fmean(dec):.1f}" if dec else f"~{ct / lat:.1f}"
            pre_s = f"{statistics.fmean(pre):.1f}" if pre else "n/a"
            say(f"{role:<18}{len(cs):>6}{pt:>11.0f}{ct:>9.0f}{lat:>8.1f}{dec_s:>14}{pre_s:>15}")
    say("\nIf task_success is well below n/n on the clean workflow, fix prompts/model before running faults.")


if __name__ == "__main__":
    main()
