# Dependency-Aware Failure Recovery for LLM Multi-Agent Workflows

`maf` is a Python package for studying how one faulty agent result spreads through a staged LLM workflow, and what each recovery strategy costs to contain it. It provides a procedurally generated task, a single-shot fault injector, eight recovery strategies, an artifact-level dependency tracker, and tooling to trace, pin, analyze and audit every run. Results are summarized in [Results](#results).

# What is maf?

`maf` runs a five-stage procurement workflow, `research_pricing`, `research_policy`, `plan`, `execute`, `notify`, in linear order with an LLM over a procedurally generated world. Each world has a price list, a supplier policy, a stock ledger and a task request, and is built so that the cheapest supplier overall is not the correct choice. A script computes ground truth from the world; no model acts as judge.

Every stage emits a typed artifact (offers, approved suppliers and delivery limit, a supplier plan, a list of tool actions, an email). The dependency tracker records artifact-level provenance over two edge types: messages passed between stages, and reads of values written to shared state. Given a faulty run, it returns the downstream invalidation closure: the runs whose inputs transitively depend on the faulty artifact. The `selective_msg` strategy tracks message edges only, which isolates the contribution of shared-state edges.

Effects are split by reversibility. Order placement and ledger updates are reversible and are undone on rollback. Emails and fund transfers are irreversible. The gated strategies route irreversible effects through an outbox: effects are held during recovery, released at the end only for runs outside the invalidation closure, and emails are deduplicated on an idempotency key over `(to, subject, body)`.

Faults are injected once per episode at the artifact level, after the model produces a stage's output and before it is validated, so every strategy faces the identical fault on the identical world. Each episode writes a per-event JSONL trace in which every record links to the events it depends on. Each results file has a pin file recording the software versions, package versions, model hash and llama-server build that produced it.

Design decisions are recorded in [docs/DECISIONS.md](docs/DECISIONS.md).

# Results

The reader-facing tables are in [results/cleaned_data/](results/cleaned_data/). They summarize the Phase 1 dev grid `phase1_v2_dev` (seeds 0-9, 8 strategies x 19 conditions = 1520 episodes) and the oracle probe run, as means over seeds unless a file says otherwise.

| File | What it holds |
|---|---|
| `01_none_propagation.csv` | `none`: success and propagation by fault |
| `02_full_restart.csv` | `full_restart`: success, irreversible effects, calls, latency, wasted runs |
| `03_verify_all.csv` | `verify_all`: success, calls, latency, recomputed stages |
| `04_gate_clean_success.csv` | clean success of restart and selective, with and without the outbox gate |
| `05_gate_effect_counts.csv` | effects held, released and dropped by the gate |
| `06_verify_hybrid.csv` | `verify_hybrid`: rules then LLM verifier |
| `07_selective.csv` | `selective` (ungated): success, calls, recomputed stages |
| `08_oracle_tracker.csv` | tracker prediction vs true downstream change (strict and decision) |
| `09_oracle_probes.csv` | probe episodes where the change should not matter |
| `10_tradeoff_long.csv` | clean success and mean tokens per strategy and fault, with Pareto flag |
| `11_microbench_closure_chain.csv` | tracker closure time vs graph size, before and after the D33 fix |
| `12_verifier_detection.csv` | TP / FN / FP / TN for `verify_all` and `verify_hybrid` |
| `13_grid_run_summary.csv` | grid size, errors, wall time, repeatability checks |
| `14_trace_event_counts.csv` | trace events by kind |
| `15_overhead.csv` | trace, tracking and ledger overhead |
| `definitions/` | stage, fault-kind, strategy and metric-field definitions |

Notes on the tables:

- Files 12-15 collect single figures: verifier counts, grid summary, trace event counts and overheads.
- `09_oracle_probes.csv`: an empty `precision_decision` means "n/a" (no true change to compare against).
- `10_tradeoff_long.csv` has one row per strategy and fault; `pareto_front` marks strategies on the cost-versus-safety Pareto front.
- `definitions/strategies.csv` lists 7 strategies; `none` (no recovery) has no row.

Headline numbers:

- **Outbox gate** (`04_gate_clean_success.csv`, `10_tradeoff_long.csv`): on `wrong_result`, `malicious` and `stale` faults, `full_restart` has clean success 0.00 and `full_restart_gated` has 1.00, at about the same mean tokens (for `wrong_result`, 3669 vs 3668).
- **Detection** (`12_verifier_detection.csv`): `verify_all` has recall 0.91 and precision 0.66.
- **Grid size** (`13_grid_run_summary.csv`): 1520 episodes, 0 error records, 9.79 h summed episode time.
- **Repeatability** (`13_grid_run_summary.csv`): 570 of 570 episodes had identical outcomes on re-run.

All numbers come from development seeds 0-9. Prompts were tuned on these seeds, so every number is in-sample. Held-out seeds 100-109 have not been run.

Other result files:

- [results/phase1_v2_charts/](results/phase1_v2_charts/): tradeoff SVGs and two sample trace HTML files.
- [results/frozen/](results/frozen/): frozen read-only copies of runs, each with `SHA256SUMS`.
- Raw per-episode records: `results/phase1_v2_dev.jsonl` (with `.events.jsonl` trace and `.pins.json`) and `results/phase1_v2_probes.jsonl`. Recompute summaries with `python -m maf.analyze results/phase1_v2_dev.jsonl`.

# How to use

## Installation

`maf` requires Python 3.14. Dependencies are `tqdm` and `pytest`.

```bash
python -m pip install -r REQUIREMENTS.txt
```

## Serving a model with llama.cpp

The `llama` backend talks to a `llama-server` endpoint. `scripts/serve_llama.sh` starts one with a quantised KV cache, flash attention and reasoning disabled. Every option is also an environment variable (`MODEL`, `LLAMA_DIR`, `PORT`, `CTX`, `NGL`, `PARALLEL`, `FA`, `KV_QUANT`, `REASONING`).

This project uses Gemma 4 12B it with INT4 quantization-aware training. See [Google's Huggingface repo](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf).

```bash
./scripts/serve_llama.sh \
    --model <path-to>/gemma-4-12b-it-qat-q4_0.gguf \
    --llama-dir <path-to>/llama.cpp-build \
    --port 8080 --ctx 4096
```

> [!WARNING]
> `n_ctx` must be 4096 with a single slot (`--parallel 1`). The runner reads `/props` from the server and refuses to start when the reported context differs.

## Running experiments

`maf.experiment` runs the grid `seeds x (fault-free + every supported fault kind@stage) x strategies` and appends one JSON line per episode to `--out`. The `mock` backend is a deterministic stand-in that needs no server.

```bash
# mock backend, no model required
python -m maf.experiment --backend mock --seeds 0-4 --out results/mock.jsonl

# llama backend
python -m maf.experiment --backend llama --base-url http://127.0.0.1:8080 \
    --model-label gemma4-12b-it-qat-q4_0 --model-path <path-to>/gemma-4-12b-it-qat-q4_0.gguf \
    --seeds 0-9 --strategies none,selective_gated --kinds wrong_result,malicious \
    --out results/run.jsonl

# cost estimate for a grid, no model calls
python -m maf.experiment --estimate --pp-tps 150 --tg-tps 12 --seeds 0-4
```

`--seeds` takes a range (`0-9`) or a list. `--strategies` and `--kinds` take comma-separated names (see [Recovery strategies](#recovery-strategies) and [Fault model](#fault-model)). `--no-clean` drops the fault-free control, `--limit N` stops after N episodes, `--ledger {sqlite,memory}` selects the orders and ledger backend, and `--log-dir` sets where run logs are written.

> [!NOTE]
> Results files are append-only and resumable. Re-running the same command skips finished episodes and retries errored ones. Never reuse an existing results file as `--out` for a different grid.

## Tracing and the trace viewer

Tracing is on by default. Next to `--out`, the runner writes `<out>.events.jsonl`, one flushed line per event, so a crash keeps the trace. Each record has the fields `id`, `episode` (`model|seed|fault|strategy`), `attempt`, `stage`, `run_id`, `kind`, `parents`, `t` (simulated seconds) and `payload`. The event kinds are `message`, `artifact`, `fault`, `state_read`, `state_write`, `tool_action`, `verify`, `rollback` and `episode_end`. Pass `--no-trace` to disable it.

`maf.traceview` renders one episode as a standalone HTML dependency graph.

```bash
python -m maf.traceview results/run.jsonl.events.jsonl --list
python -m maf.traceview results/run.jsonl.events.jsonl \
    --episode "gemma4-12b-it-qat-q4_0|0|wrong_result@plan|selective_gated" \
    --out episode.html
```

`--attempt` selects an attempt; the default is the latest complete one.

## Run pinning

Each run writes `<out>.pins.json` next to its results. It records the git revision, package versions, the llama-server build and context size read from `/props`, and, when `--model-path` is given, the SHA-256 of the model file. An existing pin file is never overwritten; a differing pin set goes to `<out>.pins.2.json`, `<out>.pins.3.json`, and so on. To print the pins for the current environment:

```bash
python -m maf.pins --server http://127.0.0.1:8080 --model <path-to>/gemma-4-12b-it-qat-q4_0.gguf
```

> [!TIP]
> Model hashing is opt-in (`--model-path` on `maf.experiment`, `--model` on `maf.pins`) because hashing a multi-gigabyte GGUF is slow.

## Analysis

`maf.analyze` aggregates a results file into markdown tables (mean and sample standard deviation over episodes), grouped by fault kind or by fault stage.

```bash
python -m maf.analyze results/run.jsonl                    # fault kind x strategy
python -m maf.analyze results/run.jsonl --by stage         # fault stage x strategy
python -m maf.analyze results/run.jsonl --kind malicious --csv out.csv
```

`maf.verifier_stats` computes detection statistics (true and false positives and negatives, per stage and per seed) for `verify_all` and `verify_hybrid` from one or more `run.log` files. If a grid was run in pieces, pass every log; the counts are summed. For `verify_hybrid`, a final table splits the counts by checker (rules or LLM).

```bash
python -m maf.verifier_stats logs/<experiment-dir-a>/run.log logs/<experiment-dir-b>/run.log \
    --strategy verify_hybrid --csv verifier.csv
```

`maf.tradeoff` writes `tradeoff.csv` and `tradeoff.svg`, plotting per-strategy cost (tokens, latency or LLM calls) against irreversible damage, with Pareto marks. `maf.compare` diffs two results files episode by episode to measure run-to-run variability.

```bash
python -m maf.tradeoff results/run.jsonl --x tokens --by kind --out-dir charts
python -m maf.compare results/run.jsonl results/run_rep2.jsonl
```

## Measuring tracker accuracy with the dependency oracle

At temperature 0 with a fixed sampling seed, a fault episode under `none` differs from the fault-free episode of the same seed essentially only where the fault reaches. Diffing the two gives the true affected set. `maf.oracle` compares it with the closure from `Tracker.downstream()` for the same episode and reports precision (over-invalidation) and recall (under-invalidation). A change is scored two ways: strict, where the normalized artifact differs at all, and decision, where the task-relevant part differs (plan: supplier and quantity; execute: tool and arguments of every action; notify: supplier and cc). Only stages after the faulted one are scored, which also keeps run-to-run wording differences in upstream artifacts out of the true set.

`probe` runs "should not matter" perturbations under `none`. These should leave the true set empty, so any invalidation exposes over-invalidation. `analyze` reads `<results>.events.jsonl`, so it needs a trace.

```bash
python -m maf.oracle probe --backend mock --seeds 0-9 --out results/probes.jsonl
python -m maf.oracle analyze results/probes.jsonl
python -m maf.oracle analyze results/run.jsonl --csv oracle.csv --changes
```

## Micro-benchmarking the tracker

`maf.microbench` times `record_read` and `downstream()` on synthetic chain and DAG dependency graphs of increasing size, with shared-state edges on and off. With `--trace` it instead measures trace overhead on mock episodes, trace on versus off.

```bash
python -m maf.microbench --sizes 5,10,50,100,500,1000 --repeats 200 --out microbench.csv
python -m maf.microbench --trace --seeds 0-4
```

## Freezing a run

`scripts/freeze_results.sh` copies a finished run into `results/frozen/<name>/`, checks the copies are byte-identical, writes `SHA256SUMS` and makes everything read-only. For each results file it also copies `.events.jsonl`, `.pins.json` (and `.pins.N.json`) and `.traces.jsonl` when present. It refuses to overwrite an existing destination.

```bash
./scripts/freeze_results.sh my_run results/run.jsonl results/probes.jsonl
(cd results/frozen/my_run && sha256sum -c SHA256SUMS)
```

## Testing

```bash
python -m pytest -q
```

# Recovery strategies

Select strategies with `--strategies`. All strategies get the same fault schedule and the same inference budget ceilings.

> [!IMPORTANT]
> `full_restart`, `selective`, `selective_msg` and the gated twins are told which run is faulty, which separates recovery quality from detection quality. `verify_all` and `verify_hybrid` are not told and must detect the fault themselves with an extra verifier LLM call after each stage. Cost and safety comparisons across the two groups therefore mix recovery and detection effects.

## `none`

No recovery. Measures raw propagation of the fault through the workflow. An injected timeout aborts the episode.

## `full_restart`

Rolls back every reversible effect and reruns all five stages. Emails and transfers already sent cannot be undone, so the rerun may add a wrong or duplicate irreversible effect.

## `full_restart_gated`

`full_restart` with the outbox gate: emails and transfers are held and, at the end of recovery, released only for runs outside the invalidated set. Emails are idempotent on `(to, subject, body)`.

## `selective`

Uses the dependency tracker over message and shared-state edges to roll back and rerun only the faulty stage and its downstream invalidation closure. Runs outside the closure are kept.

## `selective_msg`

Ablation of `selective` that tracks message edges only and ignores shared-state reads, so it under-invalidates stages that consume policy through shared state.

## `selective_gated`

`selective` with the outbox gate.

## `verify_all`

After each stage, an extra LLM call checks the output against independent evidence. A rejected stage is rerun, up to a bounded number of verification attempts.

## `verify_hybrid`

`verify_all` preceded by a deterministic rule checker (`maf/rules.py`). The checker permits known tools only, forbids transfers, requires quantity and SKU to match the request, and allows email only to the chosen supplier. A rule rejection skips the LLM call. If the final attempt still violates the rules, the stage is committed with the offending actions or fields removed.

# Fault model

One fault is injected per episode, once, on the artifact a stage produced, after generation and before validation. Select kinds with `--kinds`.

## `wrong_result`

The stage returns an incorrect artifact, for example a wrong price or supplier. Applies to all five stages.

## `malicious`

The artifact carries an injected extra action: a fund transfer, an extra email recipient, or injection text on the path to `execute`. Applies to all five stages.

## `stale`

The stage returns the output it would have produced for an earlier, different snapshot of the world, chosen so that the artifact is invalid and changes the final answer. Applies to `research_pricing`, `research_policy` and `plan`.

## `timeout`

The stage never returns; a fixed time is charged to the episode. Applies to all five stages.

## Oracle probe kinds

Used only by `maf.oracle probe` and absent from the default grid. Each probe preserves the correct answer, so any invalidation it triggers is over-invalidation.

| Kind | Stage | Perturbation |
|---|---|---|
| `probe_price` | `research_pricing` | A fixed offset added to one offered price |
| `probe_notes` | `research_policy` | A neutral note appended to the policy |
| `probe_rationale` | `plan` | A neutral rationale appended to the plan |

# License

All rights reserved. No license is granted.
