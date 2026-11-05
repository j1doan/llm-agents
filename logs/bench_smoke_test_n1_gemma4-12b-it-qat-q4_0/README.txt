First live smoke test on the fixed server settings (-c 4096 -fa on -ctk/-ctv q8_0 --reasoning off -ngl 99).
Command: python -m maf.bench --model-label gemma4-12b-it-qat-q4_0 -n 1 --strategies none --out results/bench_calls.jsonl
(run before run logging existed, so only calls.jsonl survives: no run.log, config.json or summary.txt).
Result: seed 0 task_success 1/1, 0 JSON failures, decode ~31 tok/s, ~16 s per episode.
