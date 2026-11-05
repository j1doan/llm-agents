#!/usr/bin/env bash
# Start llama-server for the experiments. Run from the project root (Git Bash on Windows). Example:
#   ./scripts/serve_llama.sh                          # default: Gemma 4 12B-it int4 QAT (dense), 4k ctx, q8 KV, no thinking
#   ./scripts/serve_llama.sh --ngl 40 --ctx 8192
#   ./scripts/serve_llama.sh --kv-quant f16 --reasoning auto
# The 12B Q4_0 file is ~7 GB, so with 99 GPU layers it barely fits in 8 GB VRAM. If VRAM is full the NVIDIA driver
# may silently spill into system RAM (decode drops to ~6 tok/s at 100% GPU util): keep --ctx small, keep the KV cache
# quantised, or lower --ngl (each layer moved to the CPU costs speed). Check `nvidia-smi` for ~0.5 GB of headroom.
# --reasoning off stops Gemma from spending the token budget on a thinking block before the JSON answer.
# Pin the llama.cpp build and GGUF hash in your notes for reproducibility.
# Flags track recent llama.cpp builds; check `llama-server --help` if one is rejected.
# Every option can also be set through the environment, e.g. LLAMA_DIR=... ./scripts/serve_llama.sh
set -euo pipefail

MODEL="${MODEL:-D:/huggingface/google/gemma-4-12b-it/gemma-4-12b-it-qat-q4_0.gguf}"
LLAMA_DIR="${LLAMA_DIR:-D:/llama-b10926}"
PORT="${PORT:-8080}"
CTX="${CTX:-4096}"
NGL="${NGL:-99}"
PARALLEL="${PARALLEL:-1}"
FA="${FA:-on}"
KV_QUANT="${KV_QUANT:-q8_0}"
REASONING="${REASONING:-off}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --llama-dir) LLAMA_DIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --ctx) CTX="$2"; shift 2 ;;
    --ngl) NGL="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --fa) FA="$2"; shift 2 ;;
    --kv-quant) KV_QUANT="$2"; shift 2 ;;
    --reasoning) REASONING="$2"; shift 2 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL (fetch with ../download-gemma_4_12b_it.sh)" >&2
  exit 1
fi

exec "$LLAMA_DIR/llama-server.exe" -m "$MODEL" --host 127.0.0.1 --port "$PORT" -c "$CTX" -ngl "$NGL" \
  --parallel "$PARALLEL" --jinja -fa "$FA" -ctk "$KV_QUANT" -ctv "$KV_QUANT" --reasoning "$REASONING"
