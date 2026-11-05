#!/usr/bin/env bash
# Freeze a finished run: copy its results file(s) and log folder into results/frozen/<name>/, check the copies
# are byte-identical, write SHA256SUMS and make everything read-only. Run from the project root (Git Bash). Example:
#   ./scripts/freeze_results.sh phase1_v2 results/phase1_v2_dev.jsonl results/phase1_v2_probes.jsonl results/phase1_v2_charts
# A results file X also brings along X.events.jsonl, X.pins.json (and X.pins.N.json) and the old-style X.traces.jsonl
# when they exist. Copies keep their timestamps. The originals are left in place
# (maf.analyze only reads them); never pass a frozen or original results file as --out again, the runner appends.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,8p' "$0"
  exit 1
fi

NAME="$1"; shift
DST="results/frozen/$NAME"
if [[ -e "$DST" ]]; then
  echo "Refusing to overwrite existing $DST" >&2
  exit 1
fi

# collect sources first so nothing is copied if one is missing
SRC=()
for p in "$@"; do
  if [[ ! -e "$p" ]]; then echo "Not found: $p" >&2; exit 1; fi
  SRC+=("$p")
  if [[ -f "$p" ]]; then
    for c in "$p.events.jsonl" "$p.traces.jsonl" "$p.pins.json" "$p".pins.[0-9]*.json; do
      if [[ -f "$c" ]]; then SRC+=("$c"); fi
    done
  fi
done

mkdir -p "$DST"
for p in "${SRC[@]}"; do
  cp -rp "$p" "$DST/"
  if ! diff -r "$p" "$DST/$(basename "$p")" > /dev/null; then
    echo "Copy differs from original: $p" >&2
    exit 1
  fi
done

(cd "$DST" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
find "$DST" -type f -exec chmod a-w {} +

echo "Frozen into $DST (read-only):"
cat "$DST/SHA256SUMS"
echo
echo "Record these hashes in docs/VERSIONS.md. Verify later with: (cd $DST && sha256sum -c SHA256SUMS)"
