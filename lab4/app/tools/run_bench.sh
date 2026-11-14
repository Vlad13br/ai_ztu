#!/usr/bin/env bash
# Обгортка над bench.py: пише результати саме в lab4/bench_results.csv (як вимагає структура здачі).
#
#   ./tools/run_bench.sh before qwen3:4b-instruct-2507-q4_K_M
#   ./tools/run_bench.sh after  qwen3:4b-instruct-2507-q8_0
#   BASE_URL=http://localhost:8080/v1 ./tools/run_bench.sh after-llamacpp local     # llama-server
#   CONCURRENCY="1 2 4" ./tools/run_bench.sh before qwen3:4b-instruct-2507-q4_K_M  # інші рівні паралельності
#
# Модель без режиму міркування (…-instruct-2507) приймає reasoning_effort і так; для гібридних (qwen3:4b)
# він обовʼязковий, інакше TTFT включить міркування. Вимкнути: REASONING="" ./tools/run_bench.sh ...
set -euo pipefail
cd "$(dirname "$0")"
LABEL="${1:?мітка, напр. before}"
MODEL="${2:?назва моделі на сервері}"
BASE_URL="${BASE_URL:-http://localhost:11434/v1}"
CONCURRENCY="${CONCURRENCY:-1 2 4 8}"
REASONING="${REASONING-none}"
OUT="$(cd ../.. && pwd)/bench_results.csv"

PY=(python3)
if ! python3 -c "import openai" 2> /dev/null; then
  if command -v uv > /dev/null; then PY=(uv run --with openai python); else
    echo "Встановіть пакет openai: python3 -m pip install openai (або встановіть uv)"; exit 1; fi
fi

EXTRA=()
[ -n "$REASONING" ] && EXTRA+=(--reasoning-effort "$REASONING")

# shellcheck disable=SC2086
"${PY[@]}" bench.py --base-url "$BASE_URL" --model "$MODEL" --label "$LABEL" \
  --concurrency $CONCURRENCY --output "$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
