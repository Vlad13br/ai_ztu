#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://localhost:11434/v1}"

if ! curl -s -m 2 "$BASE_URL/models" > /dev/null; then
  echo "⚠️  Рантайм на $BASE_URL не відповідає. Запустіть: ollama serve"
fi

echo "Відкрийте http://localhost:$PORT  (Ctrl+C  зупинити)"
( sleep 1; command -v open > /dev/null && open "http://localhost:$PORT" ) &
exec python3 -m http.server "$PORT" --bind 127.0.0.1
