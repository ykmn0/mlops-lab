#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
if [ -x "./venv/bin/python" ]; then
  PYTHON_BIN="./venv/bin/python"
fi

rm -rf mlruns mlflow.db __pycache__ tests/__pycache__

"$PYTHON_BIN" train.py
"$PYTHON_BIN" -m pytest -q

"$PYTHON_BIN" -m uvicorn app:app --host 127.0.0.1 --port 8000 &
UVICORN_PID=$!
trap 'if [ -n "${UVICORN_PID:-}" ] && kill -0 "$UVICORN_PID" 2>/dev/null; then kill "$UVICORN_PID"; fi' EXIT

READY=0
for _ in {1..20}; do
  if curl -sf http://127.0.0.1:8000/health >/dev/null; then
    READY=1
    break
  fi
  sleep 1
done

test "${READY:-0}" = "1"
curl -sf http://127.0.0.1:8000/health | grep -q '"status":"ok"'
echo "HEALTH OK"

curl -sf -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}' | grep -q "prediction"
echo "PREDICT OK"
