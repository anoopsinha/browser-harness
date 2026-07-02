#!/usr/bin/env bash
# Start the web console. Reuses the extension-service venv (which has Flask).
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$HERE"
PY="../.venv/bin/python"
if [ ! -x "$PY" ]; then
  if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
    ./.venv/bin/pip -q install flask
  fi
  PY=".venv/bin/python"
fi
exec "$PY" app.py
