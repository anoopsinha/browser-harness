#!/usr/bin/env bash
# Start the local Claude extension service. Creates a venv on first run.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$HERE"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  ./.venv/bin/pip -q install -r requirements.txt
fi
exec ./.venv/bin/python server.py
