#!/usr/bin/env bash
# Start the local Gemini extension service. Creates a venv on first run.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$HERE"

# venv
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  ./.venv/bin/pip -q install -r requirements.txt
fi

# Load GEMINI_API_KEY from the repo .env (source of truth) so the gemini
# subprocess authenticates with the key, not a stale/OAuth login.
ENVF="$HERE/../.env"
if [ -f "$ENVF" ]; then
  KEY="$(awk -F= '/^GEMINI_API_KEY=/{sub(/^GEMINI_API_KEY=/,"");gsub(/["'"'"' ]/,"");print;exit}' "$ENVF")"
  [ -n "${KEY:-}" ] && export GEMINI_API_KEY="$KEY"
fi
[ -z "${GEMINI_API_KEY:-}" ] && echo "WARNING: GEMINI_API_KEY not set (repo .env or env)" >&2

# Regenerate the agent workspace's GEMINI.md (browser-harness skill as context).
WS="$HERE/agent-workspace"
mkdir -p "$WS"
{
  echo "# Agent instructions"
  echo
  echo "You are a browser-automation agent driving the user's already-running Chrome."
  echo "For ANY browser action (navigate, click, read, screenshot, scrape, fill forms),"
  echo "use the \`browser-harness\` tool: run the \`browser-harness\` shell command with a"
  echo "Python heredoc. Keep final answers short and plain."
  echo
  browser-harness skill 2>/dev/null || echo "(browser-harness skill text unavailable at startup.)"
} > "$WS/GEMINI.md"

exec ./.venv/bin/python server.py
