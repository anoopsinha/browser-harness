#!/usr/bin/env python3
"""
Local web service that wraps Claude Code so a browser extension can trigger it.

Model A (thin trigger): the extension sends a natural-language prompt; Claude Code
does the actual browser work via the browser-harness skill (CDP into your Chrome).
This service does NOT touch the browser itself — it only relays a prompt to
`claude -p` and returns the result.

SECURITY: this endpoint can run Claude Code, which can execute Bash and edit
files. It is gated by three things:
  1. bound to 127.0.0.1 only (never exposed off-box),
  2. a bearer token (treat it like a password),
  3. an Origin check (chrome-extension:// / localhost only).
Do not weaken these or expose the port. See README.md.
"""
import hmac
import json
import os
import secrets
import subprocess
from pathlib import Path

from flask import Flask, request, jsonify, make_response

HERE = Path(__file__).resolve().parent

# ---- config (all env-overridable) ----
HOST = os.environ.get("CLAUDE_SERVICE_HOST", "127.0.0.1")
PORT = int(os.environ.get("CLAUDE_SERVICE_PORT", "8787"))
# Run claude from the repo root by default so it has sane project context and the
# local `browser-harness` wrapper is on hand. The skill itself loads globally.
WORKDIR = os.environ.get("CLAUDE_SERVICE_CWD", str(HERE.parent))
PERMISSION_MODE = os.environ.get("CLAUDE_PERMISSION_MODE", "acceptEdits")
# Tools Claude may use without prompting. Bash is what the browser-harness skill
# uses to drive Chrome. Tighten this to reduce blast radius (see README).
ALLOWED_TOOLS = os.environ.get(
    "CLAUDE_ALLOWED_TOOLS", "Bash,Read,Write,Edit,Glob,Grep,WebFetch"
)
SYSTEM_APPEND = os.environ.get(
    "CLAUDE_SYSTEM_APPEND",
    "You are being driven from a browser extension. For any browser action "
    "(navigate, click, read a page, screenshot, scrape, fill a form), use the "
    "browser-harness skill against the user's already-running Chrome. Keep the "
    "final answer short; it is shown in a small popup.",
)
# Appended for console requests (body tab_policy == "single") to keep one tab.
CONSOLE_TAB_POLICY = os.environ.get(
    "CLAUDE_CONSOLE_TAB_POLICY",
    "Single-working-tab policy: you drive the user's Chrome for a terminal "
    "console at http://127.0.0.1:8788 — never act on that console tab or the "
    "user's unrelated tabs. Keep ONE dedicated working tab and reuse it for "
    "every command in this conversation. Start by checking the current tab with "
    "page_info(). If you have not opened a working tab yet (the current tab is "
    "the console at :8788, a chrome:// page, or an unrelated page), open exactly "
    "ONE new tab with new_tab() and treat it as the working tab. Otherwise the "
    "current tab IS your working tab: navigate and act within it, do NOT open "
    "another tab. Open an additional tab only if the user explicitly asks for a "
    "new tab. Never open more than one tab per command.",
)
MAX_TURNS = os.environ.get("CLAUDE_MAX_TURNS", "50")
TIMEOUT_S = int(os.environ.get("CLAUDE_TIMEOUT_S", "600"))


def load_or_create_token() -> str:
    tok = os.environ.get("CLAUDE_SERVICE_TOKEN")
    if tok:
        return tok
    tok_file = HERE / ".token"
    if tok_file.exists():
        return tok_file.read_text().strip()
    tok = secrets.token_urlsafe(24)
    tok_file.write_text(tok + "\n")
    tok_file.chmod(0o600)
    return tok


TOKEN = load_or_create_token()
app = Flask(__name__)


def _cors(resp, origin):
    resp.headers["Access-Control-Allow-Origin"] = origin or "*"
    resp.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    resp.headers["Vary"] = "Origin"
    return resp


def _origin_ok(origin: str) -> bool:
    if not origin:
        return True  # non-browser callers (e.g. curl); token is still required
    return (
        origin.startswith("chrome-extension://")
        or origin.startswith("http://127.0.0.1")
        or origin.startswith("http://localhost")
    )


def _auth_ok(req) -> bool:
    auth = req.headers.get("Authorization", "")
    prefix = "Bearer "
    if not auth.startswith(prefix):
        return False
    return hmac.compare_digest(auth[len(prefix):], TOKEN)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "claude-extension-service"})


@app.route("/run", methods=["OPTIONS"])
def run_preflight():
    origin = request.headers.get("Origin", "")
    return _cors(make_response("", 204), origin)


@app.route("/run", methods=["POST"])
def run():
    origin = request.headers.get("Origin", "")
    if not _origin_ok(origin):
        return _cors(jsonify({"ok": False, "error": "bad origin"}), origin), 403
    if not _auth_ok(request):
        return _cors(jsonify({"ok": False, "error": "unauthorized"}), origin), 401

    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return _cors(jsonify({"ok": False, "error": "missing prompt"}), origin), 400
    session = (body.get("session") or "").strip()
    allowed = body.get("allowed_tools") or ALLOWED_TOOLS

    system_append = SYSTEM_APPEND
    if body.get("tab_policy") == "single" and CONSOLE_TAB_POLICY:
        system_append = (system_append + "\n\n" + CONSOLE_TAB_POLICY).strip()

    cmd = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--permission-mode", PERMISSION_MODE,
        "--allowedTools", allowed,
        "--max-turns", str(MAX_TURNS),
    ]
    if system_append:
        cmd += ["--append-system-prompt", system_append]
    if session:
        cmd += ["--resume", session]

    try:
        proc = subprocess.run(
            cmd, cwd=WORKDIR, capture_output=True, text=True, timeout=TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return _cors(
            jsonify({"ok": False, "error": f"claude timed out after {TIMEOUT_S}s"}),
            origin,
        ), 504

    if proc.returncode != 0 and not proc.stdout:
        return _cors(
            jsonify({"ok": False, "error": "claude failed", "stderr": proc.stderr[-2000:]}),
            origin,
        ), 500

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return _cors(
            jsonify({
                "ok": False,
                "error": "unparseable claude output",
                "raw": proc.stdout[-2000:],
                "stderr": proc.stderr[-1000:],
            }),
            origin,
        ), 500

    out = {
        "ok": not data.get("is_error", False),
        "result": data.get("result"),
        "session_id": data.get("session_id"),
        "num_turns": data.get("num_turns"),
        "cost_usd": data.get("total_cost_usd"),
        "permission_denials": data.get("permission_denials"),
    }
    return _cors(jsonify(out), origin), (200 if out["ok"] else 500)


if __name__ == "__main__":
    line = "=" * 64
    print(line)
    print("  claude-extension-service")
    print(f"  listening   http://{HOST}:{PORT}")
    print(f"  token       {TOKEN}")
    print(f"  workdir     {WORKDIR}")
    print(f"  tools       {ALLOWED_TOOLS}")
    print(f"  perm mode   {PERMISSION_MODE}")
    print("  SECURITY: token-gated, localhost-only. Anyone with the token can")
    print("            run Claude Code (Bash + file edits). Keep it private.")
    print(line)
    app.run(host=HOST, port=PORT, threaded=True)
