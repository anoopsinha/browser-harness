#!/usr/bin/env python3
"""
Local web service that wraps Gemini CLI so a browser extension can trigger it.

Model A (thin trigger): the extension sends a natural-language prompt; Gemini CLI
does the actual browser work via the browser-harness skill (CDP into your Chrome).
This service does NOT touch the browser itself — it only relays a prompt to
`gemini -p` and returns the result.

Gemini specifics handled here:
  - runs `gemini -p <prompt> -y --output-format json` from an agent workspace
    (extension-service/agent-workspace/) that pins API-key auth via a local
    .gemini/settings.json and carries the browser-harness skill in GEMINI.md;
  - parses Gemini's JSON ({session_id, response, stats}), tolerating the banner
    lines Gemini prints before the JSON;
  - session continuity uses `-r latest` (Gemini resume takes latest/index, not a
    session UUID), so "continue this conversation" maps to the most recent run
    from the workspace.

SECURITY: this endpoint can run Gemini CLI in YOLO mode, which can execute shell
commands. It is gated by three things:
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
HOST = os.environ.get("SERVICE_HOST", "127.0.0.1")
PORT = int(os.environ.get("SERVICE_PORT", "8787"))
# Gemini runs from this workspace: it holds .gemini/settings.json (pins
# gemini-api-key auth, overriding a global oauth login) and GEMINI.md (the
# browser-harness skill as context). run.sh generates GEMINI.md on startup.
AGENT_WS = os.environ.get("AGENT_WORKSPACE", str(HERE / "agent-workspace"))
GEMINI_BIN = os.environ.get("GEMINI_BIN", "gemini")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "")  # empty → gemini's default
# Prepended to every prompt (Gemini has no --append-system-prompt; GEMINI.md also
# carries guidance, this is a short reinforcement).
SYSTEM_PREAMBLE = os.environ.get(
    "SERVICE_SYSTEM_PREAMBLE",
    "For any browser action (navigate, click, read a page, screenshot, scrape, "
    "fill a form), use the browser-harness tool against the user's already-running "
    "Chrome. Keep the final answer short; it is shown in a small popup.",
)
# Appended for console requests (body tab_policy == "single") to keep one tab.
CONSOLE_TAB_POLICY = os.environ.get(
    "SERVICE_CONSOLE_TAB_POLICY",
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
# Appended for accessibility-extension requests (body tab_policy == "active"):
# act on the user's current page, not a fresh working tab.
ACTIVE_TAB_POLICY = os.environ.get(
    "SERVICE_ACTIVE_TAB_POLICY",
    "Active-tab policy: operate ONLY on the user's current tab — the page they "
    "are viewing right now. Do NOT open a new tab. Use browser-harness to locate "
    "that tab (list_tabs()/current_tab(), and switch_tab() to it if the harness "
    "is attached elsewhere), then do the task within it. Only open a new tab if "
    "the user explicitly asks for one.",
)
TIMEOUT_S = int(os.environ.get("SERVICE_TIMEOUT_S", "600"))


def load_gemini_key():
    """Take GEMINI_API_KEY from the repo .env as the source of truth (it may be
    fresher than a stale exported value)."""
    envf = HERE.parent / ".env"
    if not envf.exists():
        return
    for line in envf.read_text().splitlines():
        line = line.strip()
        if line.startswith("GEMINI_API_KEY="):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            if v:
                os.environ["GEMINI_API_KEY"] = v
            return


def load_or_create_token() -> str:
    tok = os.environ.get("SERVICE_TOKEN")
    if tok:
        return tok
    tok_file = HERE / ".token"
    if tok_file.exists():
        return tok_file.read_text().strip()
    tok = secrets.token_urlsafe(24)
    tok_file.write_text(tok + "\n")
    tok_file.chmod(0o600)
    return tok


load_gemini_key()
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


def _extract_json(text):
    """Gemini prints banner/startup lines before the JSON object; scan for the
    first '{' that decodes into a dict shaped like a Gemini result."""
    dec = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = dec.raw_decode(text, i)
            if isinstance(obj, dict) and ("response" in obj or "session_id" in obj):
                return obj
        except json.JSONDecodeError:
            pass
        i = text.find("{", i + 1)
    return None


def _num_turns(stats) -> int:
    try:
        return sum(
            m.get("api", {}).get("totalRequests", 0)
            for m in (stats or {}).get("models", {}).values()
        ) or None
    except Exception:
        return None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "gemini-extension-service"})


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

    tab_policy = body.get("tab_policy")
    active_url = (body.get("active_url") or "").strip()

    parts = []
    if SYSTEM_PREAMBLE:
        parts.append(SYSTEM_PREAMBLE)
    if tab_policy == "single" and CONSOLE_TAB_POLICY:
        parts.append(CONSOLE_TAB_POLICY)
    elif tab_policy == "active" and ACTIVE_TAB_POLICY:
        policy = ACTIVE_TAB_POLICY
        if active_url:
            policy += f"\nThe user's current tab URL is: {active_url}"
        parts.append(policy)
    parts.append(prompt)
    full_prompt = "\n\n".join(parts)

    cmd = [GEMINI_BIN, "-p", full_prompt, "-y", "--output-format", "json"]
    if GEMINI_MODEL:
        cmd += ["-m", GEMINI_MODEL]
    if session:
        cmd += ["-r", "latest"]  # Gemini resume: latest/index, not a UUID

    try:
        proc = subprocess.run(
            cmd, cwd=AGENT_WS, capture_output=True, text=True, timeout=TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return _cors(
            jsonify({"ok": False, "error": f"gemini timed out after {TIMEOUT_S}s"}),
            origin,
        ), 504

    data = _extract_json(proc.stdout) or _extract_json(proc.stdout + "\n" + proc.stderr)
    if data is None:
        return _cors(
            jsonify({
                "ok": False,
                "error": "gemini produced no parseable result",
                "raw": proc.stdout[-1500:],
                "stderr": proc.stderr[-1500:],
            }),
            origin,
        ), 500

    resp = data.get("response")
    out = {
        "ok": resp is not None,
        "result": resp,
        "session_id": data.get("session_id"),
        "num_turns": _num_turns(data.get("stats")),
        "cost_usd": None,  # Gemini CLI JSON reports tokens, not a dollar cost
    }
    return _cors(jsonify(out), origin), (200 if out["ok"] else 500)


if __name__ == "__main__":
    line = "=" * 64
    key_state = "set" if os.environ.get("GEMINI_API_KEY") else "MISSING"
    print(line)
    print("  gemini-extension-service")
    print(f"  listening   http://{HOST}:{PORT}")
    print(f"  token       {TOKEN}")
    print(f"  workspace   {AGENT_WS}")
    print(f"  model       {GEMINI_MODEL or '(gemini default)'}")
    print(f"  GEMINI_API_KEY {key_state}")
    print("  SECURITY: token-gated, localhost-only. Anyone with the token can")
    print("            run Gemini CLI in YOLO mode (shell exec). Keep it private.")
    print(line)
    app.run(host=HOST, port=PORT, threaded=True)
