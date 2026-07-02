# claude-extension-service

A tiny local web service that lets a **browser extension trigger Claude Code**,
which in turn drives your real Chrome through the **browser-harness** skill.

This is the **Model A (thin trigger)** design: the extension is just a UI. It
sends a natural-language prompt to `http://127.0.0.1:8787/run`; the service
shells out to `claude -p`; Claude does the browser work via browser-harness
(CDP into your already-running Chrome) and returns a text result.

```
Extension popup ──HTTP(+token)──▶ Flask (127.0.0.1) ──▶ claude -p ──▶ browser-harness ──CDP──▶ your Chrome
```

## Prerequisites

- `claude` CLI on PATH and already logged in (headless `claude -p` reuses your
  existing login — no API key needed).
- The browser-harness skill wired into Claude Code (it is, via your global
  `~/.claude/CLAUDE.md`).
- Chrome running with remote debugging (same setup browser-harness already uses).

## 1. Start the service

```bash
./run.sh
```

First run creates a `.venv`, installs Flask, and starts the server. It prints a
**token** (also saved to `extension-service/.token`). Copy that token.

Config is via env vars (all optional): `CLAUDE_SERVICE_PORT`,
`CLAUDE_SERVICE_TOKEN`, `CLAUDE_ALLOWED_TOOLS`, `CLAUDE_PERMISSION_MODE`,
`CLAUDE_SERVICE_CWD`, `CLAUDE_TIMEOUT_S`, `CLAUDE_MAX_TURNS`.

## 2. Point the extension at your token

Edit `extension/config.js` and paste the token into `TOKEN`.

## 3. Load the extension

1. Open `chrome://extensions`.
2. Enable **Developer mode** (top right).
3. **Load unpacked** → select the `extension/` folder.
4. Pin the extension; click its icon to open the popup.

Type a prompt, hit **Run** (or ⌘/Ctrl+Enter). "include page context" attaches
the active tab's URL/title and any selected text. **New** starts a fresh
conversation; otherwise follow-ups continue the same Claude session.

## Try it without the extension

```bash
TOKEN=$(cat .token)
curl -s http://127.0.0.1:8787/run \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Open example.com in a new tab and tell me the page title."}' | jq
```

Response shape: `{ ok, result, session_id, num_turns, cost_usd, permission_denials }`.
Pass `"session": "<session_id>"` on the next call to continue the conversation.

## Verify it's running (web console)

A terminal-style web console for smoke-testing the service lives in `console/`.
It runs on its own port (8788) and calls the service **directly** — cross-origin,
with the bearer token and an Origin header — exactly like the extension does, so
it exercises the full public contract (CORS, Origin allowlist, auth). It auto-loads
the token from `.token`, so there's nothing to paste.

```bash
cd console && ./run.sh          # reuses the service's venv; serves http://127.0.0.1:8788
```

Open http://127.0.0.1:8788, type a prompt, hit Enter. The status dot (top-left)
shows whether the service is up. Console commands: `:new` (fresh conversation),
`:health`, `:session`, `:clear`, `:help`. Follow-up prompts continue the same
Claude session until you `:new`.

## Security — read this

This endpoint can run Claude Code, which can execute **Bash and edit files**.
It is protected by three things; do not weaken them:

1. **Localhost only.** Bound to `127.0.0.1`. Never bind to `0.0.0.0` or expose the port.
2. **Bearer token.** Treat `.token` like a password. It is gitignored.
3. **Origin check.** Only `chrome-extension://` and localhost origins are accepted.

CORS does not stop a malicious page from *sending* a request to localhost, so the
token is the real gate. To shrink the blast radius, tighten the allowed tools —
e.g. restrict Bash to the harness:

```bash
CLAUDE_ALLOWED_TOOLS='Bash(browser-harness*),Read' ./run.sh
```

(Start permissive to confirm it works, then tighten.) Avoid
`--dangerously-skip-permissions`; this service intentionally does not use it.

## Upgrade paths (later)

- **Live progress:** switch `/run` to Server-Sent Events by reading
  `claude -p --output-format stream-json` line-by-line and forwarding events.
- **Embed the SDK:** replace the subprocess with `claude-agent-sdk` on an async
  server (FastAPI/Quart) for first-class streaming and session management.

## Files

| File | Purpose |
|------|---------|
| `server.py` | Flask service; `/health` and `/run`. |
| `run.sh` | Create venv + start server. |
| `requirements.txt` | Flask. |
| `extension/manifest.json` | MV3 manifest (popup + localhost host permission). |
| `extension/popup.html` / `popup.js` | The popup UI and fetch logic. |
| `extension/config.js` | Endpoint + token (you edit this). |
