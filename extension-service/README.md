# gemini-extension-service

A tiny local web service that lets a **browser extension trigger Gemini CLI**,
which in turn drives your real Chrome through the **browser-harness** skill.

This is the **Model A (thin trigger)** design: the extension is just a UI. It
sends a natural-language prompt to `http://127.0.0.1:8787/run`; the service
shells out to `gemini -p`; Gemini does the browser work via browser-harness
(CDP into your already-running Chrome) and returns a text result.

```
Extension popup ──HTTP(+token)──▶ Flask (127.0.0.1) ──▶ gemini -p ──▶ browser-harness ──CDP──▶ your Chrome
```

## Prerequisites

- `gemini` CLI on PATH (Google Gemini CLI).
- A **`GEMINI_API_KEY`** (Google AI Studio) in the repo `.env` — the service
  loads it and pins `gemini-api-key` auth via `agent-workspace/.gemini/settings.json`
  (so it works even if your global Gemini login is on a deprecated tier). Your
  global `~/.gemini/settings.json` is left untouched.
- The browser-harness skill: the service generates `agent-workspace/GEMINI.md`
  from `browser-harness skill` on each start, so Gemini knows how to drive it.
- Chrome running with remote debugging (same setup browser-harness already uses).

## 1. Start the service

```bash
./run.sh
```

First run creates a `.venv`, installs Flask, loads `GEMINI_API_KEY` from `.env`,
regenerates `agent-workspace/GEMINI.md`, and starts the server. It prints a
**token** (also saved to `extension-service/.token`). Copy that token.

Config is via env vars (all optional): `SERVICE_PORT`, `SERVICE_TOKEN`,
`GEMINI_MODEL`, `AGENT_WORKSPACE`, `SERVICE_TIMEOUT_S`, `SERVICE_SYSTEM_PREAMBLE`.

> **Session note:** Gemini resume takes `latest`/index (not a session UUID), so
> "continue this conversation" maps to `gemini -r latest` — the most recent run
> from the agent workspace. Fine for one sequential conversation at a time.

## 2. Point the extension at your token

Edit `extension/config.js` and paste the token into `TOKEN`.

## 3. Load the extension

1. Open `chrome://extensions`.
2. Enable **Developer mode** (top right).
3. **Load unpacked** → select the `extension/` folder.
4. Pin the extension; click its icon to open the popup.

Type a prompt, hit **Run** (or ⌘/Ctrl+Enter). "include page context" attaches
the active tab's URL/title and any selected text. **New** starts a fresh
conversation; otherwise follow-ups continue the same Gemini session.

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
Gemini session until you `:new`.

### Voice mode (optional)

The console has hands-free voice I/O, all native browser APIs (Web Speech +
Web Audio), off by default — typing is unaffected. Click **🎙 voice** in the bar,
or just press the talk hotkey:

- **Ctrl+M** — toggle listening: press to start talking, press again to stop and
  send. First press turns voice mode on. Pressing it also **interrupts** any reply
  that's currently being spoken.
- **Esc** — stop listening without sending; also interrupts speech.

When voice mode is on: replies are **spoken** (interruptible), and a soft
**earcon pulses while the agent is thinking** so you know to wait, with a chime on
completion and a buzz on error. Requires Chrome and microphone permission (granted
on first use; `127.0.0.1` is a secure context so the mic is allowed).

While you're speaking a command, any **audio/video playing in the browser's tabs
is paused** (so it doesn't overlap your voice) and **resumes once the spoken reply
finishes**. This runs through browser-harness (the console page can't reach other
tabs' media directly), via `POST /api/media/pause` and `/api/media/resume` on the
console server. Only media in same-origin page DOM is paused — media inside
cross-origin iframes (e.g. embedded YouTube players) can't be reached this way.

## Security — read this

This endpoint runs Gemini CLI in **YOLO mode** (`-y`), which auto-approves tool
calls including **shell execution**. It is protected by three things; do not
weaken them:

1. **Localhost only.** Bound to `127.0.0.1`. Never bind to `0.0.0.0` or expose the port.
2. **Bearer token.** Treat `.token` like a password. It is gitignored.
3. **Origin check.** Only `chrome-extension://` and localhost origins are accepted.

CORS does not stop a malicious page from *sending* a request to localhost, so the
token is the real gate. To shrink the blast radius you can constrain what the
agent may run via Gemini's policy engine (`--policy` / `--approval-mode`) instead
of blanket `-y` — start permissive to confirm it works, then tighten.

## Upgrade paths (later)

- **Live progress:** switch `/run` to Server-Sent Events by reading
  `gemini -p --output-format stream-json` line-by-line and forwarding events.
- **Scoped tools:** replace `-y` with a Gemini policy that only allows the
  `browser-harness` shell command, for a tighter blast radius.

## Files

| File | Purpose |
|------|---------|
| `server.py` | Flask service; `/health` and `/run`. |
| `run.sh` | Create venv + start server. |
| `requirements.txt` | Flask. |
| `extension/manifest.json` | MV3 manifest (popup + localhost host permission). |
| `extension/popup.html` / `popup.js` | The popup UI and fetch logic. |
| `extension/config.js` | Endpoint + token (you edit this). |
