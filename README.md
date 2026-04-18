# bu

The simplest, most powerful browser agent harness: connect to a real browser, keep setup tiny, and let the agent do the rest.

## Setup

Humans should not learn a CLI here. The normal setup is to paste the **setup prompt** into Claude Code (or any coding agent) and let it install the repo, wire up Chrome, and propose the first task.

### Setup prompt

```text
Set up the `bu` browser harness for me.

1. Clone https://github.com/browser-use/harnessless, `cd harnessless`, run `uv sync`.
2. Read `SKILL.md` end-to-end before anything else — it documents the full
   workflow, tool shapes, gotchas, and every helper you'll use.
3. On macOS, open Chrome to the remote-debugging page directly:
     open -a "Google Chrome" "chrome://inspect/#remote-debugging"
   (Linux/Windows: tell me that URL and have me open it.)
   Tell me to tick the "Discover network targets" / remote-debugging checkbox
   and accept Chrome's "Allow debugging" dialog. Wait until I confirm.
4. Verify the connection with a single call:
     uv run run.py <<<'print(page_info())'
   (The daemon auto-starts on first call.)
5. Navigate my real browser to https://github.com/browser-use/harnessless.
6. Propose your first task: **"Star the browser-use/harnessless repo on GitHub."**
   Run it after I confirm.
```

### Manual install

```bash
git clone https://github.com/browser-use/harnessless && cd harnessless
uv sync
# enable Chrome remote debugging at chrome://inspect/#remote-debugging
```

## Example tasks

```text
Star the browser-use/harnessless repo on GitHub.
Scrape every job posting on the current LinkedIn search page.
Log into my Salesforce sandbox and export the Accounts list as CSV.
Post a thread on X summarising the last 5 GitHub issues I opened.
```

The agent reads `SKILL.md`, writes plain Python against `helpers.py`, and if
something isn't covered it drops to raw CDP or edits a helper in place.

## Features

- **Anything a browser can do.** Helpers are thin; under them is raw CDP
  (`cdp("Domain.method", **params)`), so there's no "API surface" to hit the
  wall of. The agent can also edit `helpers.py` mid-task to add what it needs —
  the harness gets sharper every session.
- **Tiny, readable.** `daemon.py` 200 lines, `helpers.py` 266 lines, `run.py` 4
  lines, `SKILL.md` 107 lines. ~580 total. One pass and you've read everything.
- **Remote browsers via Browser Use cloud.** Set `BROWSER_USE_API_KEY` in
  `.env`, call `start_remote_daemon("work")`, and you get an isolated cloud
  Chrome with a live-view URL. Great for parallel sub-agents — each sub-agent
  gets its own cloud browser, its own CDP session, and its own live URL:

  ```python
  # spin up N isolated cloud browsers for N sub-agents
  for name in ("scraper-1", "scraper-2", "scraper-3"):
      print(start_remote_daemon(name)["liveUrl"])
  # then each sub-agent runs: BU_NAME=scraper-1 uv run run.py <<<'...'
  ```
- **Passes through iframes / shadow DOM / cross-origin.** `click(x, y)` goes via
  the compositor, so Azure blades, Salesforce consoles, and Stripe checkouts
  work without selector gymnastics.
- **Bulk HTTP fallback.** For static pages, skip the browser entirely with
  `http_get()` + `ThreadPoolExecutor` — hundreds of pages in seconds.

## How it works

- `daemon.py` (200 lines) — holds the CDP WebSocket, relays over a Unix socket
  at `/tmp/bu-<name>.sock`. One daemon per `BU_NAME`.
- `helpers.py` (266 lines) — ~25 small functions (`goto`, `click`, `js`,
  `cdp`, `screenshot`, `upload_file`, …), each about 5 lines. Edit any of them.
- `run.py` (4 lines) — imports helpers, ensures the daemon, execs stdin as Python.
- `SKILL.md` (107 lines) — how an agent *uses* the harness.
- `AGENTS.md` — how an agent *modifies* the harness.

No CLI, no DSL, no wrapper API. Just Python + CDP.

## Parallel / remote

`BU_NAME` picks the daemon (default `default`). Each name = independent socket,
independent daemon, independent browser (local or cloud).

```bash
BU_NAME=work uv run run.py <<< 'goto("https://example.com"); print(page_info())'
uv run python -c "from helpers import kill_daemon; kill_daemon('work')"
```

## Stop

```bash
uv run python -c "from helpers import kill_daemon; kill_daemon()"        # default
uv run python -c "from helpers import kill_daemon; kill_daemon('work')"  # named (also stops remote browser)
```
