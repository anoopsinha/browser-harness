#!/usr/bin/env python3
"""
Standalone web console (terminal UI) to exercise the gemini-extension-service.

Runs on its own port (default 8788) and serves a page whose JavaScript calls the
service **directly** — cross-origin, with the bearer token and an Origin header —
exactly like the real browser extension does. So this verifies the service's full
public contract (CORS, Origin allowlist, auth), not a server-side shortcut.

The token is read from ../.token and injected into the page, so there's nothing
to paste. Only /api/health is proxied (so the status dot works without needing
CORS on the service's /health).
"""
import json
import os
import subprocess
import urllib.request
from pathlib import Path

from flask import Flask, Response, jsonify, send_from_directory

BH_BIN = os.environ.get("BROWSER_HARNESS_BIN", "browser-harness")

HERE = Path(__file__).resolve().parent
SERVICE_URL = os.environ.get("SERVICE_URL", "http://127.0.0.1:8787").rstrip("/")
HOST = os.environ.get("CONSOLE_HOST", "127.0.0.1")
PORT = int(os.environ.get("CONSOLE_PORT", "8788"))


def token() -> str:
    t = os.environ.get("SERVICE_TOKEN")
    if t:
        return t
    f = HERE.parent / ".token"
    return f.read_text().strip() if f.exists() else ""


app = Flask(__name__)


@app.after_request
def no_cache(resp):
    # console assets change during dev; never let the browser serve stale JS
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


ASSETS = {"app.js", "voice.js", "style.css"}


@app.route("/<path:fn>")
def asset(fn):
    if fn in ASSETS:
        return send_from_directory(HERE, fn)
    return ("not found", 404)


@app.route("/config.js")
def config():
    cfg = {"SERVICE_RUN": SERVICE_URL + "/run", "TOKEN": token()}
    return Response("window.CFG=" + json.dumps(cfg) + ";", mimetype="application/javascript")


@app.route("/api/health")
def health():
    # server-side check so the status dot works without CORS on the service /health
    try:
        with urllib.request.urlopen(SERVICE_URL + "/health", timeout=3) as r:
            return jsonify({"up": True, "service": json.loads(r.read().decode())})
    except Exception as e:
        return jsonify({"up": False, "error": str(e)})


# --- media pause/resume across the browser's tabs (for voice input) -------------
# The console page can't reach other tabs' media directly, so we drive it through
# browser-harness. Paused elements are tagged so resume only replays what we paused.
_MEDIA_PAUSE = r"""
import json
paused = 0
for t in list_tabs(include_chrome=False):
    if ":8788" in t.get("url", ""):  # skip the console tab itself
        continue
    try:
        n = js("(()=>{let n=0;document.querySelectorAll('video,audio').forEach(m=>{if(!m.paused&&!m.ended){m.setAttribute('data-hp-paused','1');m.pause();n++;}});return n;})()", target_id=t["target_id"])
        paused += int(n or 0)
    except Exception:
        pass
print(json.dumps({"paused": paused}))
"""

_MEDIA_RESUME = r"""
import json
resumed = 0
for t in list_tabs(include_chrome=False):
    try:
        n = js("(()=>{let n=0;document.querySelectorAll('[data-hp-paused]').forEach(m=>{m.removeAttribute('data-hp-paused');try{const p=m.play();if(p&&p.catch)p.catch(()=>{});}catch(e){}n++;});return n;})()", target_id=t["target_id"])
        resumed += int(n or 0)
    except Exception:
        pass
print(json.dumps({"resumed": resumed}))
"""


def _run_bh(snippet):
    try:
        p = subprocess.run([BH_BIN], input=snippet, capture_output=True, text=True, timeout=30)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    for line in reversed(p.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
                d["ok"] = True
                return d
            except Exception:
                break
    return {"ok": False, "stderr": p.stderr[-300:], "raw": p.stdout[-300:]}


@app.route("/api/media/pause", methods=["POST"])
def media_pause():
    return jsonify(_run_bh(_MEDIA_PAUSE))


@app.route("/api/media/resume", methods=["POST"])
def media_resume():
    return jsonify(_run_bh(_MEDIA_RESUME))


if __name__ == "__main__":
    print(f"console  → http://{HOST}:{PORT}   (targets {SERVICE_URL})")
    if not token():
        print("WARNING: no token (../.token missing and SERVICE_TOKEN unset)")
    app.run(host=HOST, port=PORT, threaded=True)
