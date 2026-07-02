#!/usr/bin/env python3
"""
Standalone web console (terminal UI) to exercise the claude-extension-service.

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
import urllib.request
from pathlib import Path

from flask import Flask, Response, jsonify, send_from_directory

HERE = Path(__file__).resolve().parent
SERVICE_URL = os.environ.get("SERVICE_URL", "http://127.0.0.1:8787").rstrip("/")
HOST = os.environ.get("CONSOLE_HOST", "127.0.0.1")
PORT = int(os.environ.get("CONSOLE_PORT", "8788"))


def token() -> str:
    t = os.environ.get("CLAUDE_SERVICE_TOKEN")
    if t:
        return t
    f = HERE.parent / ".token"
    return f.read_text().strip() if f.exists() else ""


app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/app.js")
def appjs():
    return send_from_directory(HERE, "app.js")


@app.route("/style.css")
def css():
    return send_from_directory(HERE, "style.css")


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


if __name__ == "__main__":
    print(f"console  → http://{HOST}:{PORT}   (targets {SERVICE_URL})")
    if not token():
        print("WARNING: no token (../.token missing and CLAUDE_SERVICE_TOKEN unset)")
    app.run(host=HOST, port=PORT, threaded=True)
