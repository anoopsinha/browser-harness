# Running the Accessibility Stack

Machine-readable instructions for starting the extension-service, the web console, and the Chrome extension. All paths are relative to the repo root.

## Prerequisites

- `.env` at the repo root with `GEMINI_API_KEY=<key>` (required by the LLM backend in `api.py`).
- Node.js + npm (for the extension build).
- Python (the service `run.sh` creates/reuses its own venv; the console reuses it).

## 1. Start the extension-service (port 8787)

```bash
cd extension-service
nohup ./run.sh > server.log 2>&1 &
```

Verify:

```bash
lsof -tiTCP:8787 -sTCP:LISTEN   # prints a pid when up
tail extension-service/server.log
```

The service prints its auth token on startup; it is also written to `extension-service/.token`.

## 2. Start the console (port 8788)

Reuses the service's venv. Start the service first.

```bash
cd extension-service/console
nohup ./run.sh > console.log 2>&1 &
```

Verify:

```bash
lsof -tiTCP:8788 -sTCP:LISTEN
```

Open: http://127.0.0.1:8788 (targets http://127.0.0.1:8787).

## 3. Build + load the Chrome extension

```bash
cd accessibility-extension
npm install
npm run build
```

Load in Chrome (first time):

1. `chrome://extensions` → enable **Developer mode**
2. **Load unpacked** → select `accessibility-extension/extension`
3. First-time config only: extension popup → **Service settings** → paste the token from `extension-service/.token`

After a rebuild, hit **Reload** on the extension in `chrome://extensions`.

## Stopping

Stop by port (surgical):

```bash
lsof -tiTCP:8787 -sTCP:LISTEN | xargs kill   # extension-service
lsof -tiTCP:8788 -sTCP:LISTEN | xargs kill   # console
```

Or by process name:

```bash
pkill -f server.py
```

## Logs

- Service: `extension-service/server.log`
- Console: `extension-service/console/console.log`
