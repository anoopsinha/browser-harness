const CFG = window.CFG || {};
const screen = document.getElementById("screen");
const cmd = document.getElementById("cmd");
const dot = document.getElementById("dot");
const sessionEl = document.getElementById("session");
document.getElementById("target").textContent = "→ " + (CFG.SERVICE_RUN || "?");

let session = null;
const history = [];
let histIdx = 0;

function el(cls, text) {
  const d = document.createElement("div");
  d.className = cls;
  if (text !== undefined) d.textContent = text;
  screen.appendChild(d);
  screen.scrollTop = screen.scrollHeight;
  return d;
}

function echo(text) {
  const d = el("line echo");
  const ps1 = document.createElement("span");
  ps1.className = "ps1";
  ps1.textContent = "harness$ ";
  d.appendChild(ps1);
  d.appendChild(document.createTextNode(text));
  screen.scrollTop = screen.scrollHeight;
}

function setSession(id) {
  session = id;
  sessionEl.textContent = id ? "session " + id.slice(0, 8) + "…" : "no session";
}

function fmtDur(ms) {
  const s = Math.round(ms / 1000);
  return s < 60 ? s + "s" : Math.floor(s / 60) + "m" + String(s % 60).padStart(2, "0") + "s";
}

async function checkHealth() {
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    dot.className = d.up ? "up" : "down";
    dot.title = d.up ? "service up" : "service down: " + (d.error || "");
  } catch (_) {
    dot.className = "down";
    dot.title = "console cannot reach its own /api/health";
  }
}

async function runPrompt(prompt) {
  if (!CFG.TOKEN) {
    el("err", "No token loaded. Is ../.token present and the service running?");
    return;
  }
  if (window.Voice) window.Voice.onThinkingStart();
  const running = el("running", "· running…");
  const t0 = performance.now();
  const timer = setInterval(() => {
    running.textContent = "· running… " + fmtDur(performance.now() - t0);
  }, 1000);

  try {
    const res = await fetch(CFG.SERVICE_RUN, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + CFG.TOKEN,
      },
      // keep all console commands in one dedicated working tab (server policy)
      body: JSON.stringify({ prompt, session: session || undefined, tab_policy: "single" }),
    });
    const data = await res.json().catch(() => ({}));
    clearInterval(timer);
    running.remove();

    if (!res.ok || !data.ok) {
      const msg = data.error || data.result || "HTTP " + res.status;
      el("err", "✗ " + msg);
      if (data.stderr) el("meta", data.stderr);
      if (window.Voice) window.Voice.onError(msg);
      return;
    }
    el("result", data.result ?? "(no text result)");
    if (window.Voice) window.Voice.onResult(data.result ?? "");
    if (data.session_id) setSession(data.session_id);
    const bits = [
      (data.num_turns ?? "?") + " turns",
      data.cost_usd != null ? "$" + data.cost_usd.toFixed(4) : null,
      fmtDur(performance.now() - t0),
      data.session_id ? "session " + data.session_id.slice(0, 8) : null,
    ].filter(Boolean);
    el("meta", bits.join("  ·  "));
  } catch (e) {
    clearInterval(timer);
    running.remove();
    el("err", "✗ cannot reach the service at " + CFG.SERVICE_RUN + " — is server.py running? (" + e.message + ")");
    dot.className = "down";
    if (window.Voice) window.Voice.onError(e.message);
  } finally {
    if (window.Voice) window.Voice.onThinkingStop();
  }
}

const HELP = `commands:
  <text>      send <text> to Gemini via the service (continues the session)
  :new        start a fresh conversation (drops the session id)
  :health     re-check service status
  :say [text] speak a phrase (test text-to-speech)
  :session    show the current session id
  :clear      clear the screen
  :help       this help`;

function handle(input) {
  const s = input.trim();
  if (!s) return;
  echo(s);
  if (s === ":help") return void el("sys", HELP);
  if (s === ":clear") return void (screen.innerHTML = "");
  if (s === ":new") {
    setSession(null);
    return void el("sys", "started a new conversation");
  }
  if (s === ":session") {
    return void el("sys", session ? session : "(no session yet)");
  }
  if (s === ":health") {
    el("sys", "checking…");
    return void checkHealth();
  }
  if (s === ":say" || s.startsWith(":say ")) {
    const t = s.slice(4).trim();
    if (window.Voice && window.Voice.say) {
      window.Voice.say(t);
      el("sys", "speaking a test phrase… (Esc to stop)");
    } else el("err", "voice module not loaded");
    return;
  }
  if (s === ":voicediag") {
    if (window.Voice && window.Voice.diag) window.Voice.diag();
    else el("err", "voice module not loaded");
    return;
  }
  if (s === ":voicereset") {
    if (window.Voice && window.Voice.reset) window.Voice.reset();
    else el("err", "voice module not loaded");
    return;
  }
  if (s.startsWith(":")) return void el("err", "unknown command: " + s + "  (try :help)");
  runPrompt(s);
}

cmd.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    const v = cmd.value;
    cmd.value = "";
    if (v.trim()) {
      history.push(v);
      histIdx = history.length;
    }
    handle(v);
  } else if (e.key === "ArrowUp") {
    if (histIdx > 0) cmd.value = history[--histIdx];
    e.preventDefault();
  } else if (e.key === "ArrowDown") {
    if (histIdx < history.length - 1) cmd.value = history[++histIdx];
    else {
      histIdx = history.length;
      cmd.value = "";
    }
    e.preventDefault();
  }
});

document.addEventListener("click", (e) => {
  // don't steal focus from the bar controls (voice select / buttons)
  if (e.target.closest && e.target.closest("#bar")) return;
  cmd.focus();
});

// let voice.js submit recognized speech + print diagnostics through the console
window.Console = { submit: handle, log: (m) => el("sys", m) };

// boot
el("sys", "gemini-extension-service console — type :help. Enter to send. Voice: Ctrl+M to talk.");
checkHealth();
setInterval(checkHealth, 15000);
setSession(null);
