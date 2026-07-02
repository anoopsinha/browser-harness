const $ = (id) => document.getElementById(id);
const cfg = window.CLAUDE_CFG || {};

let sessionId = null;

// Restore the conversation id so follow-up prompts keep context.
chrome.storage.local.get("sessionId").then((v) => {
  sessionId = v.sessionId || null;
  renderMeta();
});

function renderMeta(extra = "") {
  $("meta").textContent =
    (sessionId ? `session ${sessionId.slice(0, 8)}…` : "new conversation") +
    (extra ? " · " + extra : "");
}

async function getContext() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  let selection = "";
  try {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => window.getSelection().toString(),
    });
    selection = result || "";
  } catch (_) {
    // restricted page (chrome://, web store, etc.) — no selection available
  }
  return { url: tab?.url || "", title: tab?.title || "", selection };
}

function buildPrompt(text, ctx) {
  if (!ctx) return text;
  let p = `${text}\n\n[Browser context]\nActive tab: ${ctx.title} — ${ctx.url}`;
  if (ctx.selection) p += `\nSelected text:\n${ctx.selection.slice(0, 4000)}`;
  return p;
}

async function run() {
  const text = $("prompt").value.trim();
  if (!text) return;
  if (!cfg.TOKEN || cfg.TOKEN === "PASTE_TOKEN_HERE") {
    showError("Set TOKEN in config.js (see the server startup output).");
    return;
  }

  $("run").disabled = true;
  $("status").textContent = "running…";
  $("out").textContent = "";
  $("out").classList.remove("err");

  try {
    const ctx = $("ctx").checked ? await getContext() : null;
    const prompt = buildPrompt(text, ctx);

    const res = await fetch(cfg.ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + cfg.TOKEN,
      },
      body: JSON.stringify({ prompt, session: sessionId || undefined }),
    });
    const data = await res.json();

    if (!res.ok || !data.ok) {
      showError(data.error || data.result || `HTTP ${res.status}`);
      return;
    }

    $("out").textContent = data.result ?? "(no text result)";
    if (data.session_id) {
      sessionId = data.session_id;
      await chrome.storage.local.set({ sessionId });
    }
    const cost = data.cost_usd != null ? `$${data.cost_usd.toFixed(4)}` : "";
    renderMeta([`${data.num_turns ?? "?"} turns`, cost].filter(Boolean).join(" · "));
  } catch (e) {
    showError("Could not reach the service. Is server.py running? " + e.message);
  } finally {
    $("run").disabled = false;
    $("status").textContent = "";
  }
}

function showError(msg) {
  const out = $("out");
  out.classList.add("err");
  out.textContent = msg;
}

$("run").addEventListener("click", run);
$("prompt").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
});
$("newconv").addEventListener("click", async () => {
  sessionId = null;
  await chrome.storage.local.remove("sessionId");
  $("out").textContent = "";
  renderMeta("cleared");
});
