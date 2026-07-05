# Accessibility Combine — Plan

Combine the **adaptive-accessibility core** of `AI-for-Accessibility-Toolkit-Draft/personalized-extension`
with the local **extension-service** (Gemini CLI + browser-harness), producing one
lean, all-Gemini MV3 extension: it adapts pages to the user, and hands agentic
browser tasks to the service.

Status: **plan**. Branch: `accessibility-combine` (off `accessibility-extension`).

---

## Goal

A single browser extension that:
1. **Adapts every page** to the user (font/contrast/dark/reader/TTS/simplify/alt-text…) — the accessibility layer, kept from the toolkit.
2. **Delegates agentic browser tasks** ("fill this form", "turn on captions here", "book a table") to the **extension-service** — the "Assistant", now powered by Gemini CLI + the real browser-harness instead of the toolkit's bundled JS harness.

Everything is **Gemini** end to end.

---

## Locked decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Personalization depth | **Librarian-lite** — keep the profile + per-site preference resolution (scope chain); drop the learning loop (episodic log, proposals, reflection, alarms). |
| 2 | Fast per-page AI path | **Direct Gemini API from the extension** (as the toolkit already does). Adapters keep working even if the service is off. |
| 3 | v1 scope | **Extract + strip + adapters working first** (lean accessibility extension on Gemini, no Assistant). Assistant→service is phase 2. |
| 4 | Location | New top-level **`accessibility-extension/`** dir in this repo. The old `extension-service/extension/` simple-trigger popup folds into it (one front-end). |
| 5 | Onboarding | **Minimal** profile setup (support areas + free-text → seed profile). Drop the full recommend-and-build wizard. |
| — | JS browser-harness | **Removed.** Replaced by the extension-service. |

---

## Target architecture

```
┌─ accessibility-extension (MV3, the front) ──────────────────────┐
│  content.js + 18 builtin adapters   → adapts the page           │
│  Librarian-lite (profile + per-site prefs)                      │
│  popup (settings / "describe your needs")                       │
│                                                                 │
│  fast per-page AI ─────────┐            agentic browser task ───┐│
└────────────────────────────┼───────────────────────────────────┼┘
                             ▼                                    ▼
              Gemini API (direct, in-extension)      extension-service /run
              alt-text, simplify, labels…            (gemini -p + browser-harness)
              cheap, no server needed                = the "Assistant" (phase 2)
```

Two AI paths, both Gemini, split by the *shape* of the work:
- **Fast per-page adapters** → direct Gemini API in the extension (cheap, resilient).
- **Agentic Assistant** → extension-service (where the real browser-harness lives).

---

## What we keep vs. strip

**Keep (~6–7k LOC — the actual adaptive-accessibility behavior):**
- `extension/content/content.js` (the applier)
- `skills/registry.js` + `skills/builtin/*.js` (18 adapters — the DOM adaptations)
- `utils/{dom.js, constants.js, color.js, ai.js}` (helpers + the AI seam)
- `extension/lib/{datastore.js, taxonomy.js, librarian.js, tools-registry.js}` — Librarian **trimmed to the fast lane** (see below)
- `extension/background.js` — **heavily stripped** (AI dispatch, settings→Librarian observation, classifySite, profile CRUD, librarian handlers)
- `extension/popup/*` — slimmed settings UI
- `build.js` — reduced to generate `tools-registry.js` + bundle only `content.js`

**Remove:**
- `extension/browser-harness/**` — the JS harness + Gemini agent loop (replaced by the service)
- `extension/offscreen/**`, `extension/sidepanel/**`, `extension/permission/**` — the Gemini-Live voice subsystem
- `extension/skill-builder/**` — custom-adapter authoring (Engineer; a later phase)
- `extension/demo/**` + all demo-mode branches + `extension/lib/demo-trace.js`
- `extension/onboarding/**` → replace with a minimal profile setup; drop `utils/recommender.js`
- `skill-creator/**`, `test/**`, `scripts/**` — not runtime

**Librarian-lite** = keep the deterministic fast lane of `librarian.js`:
- `getEffectivePreferences(url, contexts)` — scope-chain merge (general → context → category → origin)
- site classification cache (host-map + Gemini fallback)
- recording explicit settings as scoped preferences
Drop the slow lane: `extract()`, `reflect()`, episodic log, consent-gated proposals, `chrome.alarms`.

---

## Manifest simplification

Removing the JS harness (and voice/skill-builder) sheds the scariest permissions.

- **Before:** `<all_urls>` + `activeTab, alarms, storage, scripting, userScripts, debugger, tabs, tabGroups, notifications, sidePanel, offscreen`
- **After (target):** `storage, scripting, activeTab` (+ `<all_urls>` host access for content-script adaptation). No `debugger`, `offscreen`, `sidePanel`, `notifications`, `tabGroups`, `userScripts`, `alarms`.

The extension no longer attaches a debugger — the **service** drives Chrome via the browser-harness daemon (needs the local service running + Chrome remote debugging, same as today).

---

## The integration seam (phase 2 — the actual "combine")

In `background.js`, the Assistant handlers (`bh`, `bhAgentStart`, `runProfileActions`) currently drive the bundled harness. **Replace them with an HTTP call to the extension-service `/run`** — the same token'd localhost client used by the console.

- `utils/ai.js` → **direct Gemini** stays as-is for the fast adapters.
- New `serviceClient.js`: `POST http://127.0.0.1:8787/run` with `Authorization: Bearer <token>`, body `{prompt, tab_policy, session?}`.
- **Tab policy nuance:** the console uses an *isolated* working tab; the accessibility Assistant usually needs to act on the **user's current page**. Add a `tab_policy: "active"` to the service (operate on the active tab, don't spawn a working tab). The per-request `tab_policy` flag already exists in `server.py`.

**Config (popup settings):**
- `GEMINI_API_KEY` — for in-extension fast adapters (as today).
- Service **base URL + token** — for the Assistant (read the service `.token`).

**Graceful degradation:** adapters (direct Gemini) always work; the Assistant needs the local service running. Surface this clearly in the UI.

---

## Phasing

**Phase 0 — Extract & strip (v1 foundation)**
- Create `accessibility-extension/`; copy the keep-set; delete the strip-set.
- Trim `background.js` (drop harness/agent/voice/demo/skill-builder branches).
- Slim the manifest to the target permission set.
- Reduce `build.js`; get `content.bundle.js` + `tools-registry.js` building.
- **Verify:** load unpacked; on a real page, confirm dark mode / contrast / reader / font scaling apply; confirm AI adapters (alt-text, simplify) work with a Gemini key.

**Phase 1 — Librarian-lite + minimal onboarding**
- Wire profile + per-site preference resolution; record explicit setting changes as scoped prefs.
- Minimal onboarding: support areas + free-text → seed `mine.profile`.
- **Verify:** set a preference on one site; confirm it re-applies there and follows the category scope; confirm it does *not* leak to unrelated sites.

**Phase 2 — Assistant → extension-service**
- Add `tab_policy: "active"` to `server.py`.
- Add `serviceClient.js`; replace `bh`/`bhAgentStart`/`runProfileActions` with service calls.
- Popup config for service URL/token; degradation messaging when the service is down.
- **Verify:** from the extension, run an agentic task on the current page (e.g. "turn on captions") via the service; confirm it acts on the active tab.

**Phase 3 — Optional (later)**
- Engineer (adapter building) via the service.
- Add back full Librarian learning (proposals/memory) if wanted.
- Fold in the console's voice input as a hands-free control for the Assistant.

---

## Risks / caveats

- **Rate limits:** Gemini free tier ≈15 req/min; the agentic Assistant makes several calls per task. Consider model choice / batching; `GEMINI_MODEL` is configurable in the service.
- **Two config items** (Gemini key + service token) until/unless we route adapters through the service too. Kept separate for resilience.
- **Service dependency** for Assistant features only; adapters are independent.
- **Arbitrary code / permissive tools:** the service runs Gemini in YOLO mode (shell exec), gated by localhost + token. Keep those gates; consider a Gemini policy scoping tools to `browser-harness`.
- **`skills` naming:** the toolkit calls adapters "skills" in code identifiers; keep the internal names to minimize churn, but the user-facing word is "adapter".

---

## Definition of done (v1 = phases 0–1)

A lean `accessibility-extension/` that loads unpacked with `storage, scripting, activeTab`, applies the built-in adapters (non-AI + Gemini-backed) to real pages, remembers per-site preferences via Librarian-lite, and has no bundled JS harness / voice / demo / skill-builder. The Assistant (phase 2) then routes to the Gemini extension-service.
