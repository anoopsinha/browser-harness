// Toolkit datastore layer -- taxonomy (globalThis.AA_TAXONOMY) and the
// generated built-in tools registry (globalThis.AA_TOOLS) must load before
// datastore.js, which exposes both via Datastore.global.*.
self.importScripts(
  'lib/taxonomy.js',
  'lib/tools-registry.js',
  'lib/datastore.js',
  'lib/librarian.js'
);

// Lazy, idempotent store migrations. Safe to fire-and-forget: stores are
// readable before this resolves (migration 1 is a stamp).
Datastore.runMigrations().catch((e) =>
  console.warn('[AgenticA11y] datastore migrations failed:', e.message));

const GEMINI_MODEL = 'gemini-3.1-flash-image-preview';

function getApiUrl(apiKey, model) {
  return `https://generativelanguage.googleapis.com/v1beta/models/${model || GEMINI_MODEL}:generateContent?key=${apiKey}`;
}

// callGemini supports two third-arg shapes for backward compatibility:
//   - an array of image data URLs (legacy from main: multimodal vision calls)
//   - an options object: { images?: string[], mimeType?: string }
// New callers should use the object form; mimeType (e.g. 'application/json')
// asks Gemini to emit only valid JSON.
async function callGemini(prompt, apiKey, optsOrImages) {
  const opts = Array.isArray(optsOrImages)
    ? { images: optsOrImages }
    : (optsOrImages || {});
  const { images, mimeType, model } = opts;

  const parts = [{ text: prompt }];
  if (images && images.length > 0) {
    for (const dataUrl of images) {
      const match = dataUrl.match(/^data:(.+?);base64,(.+)$/);
      if (match) {
        parts.push({
          inlineData: { mimeType: match[1], data: match[2] }
        });
      }
    }
  }

  const generationConfig = { temperature: 0.7 };
  if (mimeType) generationConfig.responseMimeType = mimeType;

  const resp = await fetch(getApiUrl(apiKey, model), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts }],
      generationConfig,
    })
  });
  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`Gemini API error ${resp.status}: ${err}`);
  }
  const data = await resp.json();
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) throw new Error(`Gemini returned no text: ${JSON.stringify(data)}`);
  return text;
}

async function getApiKey() {
  const data = await chrome.storage.sync.get(['geminiApiKey', 'geminiKey']);
  return data.geminiApiKey || data.geminiKey || null;
}

// The Librarian's slow lane (extraction, reflection, playbooks) uses the
// same key-resolving caller.
if (globalThis.Librarian) {
  globalThis.Librarian.setGeminiCaller(async (prompt) => {
    const key = await getApiKey();
    if (!key) throw new Error('No Gemini API key configured.');
    return await callGemini(prompt, key);
  });
}

// Observe explicit setting toggles as memory signal. One listener instead
// of instrumenting every popup control: any sync-area change to a known
// tool setting is a deliberate user action (weight 3) — the popup and profile
// "Apply" buttons all write through here.
const OBSERVED_SETTING_KEYS = new Set([
  'darkMode', 'readerMode', 'keyboardNav', 'voiceCommands', 'motionReducer', 'focusMode',
  'hideDistractions', 'showProgress', 'colorBlindMode', 'fontScale', 'lineHeight',
  'letterSpacing', 'contrastMode', 'dyslexiaFont', 'largeCursor', 'enhanceFocus',
  'readingGuide', 'speechRate', 'autoWcagFix', 'autoDescribe', 'autoSimplify',
  'autoSummarize', 'autoFixLabels', 'autoCaptions', 'autoVideoDescribe',
]);
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== 'sync' || !globalThis.Librarian) return;
  const changed = Object.entries(changes).filter(([k]) => OBSERVED_SETTING_KEYS.has(k));
  if (!changed.length) return;
  (async () => {
    let origin = null;
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab?.url) origin = new URL(tab.url).hostname;
    } catch {}
    for (const [key, { oldValue, newValue }] of changed) {
      // Fast lane: make the change stick immediately (final say in the
      // effective-preferences merge); the episodic observation below is the
      // slow-lane signal for extraction/reflection.
      await globalThis.Librarian.recordExplicitSetting(key, newValue, origin).catch(() => {});
      await globalThis.Librarian.logObservation({
        type: 'setting-change',
        origin,
        text: `User changed setting ${key} from ${JSON.stringify(oldValue)} to ${JSON.stringify(newValue)}`,
        data: { key, oldValue, newValue },
      }).catch(() => {});
    }
  })();
});

// Shared Gemini dispatch: page/content contexts message this to run a
// key-resolved Gemini call.
function handleGeminiMessage(msg, sender, sendResponse) {
  (async () => {
    const callerId = sender?.id || sender?.url || 'unknown';
    try {
      const apiKey = msg.apiKey || await getApiKey();
      if (!apiKey) {
        console.log('[AgenticA11y] gemini call from', callerId, '→ no API key');
        sendResponse({ error: 'No Gemini API key configured. Go to extension settings.' });
        return;
      }
      const result = await callGemini(msg.prompt, apiKey, {
        images: msg.images,
        mimeType: msg.mimeType,
        model: msg.model,
      });
      console.log('[AgenticA11y] gemini call from', callerId, '→ result length:', result.length);
      sendResponse({ result });
    } catch (e) {
      console.log('[AgenticA11y] gemini call from', callerId, '→ error:', e.message);
      sendResponse({ error: e.message });
    }
  })();
  return true;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'gemini') {
    return handleGeminiMessage(msg, sender, sendResponse);
  }

  if (msg.type === 'saveApiKey') {
    chrome.storage.sync.set({ geminiApiKey: msg.apiKey }, () => {
      sendResponse({ success: true });
    });
    return true;
  }

  if (msg.type === 'getApiKey') {
    getApiKey().then(key => sendResponse({ apiKey: key }));
    return true;
  }

  if (msg.type === 'getActiveSkills') {
    chrome.storage.local.get(['activeSkills', 'customSkills'], (data) => {
      sendResponse({
        activeSkills: data.activeSkills || [],
        customSkills: data.customSkills || []
      });
    });
    return true;
  }

  if (msg.type === 'setActiveSkills') {
    chrome.storage.local.set({ activeSkills: msg.skills }, () => {
      sendResponse({ success: true });
    });
    return true;
  }

  if (msg.type === 'getUserProfile') {
    chrome.storage.local.get('userProfile', (data) => {
      sendResponse({ profile: data.userProfile || null });
    });
    return true;
  }

  if (msg.type === 'saveUserProfile') {
    chrome.storage.local.set({ userProfile: msg.profile }, () => {
      sendResponse({ success: true });
    });
    return true;
  }

  // --- AI Support: interpret natural language needs ---
  if (msg.type === 'interpretNeeds') {
    // The Librarian builds the prompt from the global tools registry (the
    // "does this exist in the global db?" check is grounded in AA_TOOLS, not a
    // duplicated list) and conditions it on the ability profile. Response
    // shape is unchanged for the popup.
    (async () => {
      try {
        const apiKey = await getApiKey();
        if (!apiKey) { sendResponse({ error: 'No API key' }); return; }
        const prompt = await globalThis.Librarian.interpretNeedsPrompt(msg.text);
        const result = await callGemini(prompt, apiKey);
        sendResponse({ result });
      } catch (e) {
        sendResponse({ error: e.message });
      }
    })();
    return true;
  }

  // --- Custom Profile CRUD ---
  if (msg.type === 'saveCustomProfile') {
    chrome.storage.local.get('customProfiles', (data) => {
      const profiles = data.customProfiles || [];
      const existing = profiles.findIndex(p => p.id === msg.profile.id);
      if (existing >= 0) profiles[existing] = msg.profile;
      else profiles.push(msg.profile);
      chrome.storage.local.set({ customProfiles: profiles }, () => {
        sendResponse({ success: true });
      });
    });
    return true;
  }

  if (msg.type === 'getCustomProfiles') {
    chrome.storage.local.get('customProfiles', (data) => {
      sendResponse({ profiles: data.customProfiles || [] });
    });
    return true;
  }

  if (msg.type === 'deleteCustomProfile') {
    chrome.storage.local.get('customProfiles', (data) => {
      const profiles = (data.customProfiles || []).filter(p => p.id !== msg.id);
      chrome.storage.local.set({ customProfiles: profiles }, () => {
        sendResponse({ success: true });
      });
    });
    return true;
  }

  // --- Site Classification ---
  // Delegates to the Librarian's site index: hostMap/TLD first, then a
  // one-time Gemini classification cached per origin (no more re-classifying
  // the same host on every visit). User overrides are sticky.
  if (msg.type === 'classifySite') {
    (async () => {
      const hostname = msg.hostname || '';
      const title = msg.title || '';

      let siteType = null;
      try {
        siteType = await globalThis.Librarian.getSiteCategory(hostname, { allowLlm: true, title });
      } catch (e) {
        console.warn('[AgenticA11y] classify failed:', e.message);
      }

      if (!siteType) siteType = 'other';

      const { customProfiles } = await chrome.storage.local.get('customProfiles');
      const profiles = customProfiles || [];
      const matching = profiles.find(p => p.autoApply && p.siteTypes?.includes(siteType));

      sendResponse({ siteType, matchingProfile: matching || null });
    })();
    return true;
  }

  // --- Librarian (personal memory/profile agent) ---
  // Fast lane: deterministic queries + mechanical writes. The *Now variants
  // exist for debugging the (now setTimeout-debounced) slow lane.
  if (msg.type && msg.type.startsWith('librarian')) {
    const L = globalThis.Librarian;
    if (!L) { sendResponse({ error: 'librarian not loaded' }); return false; }
    (async () => {
      try {
        switch (msg.type) {
          case 'librarianGetProfile':
            sendResponse({ profile: await L.getProfile() }); break;
          case 'librarianSetProfileField':
            sendResponse({ profile: await L.setProfileField(msg.path, msg.value) }); break;
          case 'librarianRecordScopedSettings':
            sendResponse({ ids: await L.recordScopedSettings(msg.scope, msg.settings || {}, msg.opts || {}) }); break;
          case 'librarianGetSiteCategory':
            sendResponse({ category: await L.getSiteCategory(msg.origin, msg.opts || {}) }); break;
          case 'librarianSetSiteCategory':
            await L.setSiteCategoryOverride(msg.origin, msg.category);
            sendResponse({ success: true }); break;
          case 'librarianEffectivePreferences':
            sendResponse(await L.getEffectivePreferences(msg.url, msg.contexts || [])); break;
          case 'librarianRecall':
            sendResponse(await L.recall(msg.url, msg.task || '', msg.contexts || [])); break;
          case 'librarianListMemories':
            sendResponse(await L.listMemories(msg.filter || {})); break;
          case 'librarianListProposals':
            sendResponse({ proposals: await L.listProposals(msg.status ?? 'pending') }); break;
          case 'librarianLogObservation':
            sendResponse(await L.logObservation(msg.observation || {})); break;
          case 'librarianRespondToProposal':
            sendResponse(await L.respondToProposal(msg.id, msg.response)); break;
          case 'librarianDeleteMemory':
            sendResponse({ success: await L.deleteMemory(msg.id) }); break;
          case 'librarianSetPause':
            if (msg.origin) await L.setOriginPaused(msg.origin, msg.paused);
            else await L.setMemoryPaused(msg.paused);
            sendResponse({ success: true }); break;
          case 'librarianExtractNow':
            sendResponse(await L.extract()); break;
          case 'librarianReflectNow':
            sendResponse(await L.reflect()); break;
          default:
            sendResponse({ error: `unknown librarian message: ${msg.type}` });
        }
      } catch (e) {
        sendResponse({ error: e.message });
      }
    })();
    return true;
  }
});
