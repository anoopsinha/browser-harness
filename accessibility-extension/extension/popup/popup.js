function setChecked(id, value) {
  const el = document.getElementById(id);
  if (el) el.checked = value;
}
function setValue(id, value) {
  const el = document.getElementById(id);
  if (el) el.value = value;
}

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'fixAdded') updateFixesPanel(message.stats, message.fixes);
  if (message.type === 'statusUpdate') updateStatus(message.status);
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== 'sync') return;
  for (const [key, { newValue }] of Object.entries(changes)) {
    const el = document.getElementById(key);
    if (el) {
      if (el.type === 'checkbox') el.checked = newValue;
      else if (el.type === 'range' || el.tagName === 'SELECT') el.value = newValue;
    }
  }
});

function updateStatus(status) {
  const dot = document.querySelector('.status-dot');
  const text = document.querySelector('.status span:last-child');
  if (dot) dot.classList.toggle('off', !status.active);
  if (text) text.textContent = status.text || (status.active ? 'Active' : 'Inactive');
}

// Build the save-profile modal's site-type checkboxes from the shared
// taxonomy (lib/taxonomy.js) so the vocabulary can't drift from the
// classifier's. Categories marked noMemoryDefault still appear — they gate
// the Librarian's observation logging (Phase 1), not profile auto-apply.
function renderSiteTypeGrid() {
  const grid = document.querySelector('.site-type-grid');
  if (!grid || !globalThis.AA_TAXONOMY) return;
  grid.textContent = '';
  for (const cat of AA_TAXONOMY.categories) {
    const label = document.createElement('label');
    label.className = 'site-type-checkbox';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = cat.id;
    label.appendChild(cb);
    label.appendChild(document.createTextNode(' ' + cat.label));
    grid.appendChild(label);
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  renderSiteTypeGrid();
  const settings = await chrome.storage.sync.get([
    'enabled', 'autoWcagFix', 'autoDescribe', 'autoSimplify', 'autoSummarize',
    'autoFixLabels', 'autoCaptions', 'autoVideoDescribe',
    'darkMode', 'readerMode', 'keyboardNav', 'voiceCommands', 'motionReducer', 'focusMode',
    'hideDistractions', 'showProgress', 'colorBlindMode',
    'fontScale', 'lineHeight', 'letterSpacing', 'contrastMode',
    'dyslexiaFont', 'largeCursor', 'enhanceFocus', 'readingGuide', 'speechRate',
    'geminiKey', 'selectedProfiles', 'onboardingComplete', 'nudgeDismissed'
  ]);

  // Overlay the Librarian's effective preferences for the CURRENT tab so the
  // controls reflect what's actually applied on this page — e.g. a "150% on
  // news sites" scoped preference — instead of only the global baseline. The
  // values share the popup's units (fontScale %, lineHeight multiplier, …).
  // `scopedKeys` tracks which controls are showing a site-scoped value so a
  // later change writes it back to the right scope, not the global baseline.
  const currentTabUrl = await (async () => {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      return /^https?:/.test(tab?.url || '') ? tab.url : null;
    } catch { return null; }
  })();
  // effProvenance: setting key -> scope its shown value came from. A change to
  // a key whose value is site-scoped is written back to that scope.
  let effProvenance = {};
  if (currentTabUrl) {
    const eff = await new Promise((resolve) => {
      try {
        chrome.runtime.sendMessage(
          { type: 'librarianEffectivePreferences', url: currentTabUrl, contexts: [] },
          (r) => { void chrome.runtime.lastError; resolve(r || null); });
      } catch { resolve(null); }
    });
    if (eff?.settings) {
      Object.assign(settings, eff.settings);
      effProvenance = eff.provenance || {};
    }
  }
  // Persist a settings change to the scope its current value belongs to: if the
  // value is site-scoped (category:/origin:), update that Librarian record;
  // otherwise write the global baseline. Keeps "150% on news sites" from being
  // overwritten globally when the user nudges the slider on a news page.
  async function persistSetting(key, value) {
    const scope = effProvenance[key];
    if (scope && (scope.startsWith('category:') || scope.startsWith('origin:'))) {
      await new Promise((resolve) => {
        try {
          chrome.runtime.sendMessage(
            { type: 'librarianRecordScopedSettings', scope, settings: { [key]: value } },
            () => { void chrome.runtime.lastError; resolve(); });
        } catch { resolve(); }
      });
    } else {
      await chrome.storage.sync.set({ [key]: value });
      recordNonProfileSettings({ [key]: value });
    }
  }

  // ---- Layered setting sources --------------------------------------------
  // Settings arrive from three places: the base-profile checkboxes, typed AI
  // suggestions, and manual toggles/sliders (incl. saved custom profiles).
  // `nonProfileSettings` (storage.sync) remembers the last value set by any
  // NON-checkbox source, so unchecking a base profile restores that value
  // instead of blowing the setting away.
  function recordNonProfileSettings(partial) {
    if (!partial || !Object.keys(partial).length) return;
    chrome.storage.sync.get(['nonProfileSettings']).then(({ nonProfileSettings }) => {
      chrome.storage.sync.set({ nonProfileSettings: { ...(nonProfileSettings || {}), ...partial } });
    });
  }

  const fontScale = document.getElementById('fontScale');
  const fontScaleValue = document.getElementById('fontScaleValue');
  const lineHeight = document.getElementById('lineHeight');
  const lineHeightValue = document.getElementById('lineHeightValue');
  const letterSpacing = document.getElementById('letterSpacing');
  const letterSpacingValue = document.getElementById('letterSpacingValue');
  const focusModeEl = document.getElementById('focusMode');
  const focusOptions = document.getElementById('focusModeOptions');

  // Main toggle
  const mainToggle = document.getElementById('mainToggle');
  mainToggle.checked = settings.enabled !== false;
  mainToggle.addEventListener('change', async (e) => {
    await chrome.storage.sync.set({ enabled: e.target.checked });
    sendToContent({ type: 'setEnabled', enabled: e.target.checked });
  });

  // AI feature toggles
  const aiDefaults = {
    autoWcagFix: true, autoDescribe: true, autoSimplify: false,
    autoSummarize: false, autoFixLabels: true, autoVideoDescribe: false, autoCaptions: false
  };

  Object.entries(aiDefaults).forEach(([id, defaultVal]) => {
    const el = document.getElementById(id);
    if (el) {
      el.checked = defaultVal ? settings[id] !== false : settings[id] === true;
      el.addEventListener('change', async (e) => {
        await chrome.storage.sync.set({ [id]: e.target.checked });
        recordNonProfileSettings({ [id]: e.target.checked });
        sendToContent({ type: 'settingsChanged', settings: { [id]: e.target.checked } });
      });
    }
  });

  // Simple tool toggles
  const simpleTools = {
    darkMode: 'DarkMode',
    readerMode: 'ReaderMode',
    keyboardNav: 'KeyboardNavigator',
    motionReducer: 'MotionReducer'
  };

  // Voice Commands → the harness Voice Assistant side panel (voice mode for
  // this extension), NOT the page-level speech-recognition adapter. Launcher
  // semantics: checking it opens the panel and the box doesn't stay checked.
  const vcToggle = document.getElementById('voiceCommands');
  if (vcToggle) {
    vcToggle.checked = false;
    vcToggle.addEventListener('change', (e) => {
      if (e.target.checked) { openVoicePanel(); e.target.checked = false; }
    });
  }
  // One-time cleanup: pre-rewire storage may hold voiceCommands=true, which
  // used to restart page-level listening on every page load.
  if (settings.voiceCommands) chrome.storage.sync.set({ voiceCommands: false });

  Object.entries(simpleTools).forEach(([id, toolName]) => {
    const el = document.getElementById(id);
    if (el) {
      el.checked = settings[id] === true;
      el.addEventListener('change', async (e) => {
        await chrome.storage.sync.set({ [id]: e.target.checked });
        recordNonProfileSettings({ [id]: e.target.checked });
        if (e.target.checked) sendToContent({ type: 'enableTool', tool: toolName });
        else sendToContent({ type: 'disableTool', tool: toolName });
      });
    }
  });

  // Focus Mode with sub-options
  focusModeEl.checked = settings.focusMode === true;
  if (settings.focusMode) focusOptions.classList.add('show');
  ['hideDistractions', 'showProgress'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.checked = settings[id] === true;
  });

  focusModeEl.addEventListener('change', async (e) => {
    await chrome.storage.sync.set({ focusMode: e.target.checked });
    recordNonProfileSettings({ focusMode: e.target.checked });
    focusOptions.classList.toggle('show', e.target.checked);
    if (e.target.checked) sendFocusModeUpdate();
    else sendToContent({ type: 'disableTool', tool: 'FocusMode' });
  });

  ['hideDistractions', 'showProgress'].forEach(id => {
    document.getElementById(id)?.addEventListener('change', async (e) => {
      await chrome.storage.sync.set({ [id]: e.target.checked });
      recordNonProfileSettings({ [id]: e.target.checked });
      if (focusModeEl.checked) sendFocusModeUpdate();
    });
  });

  function sendFocusModeUpdate() {
    sendToContent({
      type: 'enableTool', tool: 'FocusMode',
      options: {
        hideDistractions: document.getElementById('hideDistractions').checked,
        showProgress: document.getElementById('showProgress').checked
      }
    });
  }

  // Color blind mode
  const colorBlindEl = document.getElementById('colorBlindMode');
  if (settings.colorBlindMode) colorBlindEl.value = settings.colorBlindMode;
  colorBlindEl.addEventListener('change', async (e) => {
    await chrome.storage.sync.set({ colorBlindMode: e.target.value });
    if (e.target.value === 'none') sendToContent({ type: 'disableTool', tool: 'ColorBlindMode' });
    else sendToContent({ type: 'enableTool', tool: 'ColorBlindMode', options: e.target.value });
  });

  // Load Visual Assist settings from storage
  if (settings.fontScale !== undefined) { fontScale.value = settings.fontScale; fontScaleValue.textContent = settings.fontScale + '%'; }
  if (settings.lineHeight !== undefined) { lineHeight.value = settings.lineHeight; lineHeightValue.textContent = parseFloat(settings.lineHeight).toFixed(1); }
  if (settings.letterSpacing !== undefined) { letterSpacing.value = settings.letterSpacing; letterSpacingValue.textContent = parseFloat(settings.letterSpacing).toFixed(2) + 'em'; }
  if (settings.contrastMode) document.getElementById('contrastMode').value = settings.contrastMode;
  ['dyslexiaFont', 'largeCursor', 'enhanceFocus', 'readingGuide'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.checked = settings[id] === true;
  });
  if (settings.speechRate !== undefined) document.getElementById('speechRate').value = settings.speechRate;

  // Slider live updates
  fontScale.addEventListener('input', () => { fontScaleValue.textContent = fontScale.value + '%'; });
  lineHeight.addEventListener('input', () => { lineHeightValue.textContent = parseFloat(lineHeight.value).toFixed(1); });
  letterSpacing.addEventListener('input', () => { letterSpacingValue.textContent = parseFloat(letterSpacing.value).toFixed(2) + 'em'; });

  // Visual Assist controls — apply on any change
  const visualAssistControls = [
    'contrastMode', 'fontScale', 'lineHeight', 'letterSpacing',
    'dyslexiaFont', 'largeCursor', 'enhanceFocus', 'readingGuide'
  ];

  visualAssistControls.forEach(id => {
    const el = document.getElementById(id);
    const eventType = el?.type === 'range' ? 'input' : 'change';
    el?.addEventListener(eventType, async () => {
      const value = el.type === 'checkbox' ? el.checked
        : el.type === 'range' ? parseFloat(el.value) : el.value;
      await persistSetting(id, value);
      applyVisualAssist();
    });
  });

  document.getElementById('speechRate').addEventListener('change', async (e) => {
    await chrome.storage.sync.set({ speechRate: parseFloat(e.target.value) });
  });

  function applyVisualAssist() {
    const options = {
      contrastMode: document.getElementById('contrastMode').value,
      fontScale: parseFloat(fontScale.value) / 100,
      lineHeight: parseFloat(lineHeight.value),
      letterSpacing: parseFloat(letterSpacing.value),
      dyslexiaFont: document.getElementById('dyslexiaFont').checked,
      largeCursor: document.getElementById('largeCursor').checked,
      enhanceFocus: document.getElementById('enhanceFocus').checked,
      readingGuide: document.getElementById('readingGuide').checked
    };

    const hasChanges = options.contrastMode !== 'none' ||
      options.fontScale !== 1 || options.lineHeight !== 1.5 ||
      options.letterSpacing !== 0 || options.dyslexiaFont ||
      options.largeCursor || options.enhanceFocus || options.readingGuide;

    if (hasChanges) sendToContent({ type: 'enableTool', tool: 'VisualAssist', options });
    else sendToContent({ type: 'disableTool', tool: 'VisualAssist' });
  }

  // Read Aloud
  const readBtn = document.getElementById('readAloudBtn');
  let isReading = false;
  readBtn.addEventListener('click', () => {
    if (isReading) {
      sendToContent({ type: 'stopSpeech' });
      readBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size:14px">play_arrow</span> Read';
      isReading = false;
    } else {
      const rate = parseFloat(document.getElementById('speechRate').value);
      sendToContent({ type: 'speakPage', rate });
      readBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size:14px">stop</span> Stop';
      isReading = true;
    }
  });

  // --- Access Needs (functional, not diagnosis-based) ---
  const presets = {
    screenReader: {
      autoWcagFix: true, autoFixLabels: true, autoDescribe: true,
      autoVideoDescribe: true, keyboardNav: true
    },
    biggerText: {
      fontScale: 150, lineHeight: 2.0, letterSpacing: 0.12,
      largeCursor: true, enhanceFocus: true, autoWcagFix: true
    },
    colorAdjust: {
      autoDescribe: true, enhanceFocus: true
    },
    captions: {
      autoCaptions: true, enhanceFocus: true
    },
    altInput: {
      autoWcagFix: true, autoFixLabels: true, largeCursor: true,
      enhanceFocus: true, keyboardNav: true, voiceCommands: true
    },
    simplerContent: {
      autoSimplify: true, autoSummarize: true, fontScale: 120,
      lineHeight: 1.8, focusMode: true, hideDistractions: true, showProgress: true
    },
    fewerDistractions: {
      focusMode: true, hideDistractions: true, showProgress: true,
      motionReducer: true, autoSummarize: true
    },
    lessMotion: {
      motionReducer: true, focusMode: true, hideDistractions: true
    },
    dimmerDisplay: {
      darkMode: true, motionReducer: true
    },
    readingHelp: {
      fontScale: 115, lineHeight: 2.0, letterSpacing: 0.12, focusMode: true
    }
  };

  const profileCheckboxes = document.querySelectorAll('#profilesSection .profile-checkbox input');
  const profileCountEl = document.getElementById('profileCount');

  let savedProfiles = settings.selectedProfiles || [];
  profileCheckboxes.forEach(cb => { cb.checked = savedProfiles.includes(cb.value); });
  updateProfileCount(savedProfiles.length);
  if (savedProfiles.length > 0) applyPreset(mergePresets(savedProfiles));

  function updateProfileCount(count) {
    profileCountEl.textContent = count > 0 ? `${count} selected` : '';
  }

  function getSelectedProfiles() {
    return Array.from(profileCheckboxes).filter(cb => cb.checked).map(cb => cb.value);
  }

  function mergePresets(profileIds) {
    const numericKeys = ['fontScale', 'lineHeight', 'letterSpacing'];
    const merged = {};
    for (const id of profileIds) {
      const preset = presets[id];
      if (!preset) continue;
      for (const [key, value] of Object.entries(preset)) {
        if (numericKeys.includes(key) && typeof value === 'number') merged[key] = Math.max(merged[key] || 0, value);
        else if ((key === 'colorFilter' || key === 'colorBlindMode') && value !== 'none') merged[key] = value;
        else merged[key] = merged[key] || value;
      }
    }
    return merged;
  }

  profileCheckboxes.forEach(cb => {
    cb.addEventListener('change', async () => {
      const selectedProfiles = getSelectedProfiles();
      await chrome.storage.sync.set({ selectedProfiles });
      updateProfileCount(selectedProfiles.length);
      if (selectedProfiles.length === 0) await resetAll();
      else {
        await resetAllUI(true);
        sendToContent({ type: 'revertAll' });
        const merged = mergePresets(selectedProfiles);
        recordNonProfileSettings(merged);
        applyPreset(merged);
      }
    });
  });

  async function resetAllUI(preserveProfile = false) {
    // The user's Reset button (preserveProfile=false) means ALL settings off —
    // including the default-on AI tools. The internal pre-profile-apply reset
    // (preserveProfile=true) keeps those AI defaults on, so applying a saved
    // profile that doesn't mention them doesn't silently lose them.
    const defaultOnIds = ['showProgress', 'autoDescribe', 'autoWcagFix', 'autoFixLabels'];
    const togglesOff = ['darkMode', 'readerMode', 'focusMode', 'keyboardNav', 'voiceCommands', 'motionReducer',
      'dyslexiaFont', 'largeCursor', 'enhanceFocus', 'readingGuide', 'autoCaptions', 'autoVideoDescribe',
      'hideDistractions', 'autoSimplify', 'autoSummarize',
      ...(preserveProfile ? [] : defaultOnIds)];
    togglesOff.forEach(id => { const el = document.getElementById(id); if (el) el.checked = false; });

    const togglesOn = preserveProfile ? defaultOnIds : [];
    togglesOn.forEach(id => { const el = document.getElementById(id); if (el) el.checked = true; });

    setValue('contrastMode', 'none');
    setValue('colorBlindMode', 'none');
    setValue('speechRate', '1');

    if (fontScale) { fontScale.value = 100; fontScaleValue.textContent = '100%'; }
    if (lineHeight) { lineHeight.value = 1.5; lineHeightValue.textContent = '1.5'; }
    if (letterSpacing) { letterSpacing.value = 0; letterSpacingValue.textContent = '0.00em'; }

    if (focusOptions) focusOptions.classList.remove('show');

    if (!preserveProfile) {
      profileCheckboxes.forEach(cb => cb.checked = false);
      updateProfileCount(0);
    }

    const storageReset = {};
    togglesOff.forEach(id => storageReset[id] = false);
    togglesOn.forEach(id => storageReset[id] = true);
    storageReset.contrastMode = 'none';
    storageReset.colorBlindMode = 'none';
    storageReset.speechRate = 1;
    storageReset.fontScale = 100;
    storageReset.lineHeight = 1.5;
    storageReset.letterSpacing = 0;
    if (!preserveProfile) storageReset.selectedProfiles = [];
    // A full reset wipes the layered bookkeeping and base-profile selection
    // too — everything returns to a clean slate.
    if (!preserveProfile) {
      storageReset.baseProfiles = [];
      storageReset.nonProfileSettings = {};
      document.querySelectorAll('#baseProfileGrid input').forEach(cb => { cb.checked = false; });
    }
    await chrome.storage.sync.set(storageReset);

    // Full reset must also overwrite the Librarian's remembered explicit
    // settings (chrome.storage.local memory shards) — otherwise its
    // preference overlay re-applies the old values on the next page load and
    // the reset silently doesn't stick.
    if (!preserveProfile) {
      const { selectedProfiles, baseProfiles, nonProfileSettings, ...settingsOnly } = storageReset;
      sendMessageP({ type: 'librarianRecordScopedSettings', scope: 'general', settings: settingsOnly })
        .catch(() => {});
    }
  }

  function applyPreset(preset) {
    // Voice support means the harness Voice Assistant (side panel → local
    // extension-service), never the page-level speech-recognition adapter:
    // open the panel off this same click and force the page-STT flag off
    // (persisting false below also stops the on-page-load autostart).
    if (preset.voiceCommands) {
      openVoicePanel();
      preset = { ...preset, voiceCommands: false };
    }
    const has = (k) => preset[k] !== undefined;
    const num = (v) => typeof v === 'number' ? v : parseFloat(v) || 0;

    // Numeric display settings
    if (has('fontScale')) {
      const v = num(preset.fontScale);
      fontScale.value = v; fontScaleValue.textContent = v + '%';
    }
    if (has('lineHeight')) {
      const v = num(preset.lineHeight);
      lineHeight.value = v; lineHeightValue.textContent = v.toFixed(1);
    }
    if (has('letterSpacing')) {
      const v = num(preset.letterSpacing);
      letterSpacing.value = v; letterSpacingValue.textContent = v.toFixed(2) + 'em';
    }
    if (has('contrastMode')) setValue('contrastMode', preset.contrastMode);

    // Boolean display toggles
    ['dyslexiaFont', 'largeCursor', 'enhanceFocus', 'readingGuide'].forEach(key => {
      if (has(key)) setChecked(key, !!preset[key]);
    });

    // Tool toggles — enable OR disable based on the actual value
    const toolMap = {
      darkMode: 'DarkMode', motionReducer: 'MotionReducer', readerMode: 'ReaderMode',
      keyboardNav: 'KeyboardNavigator', voiceCommands: 'VoiceCommands'
    };
    for (const [key, toolName] of Object.entries(toolMap)) {
      if (!has(key)) continue;
      setChecked(key, !!preset[key]);
      if (preset[key]) sendToContent({ type: 'enableTool', tool: toolName });
      else sendToContent({ type: 'disableTool', tool: toolName });
    }

    // Focus Mode (with sub-options)
    if (has('focusMode')) {
      setChecked('focusMode', !!preset.focusMode);
      if (preset.focusMode) {
        focusOptions.classList.add('show');
        if (has('hideDistractions')) setChecked('hideDistractions', !!preset.hideDistractions);
        setChecked('showProgress', preset.showProgress !== false);
        sendFocusModeUpdate();
      } else {
        focusOptions.classList.remove('show');
        sendToContent({ type: 'disableTool', tool: 'FocusMode' });
      }
    }

    // Color blind mode
    if (has('colorBlindMode')) {
      setValue('colorBlindMode', preset.colorBlindMode);
      chrome.storage.sync.set({ colorBlindMode: preset.colorBlindMode });
      if (preset.colorBlindMode && preset.colorBlindMode !== 'none') {
        sendToContent({ type: 'enableTool', tool: 'ColorBlindMode', options: preset.colorBlindMode });
      } else {
        sendToContent({ type: 'disableTool', tool: 'ColorBlindMode' });
      }
    }

    // AI settings (including autoCaptions)
    ['autoWcagFix', 'autoFixLabels', 'autoDescribe', 'autoVideoDescribe', 'autoSimplify', 'autoSummarize', 'autoCaptions'].forEach(key => {
      if (has(key)) {
        setChecked(key, !!preset[key]);
        chrome.storage.sync.set({ [key]: !!preset[key] });
        sendToContent({ type: 'settingsChanged', settings: { [key]: !!preset[key] } });
      }
    });

    // Persist visual assist settings and apply
    const vaKeys = ['fontScale', 'lineHeight', 'letterSpacing', 'contrastMode',
      'dyslexiaFont', 'largeCursor', 'enhanceFocus', 'readingGuide'];
    if (vaKeys.some(k => has(k))) {
      const visualStorage = {
        fontScale: has('fontScale') ? num(preset.fontScale) : 100,
        lineHeight: has('lineHeight') ? num(preset.lineHeight) : 1.5,
        letterSpacing: has('letterSpacing') ? num(preset.letterSpacing) : 0,
        contrastMode: preset.contrastMode || 'none',
        dyslexiaFont: !!preset.dyslexiaFont, largeCursor: !!preset.largeCursor,
        enhanceFocus: !!preset.enhanceFocus, readingGuide: !!preset.readingGuide
      };
      chrome.storage.sync.set(visualStorage);
      applyVisualAssist();
    }

    // Persist tool toggles
    const toolStorage = {};
    ['darkMode', 'motionReducer', 'readerMode', 'keyboardNav', 'voiceCommands',
     'focusMode', 'hideDistractions', 'showProgress'].forEach(key => {
      if (has(key)) toolStorage[key] = !!preset[key];
    });
    if (Object.keys(toolStorage).length > 0) chrome.storage.sync.set(toolStorage);

    // Keep the Librarian's general-scope memory in step with what we just
    // persisted. At page load the Librarian's learned preferences overlay the
    // storage baseline (content.js init), so a stale "You set autoCaptions to
    // false" memory would silently override the profile the user just chose.
    const RECORDABLE = new Set(['darkMode', 'motionReducer', 'readerMode', 'keyboardNav',
      'voiceCommands', 'focusMode', 'hideDistractions', 'showProgress', 'colorBlindMode',
      'contrastMode', 'fontScale', 'lineHeight', 'letterSpacing', 'dyslexiaFont',
      'largeCursor', 'enhanceFocus', 'readingGuide', 'autoWcagFix', 'autoFixLabels',
      'autoDescribe', 'autoVideoDescribe', 'autoCaptions', 'autoSimplify', 'autoSummarize']);
    const librarianRecord = {};
    for (const [k, v] of Object.entries(preset)) {
      if (RECORDABLE.has(k)) librarianRecord[k] = v;
    }
    if (Object.keys(librarianRecord).length) {
      sendMessageP({ type: 'librarianRecordScopedSettings', scope: 'general', settings: librarianRecord })
        .catch(() => {});
    }
  }

  // Reset all
  document.getElementById('resetAll').addEventListener('click', resetAll);
  async function resetAll() {
    await resetAllUI();
    sendToContent({ type: 'revertAll' });
  }

  // API Keys
  document.getElementById('apiKeysSection').addEventListener('click', (e) => {
    if (e.target.closest('.collapsible-header'))
      document.getElementById('apiKeysSection').classList.toggle('open');
  });
  const storedGeminiKey = settings.geminiKey || '';
  document.getElementById('geminiKey').value = storedGeminiKey;
  // If key exists under old name but not new, migrate it
  if (storedGeminiKey && !settings.geminiApiKey) {
    chrome.runtime.sendMessage({ type: 'saveApiKey', apiKey: storedGeminiKey });
  }
  document.getElementById('saveKeys').addEventListener('click', async () => {
    const geminiKey = document.getElementById('geminiKey').value.trim();
    await chrome.storage.sync.set({ geminiKey });
    chrome.runtime.sendMessage({ type: 'saveApiKey', apiKey: geminiKey });
    const btn = document.getElementById('saveKeys');
    btn.textContent = 'Saved!';
    btn.classList.add('success');
    setTimeout(() => { btn.textContent = 'Save'; btn.classList.remove('success'); }, 1500);
  });

  // Scan page
  document.getElementById('scanPage').addEventListener('click', () => {
    sendToContent({ type: 'rescan' });
  });

  // Collapsible sections
  document.querySelectorAll('.section-header').forEach(header => {
    header.addEventListener('click', () => {
      const section = header.closest('.section');
      section.classList.toggle('collapsed');
      header.setAttribute('aria-expanded', !section.classList.contains('collapsed'));
    });
    header.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); header.click(); }
    });
  });

  // --- AI Support Input ---
  const aiInput = document.getElementById('aiSupportInput');
  const aiBtn = document.getElementById('aiSupportBtn');
  const aiSuggestion = document.getElementById('aiSuggestion');
  const aiLoading = document.getElementById('aiLoading');
  let pendingAISuggestion = null;

  function showAIError(msg) {
    document.getElementById('aiSuggestionSummary').textContent = msg;
    document.getElementById('aiSuggestionList').innerHTML = '';
    aiSuggestion.hidden = false;
    voiceQueryPending = false; // failed voice query must not chime on a later success
  }

  function finishAILoading() {
    aiLoading.hidden = true;
    aiBtn.disabled = false;
  }

  function sendMessageSafe(msg, timeoutMs = 60000) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Request timed out')), timeoutMs);
      chrome.runtime.sendMessage(msg, (resp) => {
        clearTimeout(timer);
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else {
          resolve(resp);
        }
      });
    });
  }

  async function submitSupportQuery(text) {
    if (!text.trim()) return;
    aiSuggestion.hidden = true;
    aiLoading.hidden = false;
    aiBtn.disabled = true;

    let resp;
    try {
      resp = await sendMessageSafe({ type: 'interpretNeeds', text: text.trim() });
    } catch (e) {
      finishAILoading();
      showAIError('Connection error: ' + (e.message || 'Try again.'));
      return;
    }

    finishAILoading();

    if (!resp || resp.error) {
      showAIError(resp?.error || 'Could not get suggestions. Check your API key.');
      return;
    }

    if (!resp.result) {
      showAIError('No response from AI. Check your API key and try again.');
      return;
    }

    try {
      const jsonMatch = resp.result.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        pendingAISuggestion = JSON.parse(jsonMatch[0]);
        showAISuggestions(pendingAISuggestion);
      } else {
        showAIError('AI returned an unexpected format. Try rephrasing.');
      }
    } catch (e) {
      showAIError('Could not parse AI response. Try rephrasing.');
    }
  }

  function scopeChipLabel(scope) {
    if (!scope || scope === 'general') return 'Everywhere';
    if (scope.startsWith('category:')) return 'On ' + scope.slice(9) + ' sites';
    if (scope.startsWith('origin:')) return 'On ' + scope.slice(7);
    if (scope.startsWith('context:')) return 'For ' + scope.slice(8) + ' content';
    return scope;
  }

  function showAISuggestions(result) {
    document.getElementById('aiSuggestionSummary').textContent = result.summary || 'Here are my suggestions:';
    const listEl = document.getElementById('aiSuggestionList');
    listEl.innerHTML = '';

    // Show where these will apply, so scoping is visible before the user acts.
    const scopeRow = document.createElement('div');
    scopeRow.className = 'ai-suggestion-scope';
    scopeRow.textContent = 'Applies: ' + scopeChipLabel(result.scope);
    listEl.appendChild(scopeRow);

    const settingLabels = {
      darkMode: 'Dark Mode', fontScale: 'Font Size', lineHeight: 'Line Height',
      letterSpacing: 'Letter Spacing', dyslexiaFont: 'Dyslexia Font', largeCursor: 'Large Cursor',
      enhanceFocus: 'Enhanced Focus', readingGuide: 'Reading Guide', focusMode: 'Focus Mode',
      hideDistractions: 'Dim Distractions', showProgress: 'Progress Bar', motionReducer: 'Reduce Motion',
      readerMode: 'Reader Mode', keyboardNav: 'Keyboard Nav', voiceCommands: 'Voice Commands',
      contrastMode: 'Contrast', colorBlindMode: 'Color Filter', autoWcagFix: 'WCAG Auto-Fix',
      autoDescribe: 'Image Alt Text', autoFixLabels: 'Generate Labels', autoCaptions: 'Captions',
      autoSimplify: 'Simplify Text', autoSummarize: 'Summarize Text', autoVideoDescribe: 'Video Descriptions'
    };

    if (result.settings) {
      for (const [key, value] of Object.entries(result.settings)) {
        const item = document.createElement('div');
        item.className = 'ai-suggestion-item';
        const label = settingLabels[key] || key;
        const displayVal = typeof value === 'boolean' ? (value ? 'ON' : 'OFF') : String(value);
        const reason = result.reasons?.[key] || '';
        item.innerHTML = `<span class="setting-name">${escapeHtml(label)}: ${displayVal}</span><span class="setting-reason">${escapeHtml(reason)}</span>`;
        listEl.appendChild(item);
      }
    }

    aiSuggestion.hidden = false;
    // Land focus on Apply so a plain Enter accepts the suggestions — matters
    // most for the voice flow (speak → suggestions → Enter), and the card's
    // aria-live region has already announced the content.
    document.getElementById('aiApplyBtn').focus();
    // Spoken query → audible "ready" cue as focus lands on Apply.
    if (voiceQueryPending) {
      voiceQueryPending = false;
      supportChime();
    }
  }

  // Apply the suggestion's built-in settings. A scoped request
  // ("...on news sites") is stored as a scoped Librarian preference (applies
  // only where it should) and pushed live to the current tab; an unscoped
  // request writes the global baseline as before.
  async function applySuggestionSettings(suggestion) {
    if (!suggestion?.settings || !Object.keys(suggestion.settings).length) return;
    // Voice → the harness Voice Assistant panel. Handled here (before any
    // await) so the user gesture from the Apply click still carries for the
    // scoped path; voiceCommands=false still flows through so any previously
    // stored page-STT flag gets turned off.
    if (suggestion.settings.voiceCommands) {
      openVoicePanel();
      suggestion.settings = { ...suggestion.settings, voiceCommands: false };
    }
    const scope = suggestion.scope;
    if (scope && scope !== 'general') {
      await sendMessageSafe({
        type: 'librarianRecordScopedSettings', scope, settings: suggestion.settings,
      });
      await sendToContent({ type: 'applyProfile', settings: suggestion.settings });
    } else {
      // Typed suggestions layer on top of the base-profile checkboxes:
      // record their contribution so a later profile-uncheck restores these
      // values instead of resetting them.
      recordNonProfileSettings(suggestion.settings);
      applyPreset(suggestion.settings);
    }
  }

  document.getElementById('aiApplyBtn').addEventListener('click', async () => {
    const sug = pendingAISuggestion;
    if (!sug) return;
    await applySuggestionSettings(sug);
    aiInput.value = '';
    aiSuggestion.hidden = true;
  });

  document.getElementById('aiSaveProfileBtn').addEventListener('click', async () => {
    if (pendingAISuggestion?.settings) {
      await applySuggestionSettings(pendingAISuggestion);
      aiSuggestion.hidden = true;
      aiInput.value = '';
      openSaveProfileModal();
    }
  });

  document.getElementById('aiDismissBtn').addEventListener('click', () => {
    aiSuggestion.hidden = true;
    pendingAISuggestion = null;
  });

  aiBtn.addEventListener('click', () => submitSupportQuery(aiInput.value));
  aiInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') submitSupportQuery(aiInput.value);
  });

  // --- Voice input for the support box (mic button, or Alt+S globally) ---
  // Local SpeechRecognition in the popup page: interim results land in the
  // textbox, the final transcript auto-submits the question. Mic permission
  // is per extension origin; if it's missing, the same grant-in-a-tab flow
  // the side panel uses (permission/mic.html) is opened.
  const aiMicBtn = document.getElementById('aiSupportMicBtn');
  let supportRec = null;
  let supportListening = false;
  // Set when a spoken query is submitted; the suggestion render plays a
  // completion chime (and clears it) so the user knows Enter will now apply.
  let voiceQueryPending = false;

  // Earcon: same two-note "done" chime as the voice panel (Web Audio, no
  // asset). Popup audio is allowed here because the flow started from a user
  // gesture (mic button / Alt+S command).
  let supportAC = null;
  function supportChime() {
    try {
      if (!supportAC) supportAC = new (window.AudioContext || window.webkitAudioContext)();
      if (supportAC.state === 'suspended') supportAC.resume();
      const blip = (freq, dur, when, vol) => {
        const t = supportAC.currentTime + when;
        const o = supportAC.createOscillator();
        const g = supportAC.createGain();
        o.type = 'sine';
        o.frequency.value = freq;
        g.gain.setValueAtTime(0.0001, t);
        g.gain.linearRampToValueAtTime(vol, t + 0.015);
        g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
        o.connect(g).connect(supportAC.destination);
        o.start(t);
        o.stop(t + dur + 0.02);
      };
      blip(660, 0.12, 0, 0.06);
      blip(880, 0.18, 0.11, 0.06);
    } catch (_) {}
  }

  async function ensureSupportMic() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((t) => t.stop());
      return true;
    } catch (_) {
      try { chrome.tabs.create({ url: chrome.runtime.getURL('permission/mic.html') }); } catch (_) {}
      showAIError('Opened a tab to enable the microphone — click Allow there, then try the mic again.');
      return false;
    }
  }

  async function startSupportVoiceInput() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { showAIError('No speech recognition in this browser.'); return; }
    if (supportListening) { try { supportRec && supportRec.stop(); } catch (_) {} return; }
    if (!(await ensureSupportMic())) return;
    supportListening = true;
    aiMicBtn.classList.add('listening');
    aiInput.value = '';
    aiInput.placeholder = 'Listening… speak, then pause to submit';
    const rec = new SR();
    supportRec = rec;
    rec.lang = 'en-US';
    rec.interimResults = true;
    rec.continuous = false; // one utterance → submit
    let finalText = '';
    rec.onresult = (e) => {
      let interim = '';
      for (const res of e.results) {
        if (res.isFinal) finalText += res[0].transcript;
        else interim += res[0].transcript;
      }
      aiInput.value = (finalText + interim).trim();
    };
    rec.onerror = (e) => {
      supportListening = false;
      aiMicBtn.classList.remove('listening');
      aiInput.placeholder = 'What support do you need? e.g. "I\'m tired and can\'t see well"';
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') ensureSupportMic();
    };
    rec.onend = () => {
      supportListening = false;
      aiMicBtn.classList.remove('listening');
      aiInput.placeholder = 'What support do you need? e.g. "I\'m tired and can\'t see well"';
      const text = (finalText || aiInput.value).trim();
      if (text) {
        voiceQueryPending = true;
        submitSupportQuery(text);
      }
    };
    try { rec.start(); } catch (_) { supportListening = false; aiMicBtn.classList.remove('listening'); }
  }

  if (aiMicBtn) aiMicBtn.addEventListener('click', startSupportVoiceInput);

  // Armed by the Alt+S command (background sets pendingVoiceSupport and opens
  // the popup; the tab fallback passes ?voice=1 instead).
  (async () => {
    try {
      if (new URLSearchParams(location.search).get('voice') === '1') { startSupportVoiceInput(); return; }
      const { pendingVoiceSupport } = await chrome.storage.session.get(['pendingVoiceSupport']);
      if (pendingVoiceSupport && Date.now() - pendingVoiceSupport < 10000) {
        await chrome.storage.session.remove('pendingVoiceSupport');
        startSupportVoiceInput();
      }
    } catch (_) {}
  })();

  // --- Custom Profiles ---
  const ALL_SETTING_KEYS = [
    'darkMode', 'readerMode', 'focusMode', 'keyboardNav', 'voiceCommands', 'motionReducer',
    'dyslexiaFont', 'largeCursor', 'enhanceFocus', 'readingGuide',
    'hideDistractions', 'showProgress',
    'fontScale', 'lineHeight', 'letterSpacing', 'contrastMode', 'colorBlindMode', 'speechRate',
    'autoWcagFix', 'autoDescribe', 'autoFixLabels', 'autoCaptions',
    'autoVideoDescribe', 'autoSimplify', 'autoSummarize'
  ];

  function captureCurrentSettings() {
    const s = {};
    for (const id of ALL_SETTING_KEYS) {
      const el = document.getElementById(id);
      if (!el) continue;
      if (el.type === 'checkbox') s[id] = el.checked;
      else if (el.type === 'range') s[id] = parseFloat(el.value);
      else s[id] = el.value;
    }
    return s;
  }

  let modalReturnFocus = null;
  const modalOverlay = document.getElementById('saveProfileModal');
  const modalDialog = modalOverlay.querySelector('.modal');

  function openSaveProfileModal() {
    modalReturnFocus = document.activeElement;
    document.getElementById('profileNameInput').value = '';
    document.querySelectorAll('.site-type-grid input').forEach(cb => cb.checked = false);
    modalOverlay.classList.add('open');
    modalOverlay.setAttribute('aria-hidden', 'false');
    document.getElementById('profileNameInput').focus();
  }

  function closeSaveProfileModal() {
    modalOverlay.classList.remove('open');
    modalOverlay.setAttribute('aria-hidden', 'true');
    if (modalReturnFocus && typeof modalReturnFocus.focus === 'function') {
      modalReturnFocus.focus();
    }
    modalReturnFocus = null;
  }

  function trapFocusInModal(e) {
    if (e.key !== 'Tab') return;
    const focusable = modalDialog.querySelectorAll(
      'input, button, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }

  document.getElementById('saveProfileBtn').addEventListener('click', openSaveProfileModal);
  document.getElementById('modalCancelBtn').addEventListener('click', closeSaveProfileModal);

  document.getElementById('modalSaveBtn').addEventListener('click', async () => {
    const name = document.getElementById('profileNameInput').value.trim();
    if (!name) { document.getElementById('profileNameInput').focus(); return; }

    const saveBtn = document.getElementById('modalSaveBtn');
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';

    const siteTypes = Array.from(document.querySelectorAll('.site-type-grid input:checked')).map(cb => cb.value);
    const profile = {
      id: 'profile-' + Date.now(),
      name,
      siteTypes,
      autoApply: siteTypes.length > 0,
      settings: captureCurrentSettings()
    };

    try {
      await chrome.runtime.sendMessage({ type: 'saveCustomProfile', profile });
      closeSaveProfileModal();
      await loadAndRenderProfiles();
    } catch (e) {
      console.warn('Save profile error:', e);
      const nameInput = document.getElementById('profileNameInput');
      nameInput.setCustomValidity('Save failed. Try again.');
      nameInput.reportValidity();
    } finally {
      saveBtn.disabled = false;
      saveBtn.textContent = 'Save';
    }
  });

  modalOverlay.addEventListener('click', (e) => {
    if (e.target === modalOverlay) closeSaveProfileModal();
  });

  modalOverlay.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeSaveProfileModal();
    trapFocusInModal(e);
  });

  async function loadAndRenderProfiles() {
    let profiles = [];
    try {
      const resp = await chrome.runtime.sendMessage({ type: 'getCustomProfiles' });
      profiles = resp?.profiles || [];
    } catch (e) {
      console.warn('Failed to load profiles:', e);
    }

    const section = document.getElementById('myProfilesSection');
    const list = document.getElementById('myProfilesList');
    const countEl = document.getElementById('myProfilesCount');

    if (profiles.length === 0) { section.hidden = true; return; }

    section.hidden = false;
    countEl.textContent = String(profiles.length);
    list.innerHTML = '';

    for (const p of profiles) {
      const row = document.createElement('div');
      row.className = 'profile-item';

      const nameSpan = document.createElement('div');
      nameSpan.style.cssText = 'flex:1;min-width:0';
      nameSpan.innerHTML = `<div class="profile-item-name">${escapeHtml(p.name)}</div>` +
        (p.siteTypes?.length ? `<div class="profile-item-sites">${p.siteTypes.join(', ')}</div>` : '');

      const applyBtn = document.createElement('button');
      applyBtn.className = 'profile-item-btn apply';
      applyBtn.textContent = 'Apply';
      applyBtn.addEventListener('click', async () => {
        await resetAllUI(true);
        sendToContent({ type: 'revertAll' });
        recordNonProfileSettings(p.settings);
        applyPreset(p.settings);
      });

      const delBtn = document.createElement('button');
      delBtn.className = 'profile-item-btn delete';
      delBtn.textContent = 'Delete';
      delBtn.addEventListener('click', async () => {
        try {
          await chrome.runtime.sendMessage({ type: 'deleteCustomProfile', id: p.id });
        } catch (e) {
          console.warn('Delete profile error:', e);
        }
        await loadAndRenderProfiles();
      });

      row.appendChild(nameSpan);
      row.appendChild(applyBtn);
      row.appendChild(delBtn);
      list.appendChild(row);
    }
  }

  loadAndRenderProfiles();

  // Query states from content
  queryToolStates();
  queryStats();

  // What the Librarian knows: pending proposals (consent gate), learned
  // memories grouped by where they apply, and standing "don't suggest"
  // instructions. All plain language; raw records stay in storage.
  setupMemoryPanel();

  // Assistant: agentic browser task sent to the local extension-service.
  setupAssistantPanel();

  // Skills: the Engineer (adaptive skill builder) + the Skills db.
  setupSkillsPanel();

  // Base ability profiles (blind, deaf, dyslexia, …) — rendered from the
  // toolkit's profile catalog (AA_PROFILES, bundled from the package).
  // Checking profiles applies the union of their settings through the same
  // applyPreset path the AI suggestions use (so voiceCommands → the harness
  // Voice Assistant panel, unknown settings ignored); unchecking turns the
  // dropped settings back off. Selection persists and seeds the Librarian's
  // supportAreas, which is what skill retrieval matches on.
  async function setupBaseProfiles() {
    const grid = document.getElementById('baseProfileGrid');
    if (!grid || !globalThis.AA_PROFILES) return;
    const { baseProfiles } = await chrome.storage.sync.get(['baseProfiles']);
    const selected = new Set(baseProfiles || []);

    for (const [id, p] of Object.entries(AA_PROFILES.profiles)) {
      const label = document.createElement('label');
      label.className = 'profile-checkbox';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.value = id;
      cb.checked = selected.has(id);
      label.append(cb, ` ${p.name}`);
      if (p.description) label.title = p.description;
      grid.append(label);
    }

    // Profile id → Librarian supportAreas (the registry's vocabulary).
    const SUPPORT_AREAS = {
      blind: ['vision'], lowVision: ['vision'], colorBlind: ['vision'],
      deaf: ['hearing'], motor: ['motor'], dyslexia: ['reading'],
      adhd: ['cognitive'], cognitive: ['cognitive'],
      olderAdult: ['vision', 'cognitive'], anxiety: ['sensory'],
      sensory: ['sensory'], photosensitive: ['vision', 'sensory'],
    };
    const NUMERIC_DEFAULTS = { fontScale: 100, lineHeight: 1.5, letterSpacing: 0 };

    // Local additions on top of the toolkit's profile data: our voice mode is
    // a spoken assistant (talks back), so unlike the toolkit's page-level
    // voice commands it belongs in the vision profiles too. voiceCommands
    // routes to the harness Voice Assistant panel via applyPreset.
    const PROFILE_TOOL_EXTRAS = {
      blind: { voiceCommands: true },
      lowVision: { voiceCommands: true },
    };
    const mergeTools = (ids) => {
      const union = AA_PROFILES.mergeProfileTools(ids);
      for (const id of ids) Object.assign(union, PROFILE_TOOL_EXTRAS[id] || {});
      return union;
    };

    let prevUnion = mergeTools([...selected]);
    grid.addEventListener('change', async (e) => {
      const cb = e.target;
      if (cb.type !== 'checkbox') return;
      // Re-derive from the DOM (a full Reset unchecks boxes externally).
      selected.clear();
      grid.querySelectorAll('input:checked').forEach((b) => selected.add(b.value));
      const union = mergeTools([...selected]);
      // Voice panel must open synchronously off the click gesture — before
      // the storage await below consumes it (applyPreset's own intercept
      // would run too late here).
      if (union.voiceCommands && !prevUnion.voiceCommands) openVoicePanel();
      // Settings no longer backed by any checked profile: restore the value a
      // non-checkbox source (typed suggestion, manual toggle, custom profile)
      // last set, else reset to the default.
      const { nonProfileSettings } = await chrome.storage.sync.get(['nonProfileSettings']);
      const layered = nonProfileSettings || {};
      const preset = {};
      for (const k of Object.keys(prevUnion)) {
        if (k in union) continue;
        if (k in layered) preset[k] = layered[k];
        else preset[k] = (k in NUMERIC_DEFAULTS) ? NUMERIC_DEFAULTS[k] : false;
      }
      Object.assign(preset, union);
      prevUnion = union;
      applyPreset(preset);
      chrome.storage.sync.set({ baseProfiles: [...selected] });
      const areas = [...new Set([...selected].flatMap((id) => SUPPORT_AREAS[id] || []))];
      sendMessageP({ type: 'librarianSetProfileField', path: 'supportAreas', value: areas })
        .catch(() => {});
    });
  }
  setupBaseProfiles();
});

// Skills panel — the toolkit's adaptive-agent loop, popup edition:
// find-or-build (reuse offered before the Engineer is asked) → try on the
// live page (nothing persisted) → feedback → revise → save only on an
// explicit click. All skill logic runs in the background service worker
// (librarian* messages); this panel is UI + the try-on-page bridge.
function setupSkillsPanel() {
  const needInput = document.getElementById('skillNeedInput');
  const findBtn = document.getElementById('skillFindBtn');
  const statusEl = document.getElementById('skillStatus');
  if (!needInput || !findBtn) return;

  const matchCard = document.getElementById('skillMatchCard');
  const builtCard = document.getElementById('skillBuiltCard');

  // View SKILL.md collapsible (same pattern as Service settings).
  const mdSection = document.getElementById('skillMdSection');
  mdSection.addEventListener('click', (e) => {
    if (e.target.closest('.collapsible-header')) mdSection.classList.toggle('open');
  });
  mdSection.querySelector('.collapsible-header').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); mdSection.classList.toggle('open'); }
  });
  let matchSkill = null;   // the offered existing skill
  let builtSkill = null;   // the Engineer's current (unsaved) skill
  let lastNeed = '';

  const setStatus = (text) => { statusEl.textContent = text || ''; };

  // Apply a skill's resolved plan to the active tab, without persisting:
  // adapter settings go through the content script's settingsChanged path;
  // action steps are handed to the Assistant (extension-service).
  async function applyPlan(skill, label) {
    const resp = await sendMessageP({ type: 'librarianResolveSkill', skill });
    const plan = resp && resp.plan;
    if (!plan) { setStatus('Could not resolve that skill.'); return; }
    if (Object.keys(plan.settings).length) {
      // applyProfile is the content script's full visual-settings path
      // (darkMode, motionReducer, focusMode, fontScale, …); settingsChanged
      // only covers the AI auto-* keys.
      await sendToContent({ type: 'applyProfile', settings: plan.settings });
    }
    let note = `Applied "${label}": ${Object.entries(plan.settings).map(([k, v]) => `${k}=${v}`).join(', ') || 'no visual settings'}.`;
    if (plan.actions.length) {
      // Action steps are agent tasks — run the first through the Assistant.
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      chrome.runtime.sendMessage(
        { type: 'assistantRun', prompt: plan.actions[0].prompt, activeUrl: tab && tab.url },
        () => { void chrome.runtime.lastError; });
      note += ` Running action via the Assistant: "${plan.actions[0].name}" (see Assistant panel).`;
    }
    setStatus(note);
  }

  // find-or-build: offer an existing skill before asking the Engineer.
  findBtn.addEventListener('click', async () => {
    lastNeed = needInput.value.trim();
    if (!lastNeed) return;
    matchCard.hidden = true;
    builtCard.hidden = true;
    setStatus('Checking your skills…');
    const resp = await sendMessageP({ type: 'librarianFindSkillForNeed', need: lastNeed });
    matchSkill = resp && resp.skill;
    if (matchSkill) {
      document.getElementById('skillMatchDesc').textContent =
        `"${matchSkill.name}" (${matchSkill.source}) — ${matchSkill.description}`;
      matchCard.hidden = false;
      setStatus('');
    } else {
      await buildNow();
    }
  });

  document.getElementById('skillUseMatchBtn').addEventListener('click', async () => {
    if (!matchSkill) return;
    matchCard.hidden = true;
    await applyPlan(matchSkill, matchSkill.name);
  });

  document.getElementById('skillBuildAnywayBtn').addEventListener('click', async () => {
    matchCard.hidden = true;
    await buildNow();
  });

  async function buildNow(previous = null, feedback = '') {
    setStatus(feedback ? 'The Engineer is revising the skill…' : 'The Engineer is building a skill…');
    findBtn.disabled = true;
    const result = await sendMessageP({
      type: 'librarianBuildSkill', need: lastNeed, opts: { previous, feedback },
    }).catch((e) => ({ skill: null, valid: false, errors: [e.message] }));
    findBtn.disabled = false;
    builtSkill = result && result.skill;
    if (!builtSkill) {
      setStatus(`The Engineer couldn't build that: ${(result && result.errors || ['no response']).join('; ')}`);
      return;
    }
    setStatus('');
    document.getElementById('skillBuiltTitle').textContent =
      feedback ? 'The Engineer revised the skill' : 'The Engineer built a skill';
    document.getElementById('skillBuiltDesc').textContent =
      `"${builtSkill.name}" — ${builtSkill.description}`;
    document.getElementById('skillBuiltValidity').textContent = result.valid
      ? 'Validated against the adapter registry. Nothing is saved until you say so.'
      : `Validation problems: ${result.errors.join('; ')}`;
    document.getElementById('skillBuiltMd').textContent =
      (globalThis.AA_SKILL_CORE && AA_SKILL_CORE.serializeSkill(builtSkill)) || JSON.stringify(builtSkill, null, 2);
    document.getElementById('skillSaveBtn').disabled = !result.valid;
    builtCard.hidden = false;
  }

  document.getElementById('skillTryBtn').addEventListener('click', async () => {
    if (builtSkill) await applyPlan(builtSkill, `${builtSkill.name} (unsaved — try before you save)`);
  });

  document.getElementById('skillReviseBtn').addEventListener('click', async () => {
    const feedback = document.getElementById('skillFeedbackInput').value.trim();
    if (!builtSkill || !feedback) return;
    await buildNow(builtSkill, feedback);
  });

  document.getElementById('skillDiscardBtn').addEventListener('click', () => {
    builtCard.hidden = true;
    builtSkill = null;
    setStatus('Discarded — nothing was saved.');
  });

  document.getElementById('skillSaveBtn').addEventListener('click', async () => {
    if (!builtSkill) return;
    const resp = await sendMessageP({ type: 'librarianSaveSkill', skill: builtSkill });
    if (resp && resp.saved) {
      setStatus(`Saved "${builtSkill.name}" to your skills.`);
      builtCard.hidden = true;
      builtSkill = null;
      renderSkillList();
    } else {
      setStatus(`Save refused: ${(resp && resp.errors || ['no response']).join('; ')}`);
    }
  });

  // The Skills db: built-in + the user's own, with Apply (and Delete for mine).
  async function renderSkillList() {
    const list = document.getElementById('skillList');
    const resp = await sendMessageP({ type: 'librarianListSkills' }).catch(() => null);
    const skills = (resp && resp.skills) || [];
    document.getElementById('skillCount').textContent = String(skills.length);
    list.textContent = '';
    for (const s of skills) {
      const row = document.createElement('div');
      row.className = 'skill-item';
      const badge = document.createElement('span');
      badge.className = `skill-badge skill-badge-${s.source}`;
      badge.textContent = s.source === 'mine' ? 'yours' : 'built-in';
      const name = document.createElement('span');
      name.className = 'skill-item-name';
      name.textContent = s.name;
      name.title = s.description || '';
      const applyBtn = document.createElement('button');
      applyBtn.className = 'btn btn-secondary btn-sm';
      applyBtn.textContent = 'Apply';
      applyBtn.addEventListener('click', () => applyPlan(s, s.name));
      row.append(badge, name, applyBtn);
      if (s.source === 'mine') {
        const delBtn = document.createElement('button');
        delBtn.className = 'btn btn-secondary btn-sm';
        delBtn.textContent = 'Delete';
        delBtn.addEventListener('click', async () => {
          await sendMessageP({ type: 'librarianDeleteSkill', name: s.name });
          renderSkillList();
        });
        row.append(delBtn);
      }
      list.append(row);
    }
  }

  renderSkillList();
}

// Assistant panel: service settings (serviceUrl/serviceToken in storage.sync)
// plus a task box that hands off to background's assistantRun. The live view is
// driven off chrome.storage.local.assistant so a result that arrives after the
// popup was closed still shows when it reopens.
function setupAssistantPanel() {
  const DEFAULT_URL = 'http://127.0.0.1:8787';
  const urlInput = document.getElementById('serviceUrlInput');
  const tokenInput = document.getElementById('serviceTokenInput');
  const saveBtn = document.getElementById('saveServiceBtn');
  const promptEl = document.getElementById('assistantPrompt');
  const runBtn = document.getElementById('assistantRunBtn');
  const clearBtn = document.getElementById('assistantClearBtn');
  const statusEl = document.getElementById('assistantStatus');
  if (!promptEl || !runBtn || !statusEl) return;

  // Open the hands-free voice Assistant in a side panel. The click itself is the
  // user gesture chrome.sidePanel.open() requires; it must run synchronously off
  // the click before any awaited work, so we resolve the window id first.
  const voicePanelBtn = document.getElementById('openVoicePanelBtn');
  if (voicePanelBtn) {
    if (!chrome.sidePanel?.open) {
      voicePanelBtn.hidden = true;
    } else {
      voicePanelBtn.addEventListener('click', () => { openVoicePanel(); });
    }
  }

  // Service settings collapsible toggle (mirrors the API Keys section).
  const svcSection = document.getElementById('serviceSettingsSection');
  svcSection?.addEventListener('click', (e) => {
    if (e.target.closest('.collapsible-header')) svcSection.classList.toggle('open');
  });
  svcSection?.querySelector('.collapsible-header')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); svcSection.classList.toggle('open'); }
  });

  // Load + persist service settings (storage.sync).
  chrome.storage.sync.get(['serviceUrl', 'serviceToken'], (data) => {
    if (urlInput) urlInput.value = data.serviceUrl || DEFAULT_URL;
    if (tokenInput) tokenInput.value = data.serviceToken || '';
  });
  function saveService() {
    const serviceUrl = (urlInput?.value || '').trim() || DEFAULT_URL;
    const serviceToken = (tokenInput?.value || '').trim();
    chrome.storage.sync.set({ serviceUrl, serviceToken });
  }
  urlInput?.addEventListener('change', saveService);
  tokenInput?.addEventListener('change', saveService);
  saveBtn?.addEventListener('click', () => {
    saveService();
    saveBtn.textContent = 'Saved!';
    saveBtn.classList.add('success');
    setTimeout(() => { saveBtn.textContent = 'Save'; saveBtn.classList.remove('success'); }, 1500);
  });

  // icon is a Material Symbols name from the restricted set loaded in
  // popup.html (icon_names=...); pass null to skip the glyph.
  function mkHead(cls, icon, label, spin) {
    const head = document.createElement('div');
    head.className = 'assistant-state-head ' + cls;
    if (icon) {
      const ic = document.createElement('span');
      ic.className = 'material-symbols-outlined' + (spin ? ' spin' : '');
      ic.setAttribute('aria-hidden', 'true');
      ic.textContent = icon;
      head.appendChild(ic);
    }
    head.appendChild(document.createTextNode(label));
    return head;
  }

  // One row of the live execution log (agent narration + harness commands).
  function renderLog(log) {
    const wrap = document.createElement('div');
    wrap.className = 'assistant-log';
    // Newest entry first so the latest harness step is always at the top.
    for (let i = log.length - 1; i >= 0; i--) {
      const e = log[i];
      const row = document.createElement('div');
      if (e.kind === 'assistant') {
        const text = (e.text || '').trim();
        if (!text) continue;
        row.className = 'log-assistant';
        row.textContent = text;
      } else if (e.kind === 'tool') {
        row.className = 'log-tool';
        const label = document.createElement('span');
        label.className = 'log-tool-label';
        label.textContent = '▶ ' + (e.name === 'run_shell_command' ? 'harness' : (e.name || 'tool'));
        const cmd = document.createElement('pre');
        cmd.className = 'log-tool-cmd';
        cmd.textContent = e.command || '';
        row.appendChild(label);
        row.appendChild(cmd);
      } else if (e.kind === 'tool_result') {
        row.className = 'log-tool-result ' + (e.status === 'success' ? 'ok' : 'warn');
        row.textContent = (e.status === 'success' ? '✓ ' : '• ') + (e.status || '');
      } else {
        continue;
      }
      wrap.appendChild(row);
    }
    return wrap;
  }

  function render(state) {
    statusEl.textContent = '';
    const status = state?.status;
    runBtn.disabled = status === 'running';
    if (clearBtn) clearBtn.hidden = !state;
    if (!state) return;

    if (status === 'running') {
      statusEl.className = 'assistant-status-region assistant-state-running';
      statusEl.appendChild(mkHead('', 'sync', ' Working on it…', true));
    } else if (status === 'done') {
      statusEl.className = 'assistant-status-region assistant-state-done';
      statusEl.appendChild(mkHead('', 'check_circle', ' Done'));
    } else if (status === 'error') {
      statusEl.className = 'assistant-status-region assistant-state-error';
      statusEl.appendChild(mkHead('', null, 'Something went wrong'));
      const body = document.createElement('div');
      body.className = 'assistant-error';
      body.textContent = state.error || 'Unknown error.';
      statusEl.appendChild(body);
      if (/service/i.test(state.error || '')) {
        const hint = document.createElement('div');
        hint.className = 'assistant-hint';
        hint.innerHTML = 'Start the service with <code>extension-service/run.sh</code>, then try again.';
        statusEl.appendChild(hint);
      }
    } else {
      statusEl.className = 'assistant-status-region';
    }

    // Live execution trace (what the browser-harness agent is doing).
    if (Array.isArray(state.log) && state.log.length) {
      statusEl.appendChild(renderLog(state.log));
    }

    // Final answer.
    if (status === 'done') {
      const body = document.createElement('div');
      body.className = 'assistant-result';
      body.textContent = state.result || 'The Assistant finished.';
      statusEl.appendChild(body);
    }

    if (state.task && status !== 'error') {
      const t = document.createElement('div');
      t.className = 'assistant-task';
      t.textContent = state.task;
      statusEl.appendChild(t);
    }
    // Newest log entry is at the top — keep it in view.
    statusEl.scrollTop = 0;
  }

  // Initial state + live updates (result may arrive after the popup reopens).
  chrome.storage.local.get('assistant', (d) => render(d.assistant || null));
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'local' && changes.assistant) render(changes.assistant.newValue || null);
  });

  runBtn.addEventListener('click', async () => {
    const prompt = promptEl.value.trim();
    if (!prompt) { promptEl.focus(); return; }
    runBtn.disabled = true;
    render({ status: 'running', task: prompt });
    let activeUrl = '';
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      activeUrl = tab?.url || '';
    } catch (e) {}
    chrome.runtime.sendMessage({ type: 'assistantRun', prompt, activeUrl }, () => {
      void chrome.runtime.lastError; // storage.local.assistant is the source of truth
    });
  });

  clearBtn?.addEventListener('click', () => {
    chrome.runtime.sendMessage({ type: 'assistantClear' }, () => { void chrome.runtime.lastError; });
    render(null);
  });
}

// Open the hands-free Voice Assistant side panel — the extension's voice
// mode (STT/TTS + Assistant → local extension-service). Call synchronously
// off a user gesture; chrome.sidePanel.open() needs the gesture to survive.
async function openVoicePanel() {
  if (!chrome.sidePanel?.open) return false;
  try {
    const w = await chrome.windows.getCurrent();
    await chrome.sidePanel.open({ windowId: w.id });
    return true;
  } catch (e) { return false; }
}

function sendMessageP(msg) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(msg, (resp) => {
      if (chrome.runtime.lastError) { resolve(null); return; }
      resolve(resp);
    });
  });
}

function setupMemoryPanel() {
  const proposalList = document.getElementById('proposalList');
  const proposalBanner = document.getElementById('proposalBanner');
  const memoryList = document.getElementById('memoryList');
  const badge = document.getElementById('memoryBadge');
  const pauseToggle = document.getElementById('memoryPauseToggle');
  if (!proposalList || !memoryList) return;

  pauseToggle?.addEventListener('change', async () => {
    await sendMessageP({ type: 'librarianSetPause', paused: !pauseToggle.checked });
  });

  function scopeLabel(scope) {
    if (scope === 'general') return 'Everywhere';
    if (scope.startsWith('category:')) return 'On ' + scope.slice(9) + ' sites';
    if (scope.startsWith('origin:')) return 'On ' + scope.slice(7);
    if (scope.startsWith('context:')) return 'For ' + scope.slice(8) + ' content';
    if (scope.startsWith('tool:')) return 'Tool: ' + scope.slice(5);
    return scope;
  }

  async function respond(id, response) {
    await sendMessageP({ type: 'librarianRespondToProposal', id, response });
    await render();
  }

  async function render() {
    const [profResp, propResp, memResp] = await Promise.all([
      sendMessageP({ type: 'librarianGetProfile' }),
      sendMessageP({ type: 'librarianListProposals' }),
      sendMessageP({ type: 'librarianListMemories', filter: { status: 'active' } }),
    ]);

    if (pauseToggle && profResp?.profile) {
      pauseToggle.checked = !profResp.profile.memoryPaused;
    }

    // --- Proposals: the consent gate ---
    // Rendered into the top-of-popup banner so a pending suggestion ("Is
    // this a reusable task?") is the first thing seen, not buried under the
    // settings sections. The in-section list is kept empty to avoid dupes.
    const proposals = propResp?.proposals || [];
    if (badge) {
      badge.hidden = proposals.length === 0;
      badge.textContent = String(proposals.length);
    }
    const buildCard = (p) => {
      const card = document.createElement('div');
      card.className = 'proposal-card';
      const title = document.createElement('div');
      title.className = 'proposal-title';
      title.textContent = 'Suggestion: ' + (p.aspectLabel || p.aspect);
      const why = document.createElement('div');
      why.className = 'proposal-rationale';
      why.textContent = p.rationale || '';
      const actions = document.createElement('div');
      actions.className = 'proposal-actions';
      const mk = (label, response, cls) => {
        const b = document.createElement('button');
        b.className = 'btn btn-sm ' + cls;
        b.textContent = label;
        b.addEventListener('click', () => respond(p.id, response));
        return b;
      };
      actions.appendChild(mk('Yes, apply', 'accept', 'btn-primary'));
      actions.appendChild(mk('Not now', 'declineOnce', 'btn-secondary'));
      actions.appendChild(mk("Don't suggest this", 'suppress', 'btn-secondary'));
      card.appendChild(title);
      card.appendChild(why);
      card.appendChild(actions);
      return card;
    };
    const target = proposalBanner || proposalList;
    target.textContent = '';
    if (proposalList !== target) proposalList.textContent = '';
    for (const p of proposals) target.appendChild(buildCard(p));
    if (proposalBanner) proposalBanner.hidden = proposals.length === 0;

    // --- Memories grouped by scope, plus suppressions ---
    memoryList.textContent = '';
    const memories = memResp?.memories || [];
    const suppressions = (memResp?.suppressions || []).filter(s => s.mode === 'permanent');
    if (!memories.length && !suppressions.length && !proposals.length) {
      const empty = document.createElement('div');
      empty.className = 'memory-empty';
      empty.textContent = 'Nothing learned yet. As you browse and use the agent, useful preferences will appear here for your review.';
      memoryList.appendChild(empty);
      return;
    }
    const groups = new Map();
    for (const m of memories) {
      const label = scopeLabel(m.scope);
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label).push(m);
    }
    const renderItem = (parent, id, text, title) => {
      const row = document.createElement('div');
      row.className = 'memory-item';
      const span = document.createElement('span');
      span.className = 'memory-text';
      span.textContent = text;
      if (title) span.title = title;
      const del = document.createElement('button');
      del.className = 'memory-delete';
      del.textContent = '✕';
      del.title = 'Forget this';
      del.setAttribute('aria-label', 'Forget: ' + text);
      del.addEventListener('click', async () => {
        await sendMessageP({ type: 'librarianDeleteMemory', id });
        await render();
      });
      row.appendChild(span);
      row.appendChild(del);
      parent.appendChild(row);
    };
    for (const [label, items] of groups) {
      const h = document.createElement('div');
      h.className = 'memory-group-title';
      h.textContent = label;
      memoryList.appendChild(h);
      for (const m of items) {
        renderItem(memoryList, m.id, m.text,
          `Learned ${new Date(m.firstSeenAt).toLocaleDateString()} · seen ${m.occurrenceCount}×`);
      }
    }
    if (suppressions.length) {
      const h = document.createElement('div');
      h.className = 'memory-group-title';
      h.textContent = "Things you've told me not to suggest";
      memoryList.appendChild(h);
      for (const s of suppressions) {
        renderItem(memoryList, s.id,
          s.text || s.aspect,
          'Since ' + new Date(s.createdAt).toLocaleDateString() + ' — delete to allow suggestions again');
      }
    }
  }

  render();
}

async function sendToContent(message) {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id && !tab.url?.startsWith('chrome://') && !tab.url?.startsWith('chrome-extension://'))
      chrome.tabs.sendMessage(tab.id, message).catch(() => {});
  } catch (e) {}
}

async function queryToolStates() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id && !tab.url?.startsWith('chrome://')) {
      const response = await chrome.tabs.sendMessage(tab.id, { type: 'getToolStates' }).catch(() => null);
      if (response?.states) {
        const toolMap = { DarkMode: 'darkMode', ReaderMode: 'readerMode', FocusMode: 'focusMode',
          KeyboardNavigator: 'keyboardNav', VoiceCommands: 'voiceCommands', MotionReducer: 'motionReducer' };
        for (const [toolName, elementId] of Object.entries(toolMap)) {
          if (response.states[toolName] !== undefined) {
            const el = document.getElementById(elementId);
            if (el) el.checked = response.states[toolName];
            if (toolName === 'FocusMode' && response.states[toolName])
              document.getElementById('focusModeOptions')?.classList.add('show');
          }
        }
      }
    }
  } catch (e) {}
}

async function queryStats() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id && !tab.url?.startsWith('chrome://')) {
      const response = await chrome.tabs.sendMessage(tab.id, { type: 'getStats' }).catch(() => null);
      if (response?.success) updateFixesPanel(response.stats, response.fixes || []);
    }
  } catch (e) {}
}

function updateFixesPanel(stats, fixes) {
  const panel = document.getElementById('fixesPanel');
  if (!panel || !stats) return;

  const summary = panel.querySelector('.fixes-summary');
  const body = panel.querySelector('.fixes-body');
  const list = panel.querySelector('.fixes-list');
  if (!summary || !body || !list) return;

  const total = (stats.wcag || 0) + (stats.images || 0) + (stats.labels || 0) + (stats.text || 0) + (stats.captions || 0);
  if (total === 0) { panel.style.display = 'none'; return; }

  const parts = [];
  if (stats.wcag) parts.push(`${stats.wcag} WCAG`);
  if (stats.images) parts.push(`${stats.images} images`);
  if (stats.labels) parts.push(`${stats.labels} labels`);
  if (stats.text) parts.push(`${stats.text} text`);
  if (stats.captions) parts.push(`${stats.captions} captions`);

  summary.innerHTML = `<strong>${total} issues fixed</strong> <span>(${parts.join(', ')})</span>`;
  panel.style.display = 'block';

  if (!fixes || !Array.isArray(fixes)) fixes = [];
  list.innerHTML = fixes.map(fix => `
    <div class="fix-item">
      <span class="fix-type">${escapeHtml(fix.type)} · ${escapeHtml(fix.element)}</span>
      <span class="fix-old">${escapeHtml(fix.old)}</span>
      <span class="fix-new">${escapeHtml(fix.new)}</span>
    </div>
  `).join('');

  const header = document.getElementById('fixesHeader');
  if (header && !header._bound) {
    header._bound = true;
    const togglePanel = () => {
      panel.classList.toggle('expanded');
      const isExpanded = panel.classList.contains('expanded');
      body.style.display = isExpanded ? 'block' : 'none';
      header.setAttribute('aria-expanded', isExpanded);
    };
    header.addEventListener('click', togglePanel);
    header.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); togglePanel(); }
    });
  }
}
function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
