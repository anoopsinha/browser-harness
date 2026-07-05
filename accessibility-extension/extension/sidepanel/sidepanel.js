// sidepanel.js — hands-free voice Assistant in a Chrome side panel.
//
// Two halves that meet in the middle:
//   1) Assistant flow — identical contract to popup.js setupAssistantPanel():
//      read serviceUrl/serviceToken from storage.sync; on Run send background
//      { type:'assistantRun', prompt, activeUrl }; the live view is driven off
//      chrome.storage.local.assistant ({status,task,log,result,error}).
//   2) Voice engine — ported from extension-service/console/voice.js:
//      SpeechRecognition (STT), speechSynthesis (TTS, wedge-safe), Google/en +
//      Samantha/Karen voice picker, Ctrl+./Ctrl+, rate, Web Audio earcons.
//      The console's media pause/resume is intentionally NOT ported (it relied
//      on the console server's browser-harness endpoints).
//
// Unlike the popup, a side panel is a persistent page, so recognition, TTS and
// earcons run reliably here.
(function () {
  'use strict';

  const DEFAULT_URL = 'http://127.0.0.1:8787';

  // ---- elements ----
  const promptEl   = document.getElementById('assistantPrompt');
  const runBtn     = document.getElementById('assistantRunBtn');
  const clearBtn   = document.getElementById('assistantClearBtn');
  const statusEl   = document.getElementById('assistantStatus');
  const micBtn     = document.getElementById('micBtn');
  const micLabel   = document.getElementById('micLabel');
  const modeToggle = document.getElementById('voiceModeToggle');
  const voiceSel   = document.getElementById('voiceSel');
  const rateEl     = document.getElementById('voiceRate');
  const badge      = document.getElementById('voiceState');

  // =====================================================================
  //  Assistant flow (mirror of popup.js)
  // =====================================================================

  function renderLog(log) {
    const wrap = document.createElement('div');
    wrap.className = 'assistant-log';
    for (const e of log) {
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

  function mkHead(cls, glyph, label, spin) {
    const head = document.createElement('div');
    head.className = 'assistant-state-head ' + cls;
    if (glyph) {
      const ic = document.createElement('span');
      ic.className = spin ? 'vp-spin' : '';
      ic.setAttribute('aria-hidden', 'true');
      ic.textContent = glyph;
      head.appendChild(ic);
      head.appendChild(document.createTextNode(' '));
    }
    head.appendChild(document.createTextNode(label));
    return head;
  }

  function render(state) {
    statusEl.textContent = '';
    const status = state && state.status;
    runBtn.disabled = status === 'running';
    if (clearBtn) clearBtn.hidden = !state;
    if (!state) { statusEl.className = 'assistant-status-region'; return; }

    if (status === 'running') {
      statusEl.className = 'assistant-status-region assistant-state-running';
      statusEl.appendChild(mkHead('', '⟳', 'Working on it…', true));
    } else if (status === 'done') {
      statusEl.className = 'assistant-status-region assistant-state-done';
      statusEl.appendChild(mkHead('', '✓', 'Done'));
    } else if (status === 'error') {
      statusEl.className = 'assistant-status-region assistant-state-error';
      statusEl.appendChild(mkHead('', '', 'Something went wrong'));
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

    if (Array.isArray(state.log) && state.log.length) {
      statusEl.appendChild(renderLog(state.log));
    }

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
    statusEl.scrollTop = statusEl.scrollHeight;
  }

  // Run a task: same handoff as the popup. Shared by the button and by voice.
  async function runTask(prompt) {
    const text = (prompt || '').trim();
    if (!text) { promptEl.focus(); return; }
    promptEl.value = text;
    runBtn.disabled = true;
    render({ status: 'running', task: text, log: [] });
    let activeUrl = '';
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      activeUrl = (tab && tab.url) || '';
    } catch (e) {}
    chrome.runtime.sendMessage({ type: 'assistantRun', prompt: text, activeUrl }, () => {
      void chrome.runtime.lastError; // storage.local.assistant is the source of truth
    });
  }

  runBtn.addEventListener('click', () => runTask(promptEl.value));
  clearBtn && clearBtn.addEventListener('click', () => {
    chrome.runtime.sendMessage({ type: 'assistantClear' }, () => { void chrome.runtime.lastError; });
    render(null);
  });

  // =====================================================================
  //  Voice engine (ported from console/voice.js, media pause/resume removed)
  // =====================================================================

  let voiceMode = true;      // ON by default
  let listening = false;
  let finalTranscript = '';

  // ---------- Web Audio earcons ----------
  let ac = null;
  function audio() {
    if (!ac) ac = new (window.AudioContext || window.webkitAudioContext)();
    if (ac.state === 'suspended') ac.resume();
    return ac;
  }
  function blip(freq, dur, when, vol, type) {
    const a = audio();
    const t = a.currentTime + (when || 0);
    const o = a.createOscillator();
    const g = a.createGain();
    o.type = type || 'sine';
    o.frequency.value = freq;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.linearRampToValueAtTime(vol == null ? 0.06 : vol, t + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(a.destination);
    o.start(t);
    o.stop(t + dur + 0.02);
  }
  let thinkTimer = null;
  function thinkingStart() {
    if (!voiceMode) return;
    stopThinking();
    const pulse = () => { blip(440, 0.12, 0, 0.045); blip(620, 0.12, 0.14, 0.035); };
    pulse();
    thinkTimer = setInterval(pulse, 2400);
  }
  function stopThinking() {
    if (thinkTimer) { clearInterval(thinkTimer); thinkTimer = null; }
  }
  function chimeDone() {
    if (!voiceMode) return;
    blip(660, 0.12, 0, 0.06); blip(880, 0.18, 0.11, 0.06);
  }
  function chimeError() {
    if (!voiceMode) return;
    blip(300, 0.2, 0, 0.07, 'square'); blip(210, 0.26, 0.17, 0.06, 'square');
  }
  function blipListenOn()  { blip(880, 0.09, 0, 0.06); }
  function blipListenOff() { blip(520, 0.09, 0, 0.05); }

  // ---------- text to speech ----------
  const hasTTS = 'speechSynthesis' in window;
  let voices = [];
  // Offered voices: Google English voices, plus Samantha and Karen (en-AU).
  function keepVoice(v) {
    return (
      (/^Google/i.test(v.name) && /^en/i.test(v.lang)) ||
      v.name === 'Samantha' ||
      v.name === 'Karen'
    );
  }
  const PREFERRED_VOICES = [
    'Google US English', 'Google UK English Female', 'Google UK English Male',
    'Samantha', 'Karen',
  ];
  let selectedVoiceName = null;
  try { selectedVoiceName = localStorage.getItem('ttsVoice'); } catch (_) {}

  const RATE_MIN = 0.5, RATE_MAX = 2.0;
  let rate = 1.0;
  try { const r = parseFloat(localStorage.getItem('ttsRate')); if (!isNaN(r)) rate = r; } catch (_) {}
  function clampRate(r) { return Math.min(RATE_MAX, Math.max(RATE_MIN, Math.round(r * 10) / 10)); }

  function refreshVoices() {
    if (!hasTTS) return;
    try { voices = speechSynthesis.getVoices() || []; } catch (_) {}
  }
  if (hasTTS) {
    refreshVoices();
    try { speechSynthesis.onvoiceschanged = () => { refreshVoices(); populateVoiceSelect(); }; } catch (_) {}
    try { speechSynthesis.cancel(); } catch (_) {} // clear any stale/stuck state on load
  }
  function resolveVoice() {
    if (!voices.length) refreshVoices();
    const kept = voices.filter(keepVoice);
    if (selectedVoiceName) {
      const v = kept.find((x) => x.name === selectedVoiceName);
      if (v) return v;
    }
    for (const name of PREFERRED_VOICES) {
      const v = kept.find((x) => x.name === name);
      if (v) return v;
    }
    return kept[0] || null;
  }

  function cleanForSpeech(s) {
    return (s || '').replace(/[`*_#>|]+/g, '').replace(/\s+/g, ' ').trim();
  }
  let lastUtter = null;  // keep a ref so Chrome doesn't GC the utterance mid-speech
  let lastText = '';     // last spoken text, for re-speaking on a rate change
  let keepAlive = null;
  function stopKeepAlive() { if (keepAlive) { clearInterval(keepAlive); keepAlive = null; } }
  function doSpeak(t) {
    lastText = t;
    const u = new SpeechSynthesisUtterance(t);
    u.rate = rate;
    u.lang = 'en-US';
    const v = resolveVoice();
    if (v) u.voice = v;
    u.onend = () => { stopKeepAlive(); };
    u.onerror = (e) => {
      stopKeepAlive();
      const err = (e && e.error) || '';
      if (err && err !== 'interrupted' && err !== 'canceled') setBadge('tts: ' + err);
    };
    lastUtter = u;
    try { speechSynthesis.resume(); } catch (_) {}
    speechSynthesis.speak(u);
    // Chrome auto-pauses long utterances (~15s) and can wedge; nudge it awake.
    stopKeepAlive();
    keepAlive = setInterval(() => {
      if (!speechSynthesis.speaking) return stopKeepAlive();
      try { speechSynthesis.resume(); } catch (_) {}
    }, 8000);
  }
  function speak(text, force) {
    if (!voiceMode && !force) return;
    if (!hasTTS) { setBadge('no text-to-speech in this browser'); return; }
    const t = cleanForSpeech(text);
    if (!t) return;
    try { speechSynthesis.cancel(); } catch (_) {} // clear prior/stuck utterance
    try { speechSynthesis.resume(); } catch (_) {}
    // Let cancel() settle a tick — a same-tick speak() after cancel() is dropped.
    // NOTE: no empty/whitespace warm-up utterance — that wedges Chrome.
    setTimeout(() => doSpeak(t), 100);
  }
  function stopSpeaking() {
    if (!hasTTS) return;
    stopKeepAlive();
    try { speechSynthesis.cancel(); } catch (_) {}
  }

  // ---------- speech recognition ----------
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let rec = null;
  function makeRec() {
    if (!SR) return null;
    const r = new SR();
    r.lang = 'en-US';
    r.continuous = true;
    r.interimResults = true;
    r.onresult = (e) => {
      let interim = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const chunk = e.results[i][0].transcript;
        if (e.results[i].isFinal) finalTranscript += chunk + ' ';
        else interim += chunk;
      }
      promptEl.value = (finalTranscript + interim).replace(/\s+/g, ' ').replace(/^\s+/, '');
    };
    r.onend = () => {
      // Chrome ends recognition on pauses; keep going until the user stops.
      if (listening) { try { r.start(); } catch (_) {} }
    };
    r.onerror = (e) => {
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
        listening = false;
        setBadge('mic blocked — allow microphone access');
        renderMic();
      }
    };
    return r;
  }

  function startListening() {
    stopSpeaking(); // the input hotkey interrupts any playing speech
    if (!SR) { setVoiceMode(true); setBadge('no speech recognition in this browser'); return; }
    if (!voiceMode) setVoiceMode(true);
    if (listening) return;
    finalTranscript = '';
    promptEl.value = '';
    if (!rec) rec = makeRec();
    try { rec.start(); } catch (_) {}
    listening = true;
    setBadge('listening — Ctrl+M to send, Esc to cancel');
    blipListenOn();
    renderMic();
  }
  function stopListening(submit) {
    if (!listening) return;
    listening = false;
    try { rec && rec.stop(); } catch (_) {}
    blipListenOff();
    setBadge('');
    renderMic();
    const text = (promptEl.value || finalTranscript).trim();
    if (submit && text) {
      // transcript stays in the textarea AND is auto-submitted as the task
      runTask(text);
    }
  }
  function toggleTalk() { listening ? stopListening(true) : startListening(); }

  // ---------- state / UI ----------
  function setVoiceMode(on) {
    voiceMode = on;
    if (modeToggle) modeToggle.checked = on;
    try { localStorage.setItem('voiceMode', on ? 'on' : 'off'); } catch (_) {}
    if (on) {
      audio(); // resume AudioContext on this user gesture
      refreshVoices();
      // Clear any stuck engine state. Do NOT speak an empty/whitespace utterance
      // to "warm up" — that wedges Chrome at speaking=true and blocks every reply.
      if (hasTTS) { try { speechSynthesis.cancel(); } catch (_) {} }
    } else {
      stopListening(false);
      stopThinking();
      stopSpeaking();
      setBadge('');
    }
    renderMic();
  }
  function setBadge(s) { if (badge) badge.textContent = s || ''; }
  function updateRateUI() { if (rateEl) rateEl.textContent = rate.toFixed(1) + '×'; }
  function setRate(r) {
    rate = clampRate(r);
    try { localStorage.setItem('ttsRate', String(rate)); } catch (_) {}
    updateRateUI();
    setBadge('rate ' + rate.toFixed(1) + '×');
    setTimeout(() => setBadge(''), 900);
    // if a reply is playing, restart it at the new rate for immediate feedback
    if (hasTTS && speechSynthesis.speaking && lastText) speak(lastText, true);
  }
  function populateVoiceSelect() {
    if (!voiceSel) return;
    refreshVoices();
    const list = voices.filter(keepVoice);
    const resolved = resolveVoice();
    voiceSel.innerHTML = '';
    const auto = document.createElement('option');
    auto.value = '';
    auto.textContent = 'auto: ' + (resolved ? resolved.name : 'default');
    voiceSel.appendChild(auto);
    list.forEach((v) => {
      const o = document.createElement('option');
      o.value = v.name;
      o.textContent = v.name.replace(/\s*\((Premium|Enhanced)\)/, ' ★') + ' · ' + v.lang;
      if (v.name === selectedVoiceName) o.selected = true;
      voiceSel.appendChild(o);
    });
  }
  function renderMic() {
    if (!micBtn) return;
    micBtn.classList.toggle('on', voiceMode);
    micBtn.classList.toggle('listening', listening);
    micBtn.setAttribute('aria-pressed', listening ? 'true' : 'false');
    if (micLabel) {
      micLabel.textContent = listening ? 'listening…' : voiceMode ? 'voice on' : 'voice off';
    }
    if (micBtn.querySelector('.vp-mic-glyph')) {
      micBtn.querySelector('.vp-mic-glyph').textContent = listening ? '🔴' : '🎙';
    }
  }

  // ---------- wiring ----------
  micBtn && micBtn.addEventListener('click', toggleTalk);
  modeToggle && modeToggle.addEventListener('change', () => setVoiceMode(modeToggle.checked));
  voiceSel && voiceSel.addEventListener('change', () => {
    selectedVoiceName = voiceSel.value || null;
    try {
      if (selectedVoiceName) localStorage.setItem('ttsVoice', selectedVoiceName);
      else localStorage.removeItem('ttsVoice');
    } catch (_) {}
    speak('Hi — this is how I sound.', true); // preview the chosen voice
  });

  const isTalkKey = (e) =>
    e.ctrlKey && !e.metaKey && !e.altKey && (e.code === 'KeyM' || e.key === 'm');
  document.addEventListener('keydown', (e) => {
    if (isTalkKey(e)) {
      e.preventDefault();
      toggleTalk();
    } else if (e.ctrlKey && !e.metaKey && !e.altKey && (e.code === 'Period' || e.key === '.')) {
      e.preventDefault();
      setRate(rate + 0.1); // faster
    } else if (e.ctrlKey && !e.metaKey && !e.altKey && (e.code === 'Comma' || e.key === ',')) {
      e.preventDefault();
      setRate(rate - 0.1); // slower
    } else if (e.key === 'Escape') {
      if (listening) { e.preventDefault(); stopListening(false); }
      stopSpeaking();
    }
  });

  // =====================================================================
  //  Bridge: assistant state transitions → voice earcons + speech
  // =====================================================================
  let lastStatus = null;
  function applyState(state) {
    render(state);
    const status = state && state.status;
    if (status === lastStatus) return;
    if (status === 'running') {
      thinkingStart();
    } else if (status === 'done') {
      stopThinking();
      chimeDone();
      const t = (state && state.result || '').trim();
      if (t) speak(t); // interruptible via Ctrl+M / Esc
    } else if (status === 'error') {
      stopThinking();
      chimeError();
      const err = (state && state.error || '').trim();
      if (err) speak('Something went wrong. ' + err);
    }
    lastStatus = status;
  }

  chrome.storage.local.get('assistant', (d) => {
    const st = d.assistant || null;
    render(st);                       // paint without firing earcons on first load
    lastStatus = st ? st.status : null;
  });
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'local' && changes.assistant) applyState(changes.assistant.newValue || null);
  });

  // =====================================================================
  //  Init
  // =====================================================================
  // Voice mode ON by default (but remember an explicit off).
  let startOn = true;
  try { startOn = localStorage.getItem('voiceMode') !== 'off'; } catch (_) {}
  voiceMode = !!(startOn && hasTTS);
  if (voiceMode) refreshVoices();
  if (modeToggle) modeToggle.checked = voiceMode;
  renderMic();
  populateVoiceSelect();
  updateRateUI();
})();
