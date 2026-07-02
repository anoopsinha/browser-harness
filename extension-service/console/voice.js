// voice.js — optional voice input/output + "thinking" earcons for the console.
//
// Everything here is OFF until voice mode is enabled (button or Ctrl+M), so the
// typed console behaves exactly as before. Uses only native browser APIs:
//   - SpeechRecognition (STT), SpeechSynthesis (TTS), Web Audio (earcons).
//
// Hotkeys:
//   Ctrl+M  toggle listening (start → stop+submit); first use turns voice on;
//           pressing it also interrupts any speech that's playing.
//   Esc     stop listening without submitting, and interrupt speech.
(function () {
  const cmd = document.getElementById("cmd");
  const btn = document.getElementById("voiceBtn");
  const badge = document.getElementById("voiceState");

  let voiceMode = false;
  let listening = false;
  let finalTranscript = "";

  const isTalkKey = (e) =>
    e.ctrlKey && !e.metaKey && !e.altKey && (e.code === "KeyM" || e.key === "m");

  // ---------- Web Audio earcons ----------
  let ac = null;
  function audio() {
    if (!ac) ac = new (window.AudioContext || window.webkitAudioContext)();
    if (ac.state === "suspended") ac.resume();
    return ac;
  }
  function blip(freq, dur, when, vol, type) {
    const a = audio();
    const t = a.currentTime + (when || 0);
    const o = a.createOscillator();
    const g = a.createGain();
    o.type = type || "sine";
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
    const pulse = () => {
      blip(440, 0.12, 0, 0.045);
      blip(620, 0.12, 0.14, 0.035);
    };
    pulse();
    thinkTimer = setInterval(pulse, 2400);
  }
  function stopThinking() {
    if (thinkTimer) {
      clearInterval(thinkTimer);
      thinkTimer = null;
    }
  }
  function chimeDone() {
    if (!voiceMode) return;
    blip(660, 0.12, 0, 0.06);
    blip(880, 0.18, 0.11, 0.06);
  }
  function chimeError() {
    if (!voiceMode) return;
    blip(300, 0.2, 0, 0.07, "square");
    blip(210, 0.26, 0.17, 0.06, "square");
  }
  function blipListenOn() { blip(880, 0.09, 0, 0.06); }
  function blipListenOff() { blip(520, 0.09, 0, 0.05); }

  // ---------- text to speech ----------
  const hasTTS = "speechSynthesis" in window;
  let voices = [];
  // macOS "Enhanced"/"Premium" voices sound far nicer than the default Samantha;
  // prefer them when installed, else fall back to a decent default.
  const PREFERRED_VOICES = [
    "Ava (Premium)", "Ava (Enhanced)", "Zoe (Premium)", "Zoe (Enhanced)",
    "Evan (Enhanced)", "Nathan (Enhanced)", "Samantha (Enhanced)",
    "Allison (Enhanced)", "Serena (Premium)", "Serena (Enhanced)",
    "Ava", "Allison", "Serena", "Samantha", "Karen", "Moira", "Tessa", "Daniel",
  ];
  let selectedVoiceName = null;
  try { selectedVoiceName = localStorage.getItem("ttsVoice"); } catch (_) {}

  function refreshVoices() {
    if (!hasTTS) return;
    try { voices = speechSynthesis.getVoices() || []; } catch (_) {}
  }
  if (hasTTS) {
    refreshVoices();
    try {
      speechSynthesis.onvoiceschanged = () => { refreshVoices(); populateVoiceSelect(); };
    } catch (_) {}
    try { speechSynthesis.cancel(); } catch (_) {} // clear any stale/stuck state on load
  }
  function resolveVoice() {
    if (!voices.length) refreshVoices();
    if (selectedVoiceName) {
      const v = voices.find((x) => x.name === selectedVoiceName);
      if (v) return v;
    }
    for (const name of PREFERRED_VOICES) {
      const v = voices.find((x) => x.name === name);
      if (v) return v;
    }
    return (
      voices.find((v) => v.default && /^en/i.test(v.lang)) ||
      voices.find((v) => /^en/i.test(v.lang)) ||
      voices[0] ||
      null
    );
  }

  function cleanForSpeech(s) {
    return (s || "").replace(/[`*_#>|]+/g, "").replace(/\s+/g, " ").trim();
  }
  let lastUtter = null; // keep a ref so Chrome doesn't GC the utterance mid-speech
  let keepAlive = null;
  function stopKeepAlive() {
    if (keepAlive) { clearInterval(keepAlive); keepAlive = null; }
  }
  function doSpeak(t) {
    const u = new SpeechSynthesisUtterance(t);
    u.rate = 1.03;
    u.lang = "en-US";
    const v = resolveVoice();
    if (v) u.voice = v;
    u.onend = stopKeepAlive;
    u.onerror = (e) => {
      stopKeepAlive();
      const err = (e && e.error) || "";
      if (err && err !== "interrupted" && err !== "canceled") setBadge("tts: " + err);
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
    if (!hasTTS) { setBadge("no text-to-speech in this browser"); return; }
    const t = cleanForSpeech(text);
    if (!t) return;
    try { speechSynthesis.cancel(); } catch (_) {} // clear prior/stuck utterance
    try { speechSynthesis.resume(); } catch (_) {}
    // Always let cancel() settle a tick — a same-tick speak() after cancel()
    // is dropped by Chrome.
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
    r.lang = "en-US";
    r.continuous = true;
    r.interimResults = true;
    r.onresult = (e) => {
      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const chunk = e.results[i][0].transcript;
        if (e.results[i].isFinal) finalTranscript += chunk + " ";
        else interim += chunk;
      }
      cmd.value = (finalTranscript + interim).replace(/\s+/g, " ").replace(/^\s+/, "");
    };
    r.onend = () => {
      // Chrome ends recognition on pauses; keep going until the user stops.
      if (listening) {
        try { r.start(); } catch (_) {}
      }
    };
    r.onerror = (e) => {
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        listening = false;
        setBadge("mic blocked — allow microphone access");
        render();
      }
    };
    return r;
  }

  function startListening() {
    stopSpeaking(); // input hotkey interrupts any playing speech
    if (!SR) {
      setVoiceMode(true);
      setBadge("no speech recognition in this browser");
      return;
    }
    if (!voiceMode) setVoiceMode(true);
    if (listening) return;
    finalTranscript = "";
    cmd.value = "";
    if (!rec) rec = makeRec();
    try { rec.start(); } catch (_) {}
    listening = true;
    setBadge("listening — Ctrl+M to send, Esc to cancel");
    blipListenOn();
    render();
  }
  function stopListening(submit) {
    if (!listening) return;
    listening = false;
    try { rec && rec.stop(); } catch (_) {}
    blipListenOff();
    setBadge("");
    render();
    const text = (cmd.value || finalTranscript).trim();
    if (submit && text) {
      cmd.value = "";
      window.Console && window.Console.submit(text);
    }
  }
  function toggleTalk() {
    listening ? stopListening(true) : startListening();
  }

  // ---------- state / UI ----------
  function setVoiceMode(on) {
    voiceMode = on;
    try { localStorage.setItem("voiceMode", on ? "on" : "off"); } catch (_) {}
    if (on) {
      audio(); // resume AudioContext on this user gesture
      refreshVoices();
      // Clear any stuck engine state. Do NOT speak an empty/whitespace utterance
      // to "warm up" — that wedges Chrome at speaking=true and blocks every reply.
      if (hasTTS) {
        try { speechSynthesis.cancel(); } catch (_) {}
      }
    } else {
      stopListening(false);
      stopThinking();
      stopSpeaking();
      setBadge("");
    }
    render();
  }
  function setBadge(s) { if (badge) badge.textContent = s || ""; }
  function populateVoiceSelect() {
    const sel = document.getElementById("voiceSel");
    if (!sel) return;
    refreshVoices();
    const en = voices.filter((v) => /^en/i.test(v.lang));
    const list = en.length ? en : voices;
    const resolved = resolveVoice();
    sel.innerHTML = "";
    const auto = document.createElement("option");
    auto.value = "";
    auto.textContent = "auto: " + (resolved ? resolved.name : "default");
    sel.appendChild(auto);
    list.forEach((v) => {
      const o = document.createElement("option");
      o.value = v.name;
      // mark the nicer Enhanced/Premium voices with a star
      o.textContent = v.name.replace(/\s*\((Premium|Enhanced)\)/, " ★") + " · " + v.lang;
      if (v.name === selectedVoiceName) o.selected = true;
      sel.appendChild(o);
    });
  }
  function render() {
    if (!btn) return;
    btn.classList.toggle("on", voiceMode);
    btn.classList.toggle("listening", listening);
    btn.textContent = listening
      ? "🔴 listening…"
      : voiceMode
      ? "🎙 voice on"
      : "🎙 voice off";
  }

  // ---------- wiring ----------
  if (btn) btn.addEventListener("click", () => setVoiceMode(!voiceMode));
  const voiceSel = document.getElementById("voiceSel");
  if (voiceSel) {
    voiceSel.addEventListener("change", () => {
      selectedVoiceName = voiceSel.value || null;
      try {
        if (selectedVoiceName) localStorage.setItem("ttsVoice", selectedVoiceName);
        else localStorage.removeItem("ttsVoice");
      } catch (_) {}
      speak("Hi — this is how I sound.", true); // preview the chosen voice
    });
  }
  document.addEventListener("keydown", (e) => {
    if (isTalkKey(e)) {
      e.preventDefault();
      toggleTalk();
    } else if (e.key === "Escape") {
      if (listening) {
        e.preventDefault();
        stopListening(false);
      }
      stopSpeaking();
    }
  });

  // hooks called by app.js (no-ops unless voice mode is on)
  window.Voice = {
    onThinkingStart: thinkingStart,
    onThinkingStop: stopThinking,
    onResult: (text) => {
      stopThinking();
      chimeDone();
      speak(text);
    },
    onError: () => {
      stopThinking();
      chimeError();
    },
    // test hook: speak regardless of voice mode (used by the :say command)
    say: (t) => speak(t && t.trim() ? t : "Voice check. One, two, three.", true),
    diag: diag,
    reset: () => {
      stopKeepAlive();
      try { speechSynthesis.cancel(); } catch (_) {}
      log("tts reset (cancelled + cleared). If still stuck, reload the page.");
    },
  };

  function log(m) {
    if (window.Console && window.Console.log) window.Console.log(m);
  }
  function diag() {
    refreshVoices();
    const lines = ["voice diagnostics:", "  hasTTS = " + hasTTS, "  voices = " + voices.length];
    voices.slice(0, 8).forEach((v) =>
      lines.push(
        "    · " + v.name + " [" + v.lang + "]" +
          (v.default ? " default" : "") + (v.localService ? " local" : " remote")
      )
    );
    if (hasTTS)
      lines.push(
        "  state: speaking=" + speechSynthesis.speaking +
          " pending=" + speechSynthesis.pending +
          " paused=" + speechSynthesis.paused
      );
    log(lines.join("\n"));
    if (!hasTTS) return;
    try { speechSynthesis.cancel(); } catch (_) {}
    const u = new SpeechSynthesisUtterance("Diagnostic. One two three.");
    u.lang = "en-US";
    u.onstart = () => log("  tts onstart ✓ (audio should be playing)");
    u.onend = () => log("  tts onend ✓");
    u.onerror = (e) => log("  tts onerror ✗ : " + ((e && e.error) || "?"));
    lastUtter = u;
    try { speechSynthesis.resume(); } catch (_) {}
    speechSynthesis.speak(u);
    log("  called speak() — watch for onstart/onend/onerror above…");
  }

  // default voice mode ON (but remember an explicit off), then fill the voice list
  let startOn = true;
  try { startOn = localStorage.getItem("voiceMode") !== "off"; } catch (_) {}
  if (startOn && hasTTS) {
    voiceMode = true;
    refreshVoices();
  }
  render();
  populateVoiceSelect();
})();
