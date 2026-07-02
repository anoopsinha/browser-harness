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
  function refreshVoices() {
    if (!hasTTS) return;
    try { voices = speechSynthesis.getVoices() || []; } catch (_) {}
  }
  if (hasTTS) {
    refreshVoices();
    try { speechSynthesis.onvoiceschanged = refreshVoices; } catch (_) {}
  }
  function pickVoice() {
    if (!voices.length) refreshVoices();
    return (
      voices.find((v) => v.default && /en/i.test(v.lang)) ||
      voices.find((v) => /en[-_]/i.test(v.lang)) ||
      voices[0] ||
      null
    );
  }

  function cleanForSpeech(s) {
    return (s || "").replace(/[`*_#>|]+/g, "").replace(/\s+/g, " ").trim();
  }
  let lastUtter = null; // keep a ref so Chrome doesn't GC the utterance mid-speech
  function doSpeak(t) {
    const u = new SpeechSynthesisUtterance(t);
    u.rate = 1.03;
    u.lang = "en-US";
    const v = pickVoice();
    if (v) u.voice = v;
    u.onerror = (e) => {
      const err = (e && e.error) || "";
      if (err && err !== "interrupted" && err !== "canceled") setBadge("tts: " + err);
    };
    lastUtter = u;
    try { speechSynthesis.resume(); } catch (_) {} // Chrome sometimes auto-pauses
    speechSynthesis.speak(u);
  }
  function speak(text, force) {
    if (!voiceMode && !force) return;
    if (!hasTTS) { setBadge("no text-to-speech in this browser"); return; }
    const t = cleanForSpeech(text);
    if (!t) return;
    const wasSpeaking = speechSynthesis.speaking || speechSynthesis.pending;
    try { speechSynthesis.cancel(); } catch (_) {}
    // Calling speak() in the same tick as cancel() drops the utterance in Chrome.
    if (wasSpeaking) setTimeout(() => doSpeak(t), 120);
    else doSpeak(t);
  }
  function stopSpeaking() {
    if (!hasTTS) return;
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
    if (on) {
      audio(); // resume AudioContext on this user gesture
      refreshVoices();
      // warm up the TTS engine inside the gesture so the first real reply speaks
      if (hasTTS) {
        try {
          speechSynthesis.cancel();
          const w = new SpeechSynthesisUtterance(" ");
          w.volume = 0;
          speechSynthesis.speak(w);
        } catch (_) {}
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
  };

  render();
})();
