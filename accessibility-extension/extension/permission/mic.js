const go = document.getElementById('go');
const msg = document.getElementById('msg');

async function ask() {
  msg.textContent = 'Requesting microphone… choose Allow in Chrome’s prompt.';
  msg.className = '';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach((t) => t.stop()); // we only needed the grant
    msg.textContent = '✅ Microphone enabled. Close this tab and return to the Voice Assistant, then press the mic (or Ctrl+M).';
    msg.className = 'ok';
    go.hidden = true;
    setTimeout(() => { try { window.close(); } catch (_) {} }, 2200);
  } catch (e) {
    msg.textContent = '⚠️ Microphone blocked (' + (e && e.name || 'error') +
      '). Click Enable again and choose Allow, or turn on the mic for this extension in Chrome site settings.';
    msg.className = 'err';
  }
}

go.addEventListener('click', ask);
