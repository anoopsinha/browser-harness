// App-owned AI provider bridge.
//
// The accessibility adapters now come from the ai-for-accessibility-toolkit
// package (single source of truth). Those adapters call the toolkit's own
// provider abstraction (toolkit/tools/utils/ai.js) via setAIProvider(). This
// file supplies the ONE piece that stays app-specific: a provider object that
// bridges every AI capability the 44 toolkit adapters need to THIS extension's
// background service worker.
//
// This extension's background exposes a single generic `gemini` handler
// ({ prompt, images } -> text), so — unlike the toolkit's basic extension,
// which has one background handler per task — each method below builds its own
// prompt client-side and sends it through that one channel. Prompt wording is
// ported from the toolkit's reference handlers so behavior matches.

export function createChromeAIProvider() {
  // Generic channel to background.js's `gemini` handler. `images` is an array
  // of data URLs / base64 frames attached as multimodal inlineData.
  function ask(prompt, images) {
    return new Promise((resolve, reject) => {
      const msg = { type: 'gemini', prompt };
      if (images && images.length) msg.images = images;
      chrome.runtime.sendMessage(msg, (response) => {
        if (chrome.runtime.lastError) return reject(new Error(chrome.runtime.lastError.message));
        if (response?.error) return reject(new Error(response.error));
        resolve((response?.result || '').trim());
      });
    });
  }

  // Pull the first JSON value out of a model reply that may be fenced or prosy.
  function parseJson(text) {
    if (!text) return null;
    const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
    const body = fenced ? fenced[1] : text;
    const start = body.search(/[[{]/);
    if (start === -1) return null;
    try { return JSON.parse(body.slice(start)); } catch { return null; }
  }

  return {
    // ── Images / vision ────────────────────────────────────────────────────
    async describeImage(imageData) {
      return ask(
        'Describe this image concisely for use as alt text on a webpage. Be specific and brief (under 125 characters). Return ONLY the alt text, no preamble.',
        [imageData]
      );
    },

    async describeVideo(frames /*, metadata */) {
      return ask(
        'These are sequential frames from a video. Describe what happens in the video in 1-2 sentences for accessibility. Return ONLY the description.',
        frames
      );
    },

    async describeElement(imageData, elementType, context) {
      const ctx = context ? ` It is a ${elementType || 'UI element'} in this context: ${context}.` : ` It is a ${elementType || 'UI element'}.`;
      return ask(
        `Describe this on-screen element for a screen-reader user in one short, useful sentence.${ctx} Return ONLY the description.`,
        imageData ? [imageData] : undefined
      );
    },

    async extractChartData(imageData, context) {
      const hint = context ? ` Context: ${context}.` : '';
      const out = await ask(
        `This image is a chart, graph, or diagram.${hint} Extract its underlying data as JSON of the shape ` +
        `{"title": string, "headers": string[], "rows": string[][]}. Use concise cell values. ` +
        'If it is not a data chart, return {"headers":[],"rows":[]}. Return ONLY the JSON.',
        [imageData]
      );
      return parseJson(out);
    },

    // ── Text ───────────────────────────────────────────────────────────────
    async simplifyText(text) {
      return ask(`Simplify the following text to about a 6th-grade reading level. Keep the meaning; use shorter words and sentences. Return ONLY the simplified text.\n\n${text}`);
    },

    async summarizeText(text) {
      return ask(`Summarize the following text in 2-3 concise sentences. Return ONLY the summary.\n\n${text}`);
    },

    async translateText(text, targetLang) {
      return ask(`Translate the following text into ${targetLang || 'English'}. Preserve meaning and tone. Return ONLY the translation, no notes.\n\n${text}`);
    },

    async defineWord(word, context) {
      const c = context ? ` As used in: "${context}".` : '';
      return ask(`Give a short, plain-language definition (one sentence, under 20 words) of "${word}".${c} Return ONLY the definition.`);
    },

    // ── Labels / links / tables ──────────────────────────────────────────────
    // generate-labels and fix-links both call these with an object context.
    async inferLabel(ctx = {}) {
      const { elementType, html, context } = ctx;
      let p = `Generate a short, descriptive accessible name (2-6 words) for a ${elementType || 'control'} on a web page.`;
      if (html) p += `\nElement HTML: ${String(html).slice(0, 500)}`;
      if (context) p += `\nSurrounding context: ${String(context).slice(0, 300)}`;
      p += '\nReturn ONLY the label text.';
      return ask(p);
    },

    // Toolkit's provider treats generateLabels and inferLabel identically.
    async generateLabels(ctx = {}) {
      return this.inferLabel(ctx);
    },

    async improveLinkText(linkText, href, context) {
      let p = `A link's visible text is "${linkText || '(empty)'}"`;
      if (href) p += ` and it points to ${href}`;
      if (context) p += `. Nearby context: ${String(context).slice(0, 300)}`;
      p += '. Write a clearer, self-describing accessible name for this link (under 8 words) so a screen-reader user knows where it goes. Return ONLY the label text.';
      return ask(p);
    },

    async inferColumnHeader(sampleData) {
      const samples = Array.isArray(sampleData) ? sampleData.slice(0, 8).join(', ') : String(sampleData || '');
      return ask(`These are sample values from one column of a data table: ${samples}. Give a short column header (1-3 words) that names what they represent. Return ONLY the header.`);
    },

    // ── Contrast ─────────────────────────────────────────────────────────────
    async fixContrast(foreground, background) {
      const out = await ask(`Foreground color "${foreground}" on background "${background}" fails WCAG AA contrast. Suggest the closest foreground color that reaches 4.5:1 while staying visually similar. Return ONLY a hex code like #1a2b3c.`);
      const m = out && out.match(/#[0-9a-fA-F]{6}/);
      return m ? m[0] : null;
    },

    // ── Media transcription ─────────────────────────────────────────────────
    // The generic gemini channel can't transcribe audio/fetch YouTube tracks,
    // so these degrade to null — the captions adapter falls back to native CC.
    async getYouTubeTranscript() { return null; },
    async transcribeVideo() { return null; },
    async transcribeAudio() { return null; },

    // ── Live-region announcements (no AI) ────────────────────────────────────
    announce(message) {
      let region = document.getElementById('ai4a11y-announcer');
      if (!region) {
        region = document.createElement('div');
        region.id = 'ai4a11y-announcer';
        region.setAttribute('role', 'status');
        region.setAttribute('aria-live', 'polite');
        region.setAttribute('aria-atomic', 'true');
        region.style.cssText = 'position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0;';
        (document.body || document.documentElement).appendChild(region);
      }
      region.textContent = message;
    },
  };
}
