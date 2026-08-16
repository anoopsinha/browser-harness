// Content script — single-sourced on the ai-for-accessibility-toolkit.
//
// The accessibility adapters AND the auditor→fixer orchestration now come from
// the toolkit package (one source of truth). This file keeps only what is
// app-specific: the Chrome AI provider bridge (utils/ai.js), the Librarian /
// site-classification / custom-profile integration, the popup messaging
// protocol, and stats reporting.

import { setAIProvider } from 'ai-for-accessibility-toolkit/tools/utils/ai.js';
import { loadSettings, getSettings, isEnabled, updateSettings }
  from 'ai-for-accessibility-toolkit/tools/profiles/settings.js';
import { clearAllMarks } from 'ai-for-accessibility-toolkit/tools/utils/dom.js';
import { runAxeAnalysis, getElementFromNode }
  from 'ai-for-accessibility-toolkit/tools/auditors/wcag-issues.js';
import { findEmptyAltImages, findCanvasElements }
  from 'ai-for-accessibility-toolkit/tools/auditors/missing-alt.js';
import { findAmbiguousLinks }
  from 'ai-for-accessibility-toolkit/tools/auditors/missing-labels.js';
import {
  getAxeHandler,
  generateImageAlt, generateCanvasDescription, generateVideoDescription,
  simplifyText, summarizeContent,
  fixTargetBlank, fixPositiveTabindex, improveAmbiguousLinks, fixAllTables, fixLandmarks,
  VisualAssist, DarkMode, MotionReducer, FocusMode, ReadAloud, ReaderMode, VoiceCommands,
  KeyboardNavigator, ColorBlindMode, AutoTranscriber, DismissOverlays, BigTargets,
  LinkHighlighter, PageOutline, BionicReading, UnpinSticky, TranslatePage, MuteSounds,
  DefineWords, StopAutoAdvance, ReduceBrightness, SoundVisualizer, LiveRegionAnnouncer,
  Magnifier, FlashGuard, DescribeOnDemand, ReflowColumn, FocusLocator, PersistentHover,
  ReadingRuler, ConfirmActions, ReadingSpot, AbbreviationExpand, LanguageTag, ExploreAChart,
  SpaFocus, SkipLinks, MathA11y, AgentWatch,
} from 'ai-for-accessibility-toolkit/tools/adapters/index.js';
import { createChromeAIProvider } from '../../utils/ai.js';

// Bridge the toolkit adapters' AI calls to this extension's background worker.
setAIProvider(createChromeAIProvider());

// ── Stats: the toolkit adapters report through these globals ────────────────
const stats = { wcag: 0, images: 0, labels: 0, text: 0, captions: 0 };
const fixes = [];
globalThis.ai4a11yIncrementStat = (key) => { if (key in stats) stats[key]++; };
globalThis.ai4a11yLogFix = (type, element, oldVal, newVal) => {
  fixes.push({
    type,
    element: typeof element === 'string' ? element : (element?.tagName?.toLowerCase() || ''),
    old: oldVal || '', new: newVal || '',
  });
  chrome.runtime.sendMessage({ type: 'fixAdded', stats: { ...stats }, fixes: [...fixes] }).catch(() => {});
};

// ── Class adapters the popup / profiles toggle by name ──────────────────────
const TOOL_MAP = {
  DarkMode, FocusMode, VisualAssist, MotionReducer, ReaderMode,
  ColorBlindMode, KeyboardNavigator, VoiceCommands, ReadAloud, AutoTranscriber,
  DismissOverlays, BigTargets, LinkHighlighter, PageOutline, BionicReading,
  UnpinSticky, TranslatePage, MuteSounds, DefineWords, StopAutoAdvance,
  ReduceBrightness, SoundVisualizer, LiveRegionAnnouncer, Magnifier, FlashGuard,
  DescribeOnDemand, ReflowColumn, FocusLocator, PersistentHover, ReadingRuler,
  ConfirmActions, ReadingSpot, AbbreviationExpand, LanguageTag, ExploreAChart,
  SpaFocus, SkipLinks, MathA11y, AgentWatch,
};

const ALL_ADAPTERS = Object.values(TOOL_MAP);
let isRunning = false;
let scanTimer = null;

// ── Settings: bridge chrome.storage.sync into the toolkit settings module ───
// The toolkit ships fixContrast ON by default (for its own basic extension).
// This app never rewrote text colors on load, and doing so — the fixer recolors
// low-contrast text white/black by background luminance — surprised users on
// startup. Keep it OFF unless explicitly enabled. (autoWcagFix/autoFixLabels/
// autoDescribe stay at the toolkit's ON defaults, matching this app's prior
// startup behavior; none of them recolor visible text.) This default sits UNDER
// the user's stored settings, so anything they or a profile turned on wins.
const APP_DEFAULTS = {
  fixContrast: false,
};

async function loadAppSettings() {
  return loadSettings(async () => ({ ...APP_DEFAULTS, ...(await chrome.storage.sync.get(null)) }));
}

// ── Visual/preference layer: enable every adapter its setting turns on ──────
function applyVisualSettings(settings) {
  const visualOptions = {};
  if (settings.contrastMode !== undefined) visualOptions.contrastMode = settings.contrastMode || 'none';
  if (settings.fontScale !== undefined) visualOptions.fontScale = settings.fontScale;
  if (settings.lineHeight !== undefined) visualOptions.lineHeight = settings.lineHeight;
  if (settings.letterSpacing !== undefined) visualOptions.letterSpacing = settings.letterSpacing;
  if (settings.largeCursor) visualOptions.largeCursor = true;
  if (settings.enhanceFocus) visualOptions.enhanceFocus = true;
  if (settings.dyslexiaFont) visualOptions.dyslexiaFont = true;
  if (settings.readingGuide) visualOptions.readingGuide = true;

  const hasVA =
    (visualOptions.contrastMode && visualOptions.contrastMode !== 'none') ||
    (visualOptions.fontScale && visualOptions.fontScale !== 100) ||
    (visualOptions.lineHeight && visualOptions.lineHeight !== 1.5) ||
    (visualOptions.letterSpacing && visualOptions.letterSpacing !== 0) ||
    visualOptions.largeCursor || visualOptions.enhanceFocus ||
    visualOptions.dyslexiaFont || visualOptions.readingGuide;
  if (hasVA) VisualAssist.enable(visualOptions);

  const colorMode = settings.colorFilter || settings.colorBlindMode;
  if (colorMode && colorMode !== 'none') ColorBlindMode.enable(colorMode);

  if (settings.focusMode) {
    FocusMode.enable({ hideDistractions: settings.hideDistractions, showProgress: settings.showProgress });
  }

  const flags = [
    ['darkMode', DarkMode], ['motionReducer', MotionReducer], ['readerMode', ReaderMode],
    ['dismissOverlays', DismissOverlays], ['bigTargets', BigTargets], ['highlightLinks', LinkHighlighter],
    ['pageOutline', PageOutline], ['bionicReading', BionicReading], ['unpinSticky', UnpinSticky],
    ['muteSounds', MuteSounds], ['defineWords', DefineWords], ['stopAutoAdvance', StopAutoAdvance],
    ['reduceBrightness', ReduceBrightness], ['soundVisualizer', SoundVisualizer],
    ['announceUpdates', LiveRegionAnnouncer], ['magnifier', Magnifier], ['flashGuard', FlashGuard],
    ['describeOnDemand', DescribeOnDemand], ['reflowColumn', ReflowColumn], ['focusLocator', FocusLocator],
    ['persistentHover', PersistentHover], ['readingRuler', ReadingRuler], ['confirmActions', ConfirmActions],
    ['rememberSpot', ReadingSpot], ['expandAbbreviations', AbbreviationExpand], ['languageTag', LanguageTag],
    ['exploreChart', ExploreAChart], ['spaFocus', SpaFocus], ['skipLinks', SkipLinks],
    ['mathAccessible', MathA11y], ['keyboardNav', KeyboardNavigator], ['voiceCommands', VoiceCommands],
    ['agentWatch', AgentWatch],
  ];
  for (const [key, adapter] of flags) if (settings[key]) adapter.enable();

  if (settings.translatePage) TranslatePage.enable({ targetLang: settings.translateTo });
  if (settings.autoCaptions) AutoTranscriber.enable();
}

// ── AI content-fix pipeline (ported from the toolkit orchestrator) ──────────
async function runScan() {
  if (isRunning) return;
  isRunning = true;
  try {
    if (isEnabled('autoWcagFix')) {
      const violations = await runAxeAnalysis();
      await processViolations(violations);
    }
    await runAdditionalScans();
    await runTextProcessing();
  } catch (e) {
    console.warn('[AI4A11y] Scan failed:', e);
  } finally {
    isRunning = false;
  }
}

function scheduleScan() {
  if (scanTimer) clearTimeout(scanTimer);
  scanTimer = setTimeout(() => { scanTimer = null; runScan(); }, 250);
}

async function processViolations(violations) {
  const settings = getSettings();
  const imageTasks = [];
  for (const violation of violations) {
    for (const node of violation.nodes) {
      const el = getElementFromNode(node);
      if (!el || el.dataset.ai4a11yProcessed) continue;
      if (isImageViolation(violation.id) && isEnabled('autoDescribe')) {
        const handler = getAxeHandler(violation.id);
        if (handler) imageTasks.push(() => handler(el));
        continue;
      }
      try { await processViolation(violation, node, el, settings); }
      catch (e) { console.warn(`[AI4A11y] Failed to fix ${violation.id}:`, e); }
    }
  }
  const BATCH = 5;
  for (let i = 0; i < imageTasks.length; i += BATCH) {
    await Promise.all(imageTasks.slice(i, i + BATCH).map(fn => fn().catch(() => {})));
  }
}

function isImageViolation(ruleId) {
  return ['image-alt', 'input-image-alt', 'role-img-alt', 'svg-img-alt', 'object-alt', 'area-alt'].includes(ruleId);
}

async function processViolation(violation, node, el) {
  const handler = getAxeHandler(violation.id);
  if (!handler) return;
  if (violation.id.startsWith('color-contrast') && !isEnabled('fixContrast')) return;
  if (violation.id.includes('label') && !isEnabled('autoFixLabels')) return;
  if (violation.id.includes('caption') && !isEnabled('autoCaptions')) return;
  if (violation.id.startsWith('color-contrast')) {
    const style = getComputedStyle(el);
    await handler(el, style.color, style.backgroundColor);
  } else {
    await handler(el);
  }
}

async function runAdditionalScans() {
  if (isEnabled('autoDescribe')) {
    for (const img of findEmptyAltImages()) await generateImageAlt(img);
    for (const canvas of findCanvasElements()) await generateCanvasDescription(canvas);
  }
  if (isEnabled('autoVideoDescribe')) {
    const videos = Array.from(document.querySelectorAll('video'))
      .filter(v => !v.dataset.ai4a11yDescribed && !v.getAttribute('aria-label'));
    for (const video of videos) await generateVideoDescription(video).catch(() => {});
  }
  if (isEnabled('autoFixLabels')) {
    const ambiguousLinks = findAmbiguousLinks();
    if (ambiguousLinks.length) await improveAmbiguousLinks(ambiguousLinks);
    await fixAllTables();
  }
  if (isEnabled('autoWcagFix')) {
    fixLandmarks();
    document.querySelectorAll('a[target="_blank"]').forEach(link => {
      if (!(link.getAttribute('rel') || '').includes('noopener')) fixTargetBlank(link);
    });
    document.querySelectorAll('[tabindex]').forEach(el => {
      if (parseInt(el.getAttribute('tabindex')) > 0) fixPositiveTabindex(el);
    });
  }
}

async function runTextProcessing() {
  if (isEnabled('autoSimplify')) {
    for (const el of findComplexText()) await simplifyText(el);
  }
  if (isEnabled('autoSummarize')) {
    for (const el of findLongContent()) await summarizeContent(el);
  }
}

function findComplexText() {
  return Array.from(document.querySelectorAll('p, li, td, div')).filter(el => {
    if (el.dataset.ai4a11ySimplified || el.dataset.ai4a11yProcessed) return false;
    if (el.querySelector('p, div, article, section')) return false;
    return el.textContent.length > 300;
  });
}

function findLongContent() {
  return Array.from(document.querySelectorAll('p, article, section, .article-body')).filter(el => {
    if (el.dataset.ai4a11ySummarize || el.dataset.ai4a11yProcessed) return false;
    if (el.closest('[data-ai4a11y-summarize]')) return false;
    return el.textContent?.trim().length > 500;
  });
}

// ── Enable/disable/revert ───────────────────────────────────────────────────
let enabledTools = new Set();

function enableTool(toolName, options) {
  const tool = TOOL_MAP[toolName];
  if (!tool) return;
  try {
    if (options !== undefined) tool.enable(options); else tool.enable();
    enabledTools.add(toolName);
  } catch (e) { console.warn(`[AI4A11y] enable ${toolName} failed:`, e); }
}

function disableTool(toolName) {
  const tool = TOOL_MAP[toolName];
  if (!tool) return;
  try { tool.disable?.(); enabledTools.delete(toolName); }
  catch (e) { console.warn(`[AI4A11y] disable ${toolName} failed:`, e); }
}

function revertAll() {
  for (const adapter of ALL_ADAPTERS) { try { adapter.disable?.(); } catch (e) {} }
  try { ReadAloud.stop?.(); } catch (e) {}
  enabledTools.clear();

  // Undo the toolkit's DOM-rewriting AI fixers (simplify / contrast / labels).
  document.querySelectorAll('.ai4a11y-simplified').forEach(el => {
    const wrap = el.querySelector('.ai4a11y-original-content');
    if (wrap) {
      el.querySelector('.ai4a11y-text-content')?.remove();
      el.querySelector('.ai4a11y-toggle-original')?.remove();
      while (wrap.firstChild) el.appendChild(wrap.firstChild);
      wrap.remove();
    }
    el.classList.remove('ai4a11y-simplified');
  });
  document.querySelectorAll('.ai4a11y-contrast-fixed').forEach(el => {
    if (el.dataset.ai4a11yOriginalColor) el.style.color = el.dataset.ai4a11yOriginalColor;
    el.classList.remove('ai4a11y-contrast-fixed');
  });
  clearAllMarks();
  stats.wcag = stats.images = stats.labels = stats.text = stats.captions = 0;
  fixes.length = 0;
}

function getToolStates() {
  const states = {};
  for (const [name, adapter] of Object.entries(TOOL_MAP)) states[name] = !!adapter.enabled;
  return states;
}

// ── Apply a settings object (custom profile / Librarian prefs) ──────────────
function applyProfileSettings(settings) {
  updateSettings(settings);
  applyVisualSettings(getSettings());
  scheduleScan();
}

// ── Message protocol (unchanged surface for the popup / background) ─────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'enableTool') { enableTool(msg.tool, msg.options); sendResponse({ success: true }); }
  else if (msg.type === 'disableTool') { disableTool(msg.tool); sendResponse({ success: true }); }
  else if (msg.type === 'settingsChanged') {
    updateSettings(msg.settings || {});
    applyVisualSettings(getSettings());
    scheduleScan();
    sendResponse({ success: true });
  }
  else if (msg.type === 'revertAll') { revertAll(); sendResponse({ success: true }); }
  else if (msg.type === 'rescan') { revertAll(); init(); sendResponse({ success: true }); }
  else if (msg.type === 'setEnabled') {
    updateSettings({ enabled: msg.enabled });
    if (!msg.enabled) revertAll(); else init();
    sendResponse({ success: true });
  }
  else if (msg.type === 'getToolStates') { sendResponse({ states: getToolStates() }); }
  else if (msg.type === 'getStats') { sendResponse({ success: true, stats: { ...stats }, fixes: [...fixes] }); }
  else if (msg.type === 'speakPage') { ReadAloud.speakPage({ rate: msg.rate || 1 }); sendResponse({ success: true }); }
  else if (msg.type === 'stopSpeech') { ReadAloud.stop(); sendResponse({ success: true }); }
  else if (msg.type === 'applyProfile') { if (msg.settings) applyProfileSettings(msg.settings); sendResponse({ success: true }); }
  // No unconditional `return true`: every handled branch responds synchronously.
});

// ── Librarian / site-classification integration (app-specific) ──────────────
function sendMessageAsync(msg) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(msg, (resp) => {
      if (chrome.runtime.lastError) { resolve(null); return; }
      resolve(resp);
    });
  });
}

function detectPageContexts() {
  const contexts = [];
  try {
    if (document.querySelector('video, audio, iframe[src*="youtube.com"], iframe[src*="vimeo.com"], iframe[src*="player"]')) contexts.push('video');
    if (document.querySelectorAll('form input, form select, form textarea').length >= 3) contexts.push('form');
    if ((document.body?.innerText || '').length > 8000) contexts.push('document');
  } catch (_) {}
  return contexts;
}

async function init() {
  await loadAppSettings();
  if (getSettings().enabled === false) return;

  try {
    const profilesResp = await sendMessageAsync({ type: 'getCustomProfiles' });
    const profiles = profilesResp?.profiles || [];
    const autoApplyProfiles = profiles.filter(p => p.autoApply && p.siteTypes?.length > 0);
    const contexts = detectPageContexts();

    const meta = document.querySelector('meta[name="description"]');
    const classifyResp = await sendMessageAsync({
      type: 'classifySite', hostname: location.hostname, title: document.title,
      metaDescription: meta?.content || '',
    });

    let appliedProfile = false;
    if (autoApplyProfiles.length > 0 && classifyResp?.matchingProfile?.settings) {
      applyProfileSettings(classifyResp.matchingProfile.settings);
      appliedProfile = true;
      if (classifyResp.matchingProfile.actions?.length > 0) {
        chrome.runtime.sendMessage({
          type: 'runProfileActions', actions: classifyResp.matchingProfile.actions, sourceUrl: location.href,
        });
      }
    }
    if (!appliedProfile) {
      applyVisualSettings(getSettings());
      scheduleScan();
    }

    const prefs = await sendMessageAsync({
      type: 'librarianEffectivePreferences', url: location.href, contexts,
    });
    if (prefs?.settings && Object.keys(prefs.settings).length > 0) {
      const VA_KEYS = ['contrastMode', 'fontScale', 'lineHeight', 'letterSpacing',
        'dyslexiaFont', 'largeCursor', 'enhanceFocus', 'readingGuide'];
      let overlay = prefs.settings;
      if (VA_KEYS.some(k => overlay[k] !== undefined)) {
        const baseline = await chrome.storage.sync.get(VA_KEYS);
        overlay = { ...baseline, ...overlay };
      }
      applyProfileSettings(overlay);
    }
  } catch (e) {
    console.warn('[AI4A11y] Init failed, applying stored settings only:', e);
    applyVisualSettings(getSettings());
    scheduleScan();
  }
}

init();
