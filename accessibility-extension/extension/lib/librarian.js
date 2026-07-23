// Librarian — the personal memory/profile agent. Sole writer of the
// Librarian-owned stores (mine.profile, mine.suppressions, mine.siteIndex,
// mine.views, memory shards). Everything else — popup, content script,
// onboarding — goes through the message handlers in background.js or calls
// globalThis.Librarian directly from the same service worker. Never write
// these stores elsewhere.
//
// Fast lane only — deterministic, no LLM, milliseconds: profile reads, cached
// site classification, scope-chain preference merge, scored recall, explicit
// user edits. Gemini is still injected (setGeminiCaller) for the two
// deterministic-boundary uses that need it: getSiteCategory's one-time
// classification fallback for unknown hosts and interpretNeedsPrompt's caller.
//
// Classic script (assigns globalThis.Librarian); loaded by background.js
// via importScripts after datastore.js. Gemini access is injected via
// setGeminiCaller from the background service worker.

(() => {
  const DS = () => globalThis.Datastore;
  const TAX = () => globalThis.AA_TAXONOMY;

  // ---- LLM wiring -----------------------------------------------------------
  let _gemini = null; // async (prompt) => string

  // ---- helpers --------------------------------------------------------------
  function newId(prefix) {
    return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  }

  function originOf(url) {
    try { return new URL(url).hostname.toLowerCase().replace(/^www\./, ''); }
    catch { return null; }
  }

  // Decay half-lives per class (ms). Stable facts effectively don't decay.
  const DECAY_HALF_LIFE = {
    stable: Infinity,
    slow: 1000 * 60 * 60 * 24 * 90,  // ~90 days
    fast: 1000 * 60 * 60 * 24 * 7,   // ~7 days
  };

  // Retrieval score: recency x importance x confidence. Deterministic, no
  // embeddings — scope sharding already did the relevance cut (we only load
  // the shards the current page belongs to).
  function scoreRecord(r, now) {
    const half = DECAY_HALF_LIFE[r.decayClass] || DECAY_HALF_LIFE.slow;
    const age = now - (r.lastAccessed || r.updatedAt || r.createdAt || now);
    const recency = half === Infinity ? 1 : Math.pow(0.5, age / half);
    return recency * ((r.importance || 5) / 10) * (r.confidence ?? 0.7);
  }

  function conditionsMet(r, now) {
    if (!r.conditions) return true;
    const d = new Date(now);
    if (r.conditions.timeOfDay) {
      const h = d.getHours();
      const { fromHour = 0, toHour = 24 } = r.conditions.timeOfDay;
      const inWindow = fromHour <= toHour
        ? (h >= fromHour && h < toHour)
        : (h >= fromHour || h < toHour); // overnight window
      if (!inWindow) return false;
    }
    if (Array.isArray(r.conditions.daysOfWeek) && r.conditions.daysOfWeek.length) {
      if (!r.conditions.daysOfWeek.includes(d.getDay())) return false;
    }
    return true;
  }

  const VALID_SCOPE = /^(general|category:[a-z-]+|context:[a-z-]+|origin:[a-z0-9.-]+|tool:[a-zA-Z0-9_-]+)$/;

  // Coerce a settings object into the canonical units/ranges declared in the
  // registry's settingsMeta. Guards against LLM-written values in the wrong
  // unit — e.g. an extracted memory with `fontScale: 1.5` (a multiplier) when
  // the pipeline expects a percentage (`150`); applied raw, 1.5 / 100 collapses
  // the font. A value far below its range whose ×100 lands in range is treated
  // as a multiplier; everything is then clamped to range.
  function sanitizeSettings(settings) {
    if (!settings || typeof settings !== 'object') return settings;
    let meta = {};
    try { meta = DS().global.tools().settingsMeta || {}; } catch (_) {}
    const out = {};
    for (const [k, v] of Object.entries(settings)) {
      const m = meta[k];
      if (m && m.type === 'number' && Array.isArray(m.range) && typeof v === 'number') {
        const [min, max] = m.range;
        let val = v;
        if (val < min && val * 100 >= min && val * 100 <= max) val = val * 100;
        out[k] = Math.min(max, Math.max(min, val));
      } else {
        out[k] = v;
      }
    }
    return out;
  }

  function normalizeRecord(raw, now) {
    const r = { ...raw };
    r.id = r.id || newId('mem');
    r.text = String(r.text || '').slice(0, 500);
    r.tier = ['profile', 'preference', 'site', 'task'].includes(r.tier) ? r.tier : 'preference';
    r.scope = VALID_SCOPE.test(r.scope || '') ? r.scope : 'general';
    r.kind = ['preference', 'procedural', 'suppression', 'rule', 'observation'].includes(r.kind) ? r.kind : 'preference';
    r.importance = Math.min(10, Math.max(1, Number(r.importance) || 5));
    r.confidence = Math.min(1, Math.max(0, Number(r.confidence ?? 0.7)));
    r.decayClass = ['stable', 'slow', 'fast'].includes(r.decayClass) ? r.decayClass : 'slow';
    r.conditions = r.conditions || null;
    r.settings = (r.settings && typeof r.settings === 'object') ? sanitizeSettings(r.settings) : null;
    r.aspect = r.aspect || null;
    r.occurrenceCount = Math.max(1, Number(r.occurrenceCount) || 1);
    r.firstSeenAt = r.firstSeenAt || now;
    r.createdAt = r.createdAt || now;
    r.updatedAt = now;
    r.lastAccessed = r.lastAccessed || now;
    r.status = ['active', 'superseded', 'expired'].includes(r.status) ? r.status : 'active';
    r.supersededBy = r.supersededBy || null;
    r.source = r.source || 'inferred';
    return r;
  }

  // Scopes relevant to a page, least → most specific (merge order).
  function scopesFor(url, contexts) {
    const scopes = ['general'];
    for (const c of contexts || []) {
      if (TAX().contexts.some(x => x.id === c)) scopes.push(`context:${c}`);
    }
    const origin = originOf(url);
    return { scopes, origin };
  }

  async function loadScopeShards(url, contexts) {
    const { scopes, origin } = scopesFor(url, contexts);
    let category = null;
    if (origin) {
      category = await Librarian.getSiteCategory(origin); // cached/deterministic only
      if (category) scopes.splice(scopes.length, 0, `category:${category}`);
      scopes.push(`origin:${origin}`);
    }
    const shards = {};
    for (const s of scopes) shards[s] = await DS().getMemoryShard(s);
    return { scopes, shards, origin, category };
  }

  // ---- profile ---------------------------------------------------------------
  const PROFILE_DEFAULTS = {
    schemaVersion: 1,
    supportAreas: [],
    freeText: '',
    fields: {},          // canonical ability fields, e.g. { vision: { fontScale: 130 } }
    metaPreferences: {
      consentBoundary: 'profile-only',  // 'profile-only' | 'all-tiers'
      language: 'standard',             // 'standard' | 'plain'
      maxProposalsPerWeek: 30,
    },
    memoryPaused: false,
  };

  async function getOrInitProfile() {
    let p = await DS().get('mine.profile');
    if (!p) {
      // Seed from the legacy onboarding profile if present (it was written
      // once by onboarding and never read — give it a life).
      const legacy = await DS().get('mine.onboardingProfile');
      p = structuredClone(PROFILE_DEFAULTS);
      if (legacy) {
        p.supportAreas = legacy.supportAreas || [];
        p.freeText = legacy.freeText || '';
      }
      p.createdAt = Date.now();
      p.updatedAt = Date.now();
      await DS().set('mine.profile', p);
    }
    return p;
  }

  // ---- public surface ----------------------------------------------------------
  const Librarian = {
    setGeminiCaller(fn) { _gemini = fn; },

    // ====================== FAST LANE (no LLM) ======================

    async getProfile() {
      return await getOrInitProfile();
    },

    // User-initiated edit — bypasses the proposal gate by design (the gate
    // exists for *inferred* changes; explicit user intent needs no consent).
    async setProfileField(path, value) {
      return await DS().patch('mine.profile', async (p) => {
        p = p || structuredClone(PROFILE_DEFAULTS);
        const parts = String(path).split('.');
        let obj = p;
        for (let i = 0; i < parts.length - 1; i++) {
          if (typeof obj[parts[i]] !== 'object' || obj[parts[i]] == null) obj[parts[i]] = {};
          obj = obj[parts[i]];
        }
        obj[parts[parts.length - 1]] = value;
        p.updatedAt = Date.now();
        return p;
      });
    },

    // Fast lane for manual setting flips (popup toggle, onboarding choice).
    // A deliberate change is the strongest preference signal there is, so it
    // is recorded immediately as a durable user-explicit record that gets
    // FINAL say in getEffectivePreferences — without this, an auto-apply
    // profile or a learned record re-imposes the old value on the next page
    // and the user's change silently "doesn't stick". One record per setting
    // key, updated in place on subsequent changes. Recorded even while
    // memory is paused: this is a direct user command, not an inference.
    async recordExplicitSetting(key, value, origin) {
      return this.recordScopedSettings('general', { [key]: value }, { origin });
    },

    // Generalized explicit-setting writer: upserts one durable user-explicit
    // record PER setting key at the given scope (general | category:<id> |
    // origin:<host> | context:<id>). These get final say in
    // getEffectivePreferences, but a scoped record only loads when the page
    // matches that scope — so "make news sites easier to read" lands on
    // category:news and does NOT leak to every site. scopeLabel is a
    // human phrase for the record text. Returns the record ids.
    async recordScopedSettings(scope, settings, opts = {}) {
      const now = Date.now();
      scope = VALID_SCOPE.test(scope || '') ? scope : 'general';
      const where = opts.scopeLabel || (
        scope === 'general' ? '' :
        scope.startsWith('category:') ? ` on ${scope.slice(9)} sites` :
        scope.startsWith('origin:') ? ` on ${scope.slice(7)}` :
        scope.startsWith('context:') ? ` for ${scope.slice(8)} content` : '');
      const shard = await DS().getMemoryShard(scope);
      const ids = [];
      for (const [key, value] of Object.entries(settings || {})) {
        const aspect = `setting.${key}`;
        const text = `You set ${key} to ${JSON.stringify(value)}${where}.`;
        let rec = shard.find(r => r.source === 'user-explicit' && r.aspect === aspect && r.status === 'active');
        if (rec) {
          rec.settings = { [key]: value };
          rec.text = text;
          rec.occurrenceCount = (rec.occurrenceCount || 1) + 1;
          rec.updatedAt = now;
          rec.lastAccessed = now;
        } else {
          rec = normalizeRecord({
            kind: 'preference', tier: 'preference', scope, aspect,
            source: 'user-explicit', confidence: 1, importance: 8,
            decayClass: 'stable', settings: { [key]: value }, text,
          }, now);
          shard.push(rec);
        }
        ids.push(rec.id);
      }
      await DS().setMemoryShard(scope, shard);
      return ids;
    },

    // Classify once, cache forever; user override wins and is sticky.
    // Deterministic by default — pass {allowLlm: true, title} to let the
    // background's classify handler fall through to Gemini for unknown hosts.
    async getSiteCategory(origin, opts = {}) {
      origin = (origin || '').toLowerCase().replace(/^www\./, '');
      if (!origin) return null;
      const idx = await DS().get('mine.siteIndex');
      const hit = idx[origin];
      if (hit && (hit.source === 'user' || hit.taxonomyVersion === TAX().version)) {
        return hit.category;
      }
      let category = TAX().categoryForHost(origin);
      let source = 'hostmap';
      if (!category && opts.allowLlm && _gemini) {
        try {
          const valid = TAX().categoryIds();
          const out = await _gemini(
            `Classify this website into exactly one category. Hostname: "${origin}", Title: "${opts.title || ''}". Categories: ${valid.join(', ')}. Return ONLY the category word, nothing else.`
          );
          const cleaned = (out || '').trim().toLowerCase();
          category = valid.includes(cleaned) ? cleaned : 'other';
          source = 'llm';
        } catch { category = null; }
      }
      if (category) {
        await DS().patch('mine.siteIndex', (cur) => {
          cur[origin] = { category, source, classifiedAt: Date.now(), taxonomyVersion: TAX().version, ...(cur[origin]?.paused ? { paused: true } : {}) };
          return cur;
        });
      }
      return category;
    },

    async setSiteCategoryOverride(origin, category) {
      origin = (origin || '').toLowerCase().replace(/^www\./, '');
      await DS().patch('mine.siteIndex', (cur) => {
        cur[origin] = { ...(cur[origin] || {}), category, source: 'user', classifiedAt: Date.now(), taxonomyVersion: TAX().version };
        return cur;
      });
    },

    // Deterministic scope-chain merge of machine-actionable settings.
    // Order (later wins): general → context → category → explicit
    // customProfile (user-authored beats inferred at category level) →
    // origin. Rule records (kind 'rule') in a shard apply after that
    // shard's preferences. Conditions (time windows) filter throughout.
    async getEffectivePreferences(url, contexts = []) {
      const now = Date.now();
      const { scopes, shards, origin, category } = await loadScopeShards(url, contexts);
      const merged = {};
      const applied = [];
      // provenance: key -> scope of the record that set its final value, so a
      // consumer (the popup) can write a change back to the same scope rather
      // than clobbering the global baseline.
      const provenance = {};
      const assign = (src, scope) => {
        const clean = sanitizeSettings(src) || {};
        Object.assign(merged, clean);
        for (const k of Object.keys(clean)) provenance[k] = scope;
      };
      // Manual user choices (recordExplicitSetting) are deferred and applied
      // after everything else: a deliberate toggle must beat profiles and
      // learned records at any scope, or the user's change reverts on the
      // next page load.
      const explicit = [];
      // sanitizeSettings defensively here too: records written before the
      // unit-coercion fix may still hold a multiplier (fontScale 1.5), and we
      // must not collapse the font on read.
      const applyShard = (scope) => {
        const recs = (shards[scope] || [])
          .filter(r => r.status === 'active' && r.settings && conditionsMet(r, now))
          .sort((a, b) => (a.kind === 'rule') - (b.kind === 'rule')); // rules last
        for (const r of recs) {
          if (r.source === 'user-explicit') { explicit.push({ r, scope }); continue; }
          assign(r.settings, scope);
          applied.push({ id: r.id, scope, text: r.text });
        }
      };
      for (const s of scopes) {
        if (s.startsWith('origin:')) continue; // origin applies last, below
        applyShard(s);
        // Explicit user profiles slot in right after their category.
        if (s.startsWith('category:') && category) {
          const profiles = (await DS().get('mine.profiles')) || [];
          const match = profiles.find(p => p.autoApply && p.siteTypes?.includes(category));
          if (match?.settings) {
            assign(match.settings, s);
            applied.push({ id: match.id, scope: s, text: `Profile "${match.name}"`, explicit: true });
          }
        }
      }
      if (origin) applyShard(`origin:${origin}`);
      // Among explicit records, the most SPECIFIC scope wins (origin > category
      // > context > general); ties broken by recency. Otherwise a newer global
      // toggle would override a site-scoped choice on its own site.
      const specificity = (sc) => sc.startsWith('origin:') ? 3 : sc.startsWith('category:') ? 2 : sc.startsWith('context:') ? 1 : 0;
      explicit.sort((a, b) => (specificity(a.scope) - specificity(b.scope))
        || ((a.r.updatedAt || 0) - (b.r.updatedAt || 0)));
      for (const { r, scope } of explicit) {
        assign(r.settings, scope);
        applied.push({ id: r.id, scope, text: r.text, explicit: true });
      }
      return { settings: merged, applied, provenance, category, origin };
    },

    // Context block for agent prompts: core memory block + scored facts for
    // this page + category playbook. Deterministic; markdown at the
    // boundary, records at rest.
    async recall(url, task = '', contexts = []) {
      const now = Date.now();
      const { scopes, shards, origin, category } = await loadScopeShards(url, contexts);
      const profile = await getOrInitProfile();
      const views = await DS().get('mine.views');

      const facts = [];
      for (const s of scopes) {
        for (const r of (shards[s] || [])) {
          if (r.status !== 'active' || r.kind === 'suppression' || !conditionsMet(r, now)) continue;
          facts.push({ ...r, _scope: s, _score: scoreRecord(r, now) });
        }
      }
      facts.sort((a, b) => b._score - a._score);
      const top = facts.slice(0, 12);

      // Touch lastAccessed on what we surfaced (recency feedback loop).
      const touched = new Set(top.map(r => r.id));
      for (const s of new Set(top.map(r => r._scope))) {
        const shard = shards[s].map(r => touched.has(r.id) ? { ...r, lastAccessed: now } : r);
        await DS().setMemoryShard(s, shard);
      }

      const lines = [];
      const core = views.coreBlock
        || `Support areas: ${profile.supportAreas.join(', ') || 'not specified'}.`
        + (profile.freeText ? ` Notes: ${profile.freeText}` : '');
      lines.push('### About this user', core);
      const byScope = (pred, title) => {
        const hits = top.filter(pred);
        if (hits.length) {
          lines.push(`### ${title}`);
          for (const f of hits) lines.push(`- ${f.text}`);
        }
      };
      byScope(f => f._scope === 'general', 'General preferences');
      byScope(f => f._scope.startsWith('context:'), 'For this kind of content');
      byScope(f => f._scope.startsWith('category:'), category ? `On ${category} sites` : 'On sites like this');
      byScope(f => f._scope.startsWith('origin:'), origin ? `On ${origin}` : 'On this site');
      const playbook = category && views.playbooks && views.playbooks[category];
      if (playbook) lines.push(`### Playbook: ${category} sites`, playbook);

      return { block: lines.join('\n'), facts: top, profile, category, origin };
    },

    async listMemories(filter = {}) {
      const out = [];
      const meta = await chrome.storage.local.get(null);
      for (const [key, recs] of Object.entries(meta)) {
        if (!key.startsWith('aa.mine.memory.')) continue;
        const scope = key.slice('aa.mine.memory.'.length);
        for (const r of (recs || [])) {
          if (filter.status && r.status !== filter.status) continue;
          if (filter.scope && scope !== filter.scope) continue;
          out.push({ ...r, scope });
        }
      }
      const supp = await DS().get('mine.suppressions');
      return { memories: out, suppressions: supp };
    },

    async deleteMemory(id) {
      const all = await chrome.storage.local.get(null);
      for (const [key, recs] of Object.entries(all)) {
        if (!key.startsWith('aa.mine.memory.')) continue;
        const idx = (recs || []).findIndex(r => r.id === id);
        if (idx >= 0) {
          recs.splice(idx, 1);
          await chrome.storage.local.set({ [key]: recs });
          return true;
        }
      }
      // Suppressions are deletable too (un-suppress).
      const removed = await DS().patch('mine.suppressions', (s) =>
        s.filter(x => x.id !== id));
      return Array.isArray(removed);
    },

    // Prompt for the popup's "what support do you need?" flow. The Librarian
    // owns it so the "does this exist in the global db?" decision is grounded
    // in the actual tools registry (Datastore.global.tools) and conditioned
    // on the ability profile — not a hand-maintained vocabulary copy.
    // Fast lane: builds a string, never calls the LLM itself.
    async interpretNeedsPrompt(text) {
      const tools = DS().global.tools();
      const profile = await getOrInitProfile();
      const profileBlock = (profile.supportAreas.length || profile.freeText)
        ? `\n\nWhat we know about this user:\n- Support areas: ${profile.supportAreas.join(', ') || 'unspecified'}`
          + (profile.freeText ? `\n- In their words: "${profile.freeText}"` : '')
        : '';
      return `You are an accessibility assistant for a browser extension. The user describes what they need in plain language. Map their description to specific extension settings.

Available settings (use these exact keys):
${tools.settingsVocabularyLines().join('\n')}

Built-in tools these settings belong to (for context on what already exists):
${tools.forPrompt().map(t => `- ${t.name}: ${t.description}`).join('\n')}${profileBlock}

Site categories (for scoping): ${TAX().categoryIds().join(', ')}.

User says: "${text}"

Return ONLY valid JSON with:
{
  "summary": "One friendly sentence describing what you understood",
  "scope": "Where these settings should apply. Use 'general' for everywhere (the default). If the user limits it to a kind of site, use 'category:<id>' with one of the categories above (e.g. 'on news sites' -> 'category:news', 'when watching videos' -> 'category:video'). If they name a specific website, use 'origin:<hostname>' (e.g. 'on youtube.com'). Only narrow the scope when the user explicitly limits it.",
  "settings": { /* only keys that should change, with their values */ },
  "reasons": { /* same keys as settings, each with a short reason why */ },
  "newSkills": [ /* ONLY if the user's need CANNOT be fully met by the settings and built-in tools above, suggest custom skills to build. Each object has "name" (short) and "description" (1-2 sentences of what it would do). Leave as empty array [] if existing settings are sufficient. */ ]
}`;
    },

    // ================== SKILLS (the Engineer + Skills db) ==================
    // Adaptive-agent layer from the toolkit: SKILL.md playbooks that compose
    // adapters. Parse/validate/resolve/match + the Engineer's buildSkill come
    // from AA_SKILL_CORE (bundled out of the ai-for-accessibility-toolkit
    // package by build.js); this section is the Librarian's storage + consent
    // side. Nothing is saved without the user validating first.

    // All skills available to this person: built-in (global tier, from
    // skill-docs.js) + their own (mine.skillDocs).
    async listSkills() {
      const builtin = (DS().global.skills() || []).map(s => ({ ...s, source: 'builtin' }));
      const mine = (await DS().get('mine.skillDocs') || []).map(s => ({ ...s, source: 'mine' }));
      return [...builtin, ...mine];
    },

    // Best-fitting skill for a page + this person. Deterministic scoring over
    // the profile's support areas and the page category — no LLM.
    async retrieveSkill(url, contexts = []) {
      const profile = await getOrInitProfile();
      const origin = originOf(url);
      const category = origin ? await this.getSiteCategory(origin) : null;
      const ctx = { supportAreas: profile.supportAreas || [], category };
      const scored = (await this.listSkills())
        .map(s => ({ skill: s, score: globalThis.AA_SKILL_CORE.matchSkill(s, ctx) }))
        .filter(x => x.score > 0)
        .sort((a, b) => b.score - a.score);
      return scored.length ? scored[0].skill : null;
    },

    // "Does the skill exist in the db?" — checked BEFORE the Engineer builds
    // anything. Deterministic keyword match, so the reuse offer works without
    // an API key. Returns the best fit or null.
    async findSkillForNeed(need) {
      const scored = (await this.listSkills())
        .map(s => ({ skill: s, score: globalThis.AA_SKILL_CORE.matchSkillToNeed(s, need) }))
        .filter(x => x.score >= 4)
        .sort((a, b) => b.score - a.score);
      return scored.length ? scored[0].skill : null;
    },

    // Compile a skill to the deterministic apply-plan (settings + adapter ids
    // + agent actions). No LLM at apply-time.
    resolveSkill(skill) {
      return globalThis.AA_SKILL_CORE.resolveSkill(skill);
    },

    // The Engineer: build a new skill from a plain-language need, grounded in
    // the real adapter catalog. Does NOT save — the user validates first (the
    // adaptive evaluation interface). On rejection, pass the attempt back as
    // { previous, feedback } and the Engineer revises it.
    async buildSkill(need, opts = {}) {
      const profile = await getOrInitProfile();
      return await globalThis.AA_SKILL_CORE.buildSkill(need, {
        llm: _gemini,
        tools: DS().global.tools(),
        taxonomy: TAX(),
        profile,
        previous: opts.previous || null,
        feedback: opts.feedback || '',
      });
    },

    // Persist a user-validated skill to their Skills db (mine.skillDocs).
    // Re-validates against the registry so a malformed skill can't be stored.
    // (The full toolkit also logs the save as a profile observation; the
    // Librarian-lite has no extraction pipeline to fold it, so we don't.)
    async saveSkill(skill) {
      const { valid, errors } = globalThis.AA_SKILL_CORE.validateSkill(skill, { tools: DS().global.tools() });
      if (!valid) return { saved: false, errors };
      await DS().patch('mine.skillDocs', (skills) => {
        const idx = skills.findIndex(s => s.name === skill.name);
        const entry = { ...skill, savedAt: Date.now() };
        if (idx >= 0) skills[idx] = entry; else skills.push(entry);
        return skills;
      });
      return { saved: true, errors: [] };
    },

    async deleteSkill(name) {
      let removed = false;
      await DS().patch('mine.skillDocs', (skills) => {
        const next = skills.filter(s => s.name !== name);
        removed = next.length !== skills.length;
        return next;
      });
      return removed;
    },

    async setMemoryPaused(paused) {
      await DS().patch('mine.profile', (p) => {
        p = p || structuredClone(PROFILE_DEFAULTS);
        p.memoryPaused = !!paused;
        p.updatedAt = Date.now();
        return p;
      });
    },

    async setOriginPaused(origin, paused) {
      origin = (origin || '').toLowerCase().replace(/^www\./, '');
      await DS().patch('mine.siteIndex', (cur) => {
        cur[origin] = { ...(cur[origin] || {}), paused: !!paused };
        return cur;
      });
    },

  };

  globalThis.Librarian = Librarian;
})();
