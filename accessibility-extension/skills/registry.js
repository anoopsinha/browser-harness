// Re-export shim — the canonical registry lives in the toolkit so every host
// (this example extension, the toolkit server, XR, mobile) shares ONE settings
// vocabulary. Do not add tool entries here; edit toolkit/registry/tools.js.
// Same pattern the toolkit itself uses for its adapter shims.
export * from 'ai-for-accessibility-toolkit/toolkit/registry/tools.js';
