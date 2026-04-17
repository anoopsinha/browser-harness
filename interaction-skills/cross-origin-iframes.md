# Cross-Origin Iframes

Cross-origin iframes are the case where CDP target handling matters most, because DOM access from the top page is blocked and the iframe may appear as its own target. This note should focus on `iframe_target(...)`, target attachment, and when compositor-level coordinate clicks are still the lower-friction option.
