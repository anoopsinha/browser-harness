# Shadow DOM

Normal `document.querySelector(...)` stops at shadow boundaries, so this note should focus on walking `shadowRoot` recursively instead of assuming one global selector works. Also call out when coordinate clicking is simpler than DOM traversal, especially for deeply nested component libraries.
