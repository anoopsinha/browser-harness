# Iframes

Same-origin iframes usually need DOM traversal through `contentDocument` or `contentWindow`, while page-level coordinate clicks still land at the compositor layer. Keep the coordinate-system warning front and center: iframe element rects are local to the frame and must be offset into page coordinates before clicking.
