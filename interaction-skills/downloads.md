# Downloads

Document how to detect that a download actually started, because many sites only show a transient toast, spinner, or browser-level indicator. Also separate browser-triggered downloads from direct `http_get(...)` fetches, since static assets often do not need the browser at all.
