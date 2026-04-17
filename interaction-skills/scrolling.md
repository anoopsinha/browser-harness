# Scrolling

Scrolling needs separate recipes for page scroll, nested scroll containers, virtualized lists, and dropdown menus with their own internal scroll regions. The main rule is to identify which element is actually consuming wheel events before you scroll, especially on pages with several independent containers visible at once.
