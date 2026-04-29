# Quant UI Review

## What is making the current UI feel unfinished

- `src/index.css` is still carrying the Vite starter shell. The fixed `#root` width, centered layout, and `border-inline` are what create the large empty gutters and vertical divider lines in the screenshot.
- `src/App.tsx` is acting like a quick flexbox wrapper instead of an application shell. There is no research header, no shared status state, and no top-level ownership of the simulation run.
- `src/components/Sidebar.tsx` exposes only two unlabeled sliders with no visible values, no presets, and no context on what the numbers mean.
- `src/components/ChartPanel.tsx` is combining fetching, transformation, and rendering in one place. It also re-calls the backend when the user only changes the selected asset because `selectedAsset` is part of the request effect dependency list.
- `src/components/Insights.tsx` is still a placeholder, so the right panel weakens credibility instead of adding analytical value.
- `backend/main.py` ignores `horizon`, hardcodes only 10 condition windows, and does not return metadata like `start_prices` or `path_count`, which makes the frontend rely on fragile local assumptions.

## Files currently lacking

- `src/types/simulation.ts`
  This should hold the API contract and shared types so the UI stops relying on `any`.
- `src/lib/analytics.ts`
  This should convert latent returns into price paths, compute quantiles, and generate the metrics shown in the chart and insights panel.

## Proposed rewritten files

- [App.tsx](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/proposed/finance-ui/src/App.tsx)
- [App.css](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/proposed/finance-ui/src/App.css)
- [index.css](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/proposed/finance-ui/src/index.css)
- [Sidebar.tsx](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/proposed/finance-ui/src/components/Sidebar.tsx)
- [ChartPanel.tsx](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/proposed/finance-ui/src/components/ChartPanel.tsx)
- [Insights.tsx](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/proposed/finance-ui/src/components/Insights.tsx)
- [analytics.ts](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/proposed/finance-ui/src/lib/analytics.ts)
- [simulation.ts](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/proposed/finance-ui/src/types/simulation.ts)
- [main.py](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/proposed/backend/main.py)
