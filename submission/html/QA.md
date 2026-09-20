# HTML QA — 2026-09-19

Used the Playwright skill and local Chromium via playwright-cli; screenshots were opened and visually inspected using view_image.

## Passed

- Viewport widths 320, 390, 768, and 1440 px: document scroll width equals viewport width. Wide data tables scroll within their own regions.
- Final browser console: 0 errors and 0 warnings.
- All images loaded; all 66 expected audited policy endpoints rendered.
- No external resource requests. CSS, scripts, PNGs, SVGs and native MathML are local.
- All three JEPA seeds and 16 saved contexts are selectable; seed 2/context 16 displays correctly. Flow is explicitly unavailable in seed-1/2 diagnostic files.
- JEPA imagination plus model-seed-2 filter returns 9 endpoints.
- Keyboard End on the frame slider selects future step +8; requirement-selector buttons change explanation and pressed state.
- Desktop/mobile screenshot review: opening, real-frame comparison, objective diagram, Craftax tables, gate table, seed-separated plot, nuisance factorial.
- Print-media screenshot reviewed: interactive controls excluded; table colors retained and text readable. This is print-CSS verification, not a pagination audit of an exported PDF.
- An independent agent checked all 66 exact endpoints, group means, discoveries, Craftax values, exported hashes, local links and anchors against the supplied evidence. Its corrections to statistical scope, gray-noise conditions, held-out-head scope, contrast and stale-image failure handling were incorporated.

## Boundaries

The Playwright runner blocks file:// navigation. Direct-file browser loading was not verified by that runner; localhost loading and zero external dependencies were verified. Scientific inference remains limited as stated in the article, especially nonpaired real-v-imagined endpoints, different PPO protocols, off-budget probes, one flow fit and one Craftax fit.

## Representative screenshots

- output/playwright/desktop-top.png
- output/playwright/desktop-rollout.png
- output/playwright/desktop-tables.png
- output/playwright/desktop-craftax.png
- output/playwright/desktop-control.png
- output/playwright/desktop-nuisance.png
- output/playwright/mobile-top.png
- output/playwright/mobile-rollout.png
- output/playwright/mobile-objective.png
- output/playwright/mobile-nuisance.png
- output/playwright/mobile-equation.png
- output/playwright/print-table.png
