# HTML research companion

Open `index.html` with its neighboring CSS, JS, and `assets/` intact. The article makes no external resource requests; links to primary papers are optional references. Serve the worktree root with a local HTTP server to also resolve manuscript and research-report links.

- `index.html`: English V1 research article, audited pre-existing evidence only.
- `styles.css`: editorial theme, responsive tables, print styles and reduced-motion support.
- `article.js`: requirement selector, actual saved-frame scrubber and 66-endpoint filters.
- `assets/rollout-*.png`: direct nearest-neighbor sprite exports, 8 future steps by 16 saved contexts. No image generation, frame recoloring or altered model predictions.
- `assets/rollout-manifest.json`: original NPZ paths and SHA-256 hashes.
- `assets/evidence.json`: exact final endpoints, source hashes and per-model aggregates.
- `assets/control_by_seed.svg`: shared scientific plot; provenance in `figure-captions.json`.
- `build_assets.py`: regenerate local data exports from the worktree's archived runs and audited ledger (requires NumPy/Pillow).
- `output/playwright/`: screenshots inspected during browser QA.

Theme: warm paper, navy text, rust JEPA, teal real transitions, violet flow. Table color denotes a method family; it does not encode statistical significance. The nuisance table shade marks its larger measured error.

The official-format PDF is the submission artifact. This HTML is a local research companion, not a deployed website.
