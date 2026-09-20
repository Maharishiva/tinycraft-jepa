# V2 HTML working state

This is a separate copy. V1 freeze manifest was verified unchanged after creating it.

Completed and integrated:
- Fresh Craftax384-episode data from research/craftax_table_v2.json, one trained checkpoint, 3 evaluation batches.
- Actual start/goal frame selector, metadata-only example selection, measured subgroup table and Wilson intervals.
- 3.5x latent-transition and6x candidate-sequence search cost of replanning, local six-action scope, replay as privileged reference.
- Prospective target-gradient method diagram; explicitly no outcome claim.
- Historical endpoint explorer collapsed by default.
- Browser screenshot QA at320/390/768/1440, no page overflow; measured subset and image selection checked.

Not final:
- Hero/title/abstract and historical V1 sections remain provisional. Replace them with at most2–3 completed findings after author/root fixes thesis.
- Await45-run matched MLP,27-run native-grid sensitivity, forecast readout matrices,12 target-gradient fits/readouts, and PVE history-teacher result.
- Use output/revision/matched-analysis/publication.json and output/revision/native-grid-analysis/publication.json only when complete, with all denominators and per-model outcomes preserved.
- Native-grid is a separate policy-interface sensitivity, not more MLP replications.
- Main target-gradient endpoint is mean absolute error per coordinate in pixels. It is NOT numerically identical to V1 cube coordinate-summed L1 error.
- Forecast probes require real→real, real→forecast, forecast→forecast, forecast→real plus copied-current baseline. A recovered forecast readout changes the erasure narrative.
- Paper link already points to../v2/main.pdf.

Assets are all local in assets/. New shared figures were copied from shared-v2. No frozen V1 assets are referenced for mutable content.

New external relative source link: ../../artifacts/revision/craftax_goal_v2.json. Other preserved source links point to../../research and../v2/main.pdf.

## Analytical illustration completed

- Optional, collapsed by default under theory; native SVG/CSS/JS files analytical-illustration.{html,css,js}.
- Theory reviewer verified one-lag prediction and single-snapshot readout formulas. Across-episode variance1 and within-episode variance alpha are distinguished.
- Alpha/rho bothdefault0.5. Source attribution is the knownSobal2022shortcut; no newtheorem or controlclaim.
- Keyboard limits checked: alpha0 MSE0/R20; alpha1rho.5 MSE.75/R21; alpha1rho.99 MSE.0199/R21; alpha1rho0 MSE1/R21.
- No-JavaScript context verified: interactive sliders hidden, exact default staticSVG and equations remain.
- Screenshots visually checked: output/playwright/gaussian-{desktop,mobile,static}.png. Nooverflow at320,390,768,1440.
