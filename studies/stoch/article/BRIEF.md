# Blog article brief (Russian, for the personal blog; companion to the ICLR 2027 paper)

Working title (RU): «Научиться игнорировать важное: JEPA в стохастических мирах».
Output: `studies/stoch/article/index.html` (single self-contained page, its own CSS, figures referenced from
`studies/stoch/article/assets/`), plus `studies/stoch/article/index.md` (the same text in Markdown for reuse).
Language: Russian, first person plural is fine ("мы"), tone = engineering blog: direct, concrete, no hype, the
evidence is control (survival), not probes or pretty videos. Length ~2500-3500 words. Code snippets minimal.

Audience: ML engineers / researchers who follow world-model discourse (LeCun's JEPA vs generative world models
such as Dreamer / diffusion-forcing). The article is the "killing case" version of the paper: one fair,
controlled, adversarial environment that is easy to present, and an allegory for real robotics control
(perceptual noise, nuisance appearance, control noise, a safety margin, and an event you must wait to observe).

Structure:
1. Hook: the question (does a JEPA + planner work as a world model when the future is a distribution?), the
   two hyped camps (LeCun's JEPA + MPC vs Dreamer-style generative WM + PPO in imagination), what we did in one
   night on two GPUs with 10-minute models.
2. The worlds, in one figure (fig_worlds.png): 6 cube worlds + the gate; the gate rules and why it is
   adversarial but fair (the flow model gets the same data, the same budget; random play dies at the first wall).
3. The two mechanisms, stated plainly, with the theory figure (fig_theory.png): representation (texture trap /
   noise basin, from our earlier post) and planning (the mean is not a state of the world; the one-step
   witness; open-loop plans cannot value waiting).
4. The three models and the controllers (a compact table), the fixed reward-head bug in the flow recipe.
5. The half-open wall figure (fig_predictions.png): same context, the two real futures, what each JEPA latent can
   hold vs what its mean rollout predicts, three flow samples. Describe exactly what is visible: the class-token
   LeWM loses the cube and predicts a faint half-intensity wall; the grid JEPA keeps the cube but its rollout
   drops the wall entirely after one step even though its encoder holds it; the flow samples show the wall
   scrolling, the gap on top in two samples and at the bottom in one, and the collision.
6. Results: the gate (fig_gate_results.png + table), the six cube worlds (fig_cube_results.png), Craftax
   relevance (grid JEPA on Craftax: goal reaching and reward MPC numbers). Report where JEPA is fine.
7. What this means for robotics (the allegory), what we did not test (one seed, 10-minute budgets, the LeWM
   shrunk), and the pointer to the paper (anonymous for now) and the code.

Numbers come from runs/stoch/summary.json (survival %, lengths, walls passed) and the markdown table the
summarize script prints; do not invent any number: if a cell is missing, say it is still running.
