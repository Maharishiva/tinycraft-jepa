# Four ICLR 2026 oral papers: full-text reading notes

These are four relevant, officially selected ICLR 2026 oral papers, not an objective ranking of the year's four best papers. Reading was selective and bounded as recorded below; it was not independent reproduction or proof verification. Source-derived summaries are kept short; recommendations are our inferences.

## World-in-World: World Models in a Closed-Loop World

**Claim and evidence.** The paper argues that visual fidelity is an inadequate proxy for embodied utility. It contributes a proposal/simulation/revision evaluation interface, action adapters, four task families, and post-training/data/inference-compute ablations. The accepted paper additionally tests revision policies and transfer across scene distributions. Tables expose the base policy, control interface, input modality and parameter count. Its strongest writing move is to repeat one precise question across motivation, measurement and analysis: does the simulated future improve the executed decision? The argument proceeds from task outcomes to diagnostic scatterplots and then controlled adaptation studies.

**Lesson for Tinycraft (inference).** Lead with the gap between a predictive representation and a decision-useful representation. Connect every diagnostic to actual control outcomes. Use consistent colors for model families, separate base-policy and world-model contributions, and put the evaluation contract beside the headline plot. Figure 5's paired scatterplots make the surrogate-versus-outcome argument immediately visible; pale colored table headers group tasks without overwhelming numerical comparisons.

**Scope.** Its controllability proxy is 1-LPIPS, not a causal intervention metric. Manipulation gains depend heavily on the proposal policy, and long-horizon/generalization limitations remain. Do not inherit universal claims from its benchmark framing.

**Reading ledger.** HTML main text Sections 1–5, Tables 1–4 and figure captions 1–7; PDF pages 3 and 9 visually inspected, including Figure 5 and Tables 2–3. Initially read arXiv v2 (August 2026). Subsequently compared the official 40-page proceedings PDF against the 43-page arXiv PDF, read accepted pp. 7–10 including additional Tables 5–6, and visually inspected accepted p. 8 (Figures 5–7). These versions are not identical; details below. Downloaded PDF: `/tmp/tinycraft-paper-reading/world-in-world.pdf`.

**Sources.** [Official proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/5b4263be85820683d78675cc18d2efc7-Paper-Conference.pdf); [Official oral](https://iclr.cc/virtual/2026/oral/10006575); [full paper HTML](https://arxiv.org/html/2510.18135v2); [PDF](https://arxiv.org/pdf/2510.18135v2); [author project](https://world-in-world.github.io/).

## Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling

**Claim and evidence.** LPWM combines object-like latent particles with per-particle stochastic latent actions, removing explicit tracking while enabling conditioning and video prediction. Its evidence moves from seven visual datasets and matched patch-based DVAE comparisons to ablations and two imitation-learning environments. The main-text method distinguishes each module's purpose and connects the novel context mechanism to simultaneous local dynamics. This is an architectural contribution supported by video and control evidence, not a theorem about general intelligence.

**Lesson for Tinycraft (inference).** Borrow the separation between representation quality, predictive dynamics and downstream action. A compact paired trajectory panel should show the same imagined and executed episode rather than unrelated attractive videos. Figure 3 uses stable pastel colors for encoder, decoder and dynamics, with a zoomed context panel; this is effective for explaining one mechanism. Tinycraft should keep its simpler diagnostic mechanism equally legible and retain a genuinely matched baseline.

**Scope.** LPWM is trained with reconstruction and is not a JEPA. It requires small camera motion and recurring scenarios. It trails EC-Diffuser on harder PandaPush tasks, achieves only 6% on OGBench task2, and fails tasks4–5. These failures matter when citing its decision-making scope.

**Reading ledger.** HTML Sections 1–8 including methods, experimental tables and explicit limitations; selected Appendix A.4 encoder/decoder/context details. PDF pages 5 and 7 visually inspected, including architecture Figure 3. Downloaded PDF: `/tmp/tinycraft-paper-reading/lpwm.pdf`.

**Sources.** [Official proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/cf7ba4b2d14e0f6a0e8247af77745094-Paper-Conference.pdf); [Official oral](https://iclr.cc/virtual/2026/oral/10007677); [full paper HTML](https://arxiv.org/html/2603.04553v1); [PDF](https://arxiv.org/pdf/2603.04553v1); [author project](https://taldatech.github.io/lpwm-web/).


## TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning

TD-JEPA turns latent prediction into the core training objective for reward-free, offline zero-shot RL. Its concrete advance is policy-conditioned, multi-step prediction through temporal-difference targets, with separate state and task encoders. The exposition builds one bridge at a time: prediction to successor features, successor features to reward projection, and that projection to policies. Idealized theory connects representation gradients to successor-measure factorization and bounds policy-evaluation error; non-collapse assumes continuous-time updates with optimal predictors. Other principal results impose tabular/linear structure, covariance normalization, uniform state sampling, and symmetric transitions.

Evidence spans 65 tasks across 13 ExoRL/OGBench datasets, pixels and proprioception. Baselines share architecture and comparable tuning grids; ablations isolate prediction horizon, policy conditioning, and encoder asymmetry. Results acknowledge task-specific losses. Visually inspected Figure 2 uses diverging probability-of-improvement heatmaps, numerical uncertainty, and dotted significance boundaries; Figure 3 shows signed per-domain differences, including regressions.

Tinycraft lesson: organize every experiment around a falsifiable question; match architecture and tuning; distinguish reward-free pretraining from test-time reward inference. A failure of one-step planning does not refute this policy-conditioned successor-feature result.

Reading ledger: complete main text, printed pp. 1–10, §§1–7 of the official conference version, established by reading arXiv v1 §§1–6, text-diffing the main text, and reading the added/revised conference passages; Table 1; Figures 1–4. Visual inspection: official conference PDF p. 9, Figures 2–3, rendered locally at 1800 pixels. Version: official ICLR 2026 proceedings, PDF creation date 25 February 2026, 34 pages, SHA-256 `8153e49f01785ed0834475e2aa2f6cae72b8bfdf801e43fa49c2a62dc46d7264`.

Sources: [official conference paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/3d158f054ff0cb83397367234899db07-Paper-Conference.pdf), [earlier arXiv version](https://arxiv.org/pdf/2510.00739), [confirmed ICLR 2026 oral](https://iclr.cc/virtual/2026/oral/10009367).

Conference-version delta: added §5 Related Work explicitly distinguishes non-contrastive TD-JEPA from contrastive FB; clarifies that its asymmetric practical objective uses forward TD prediction, not bidirectional prediction; and identifies evaluated BYOL-gamma as a new successor-feature instantiation. The expanded conclusion emphasizes predicting diverse long-term behaviors. Experiments move to §6 and conclusion to §7. All 30 normalized Table 1 numerical rows match arXiv v1 exactly; the reviewed conclusions and visual lessons remain unchanged. Equation 2 corrects the reward's integration variable from s-prime to s-plus. The official page has smaller figures within the ICLR layout and visible PDF link outlines; those outlines are not a recommended design choice.

## Information Shapes Koopman Representation

This paper frames finite-dimensional Koopman learning as a tradeoff between simple temporal dynamics and retaining enough independent modes. It develops an information-flow argument, relates retained information to operator spectra, and motivates mutual-information, structural-consistency, reconstruction, and covariance-entropy terms. Its strongest storytelling device is the repeated connection between a mathematical desideratum, an objective term, and a corresponding empirical diagnostic.

Experiments cover physical forecasting, image-based control, and graph dynamics. Main-text diagnostics expose operator eigenvalues, latent manifolds, forecast errors, noise robustness, and objective ablations. The authors acknowledge missing sample-complexity and non-asymptotic convergence analysis. Visually inspected Figure 1 combines architecture, information decomposition, and spectral allocation with consistent pastel colors. Figure 6 makes omitted objectives interpretable through visibly different pendulum manifolds; Figure 5 pairs clean/noisy rollout curves with uncertainty bands. These are stronger design references than decorative color alone.

Tinycraft lesson: distinguish non-collapse from preserving useful state, then test action effects and long-horizon consequences directly. Pair a simple mechanism illustration with matched measurements. Do not import linear-Gaussian Koopman assumptions as guarantees for a nonlinear action-conditioned JEPA. Theory below was read, not independently proved.

Reading ledger: complete main text, printed pp. 1–10, §§1–5; Tables 1–2; Figures 1–6. Also Appendix D and the opening of Appendix F through F.2. Visual inspection: PDF pp. 5 and 10, Figures 1, 5–6, rendered locally at 1800 pixels. Version: arXiv v2, 4 February 2026; official proceedings PDF subsequently verified text-identical except removal of the arXiv identifier (50 pages each).

Sources: [paper PDF](https://arxiv.org/pdf/2510.13025v2), [full HTML](https://arxiv.org/html/2510.13025v2), [confirmed ICLR 2026 oral](https://iclr.cc/virtual/2026/oral/10009364).

### Preliminary theoretical caution (not a full-proof audit)

Appendix Fact F.1 infers H(z) <= H(x) from data processing for a probabilistic encoder. That inference alone is not generally valid: a stochastic encoder can add independent noise. Its intended assumptions and the distinction between encoded and autoregressively propagated variables require clarification before reusing subsequent information inequalities. This observation is scoped to that stated inference, not a verdict on all results.

Local reading artifacts: `/tmp/td-jepa.pdf`, `/tmp/td-jepa.txt`, `/tmp/td-jepa-figure2.png`, `/tmp/koopman.pdf`, `/tmp/koopman.txt`, `/tmp/koopman-figure1.png`, `/tmp/koopman-figure6.png`.

## Official proceedings provenance audit

All four official abstract/PDF links were found through the [ICLR 2026 proceedings index](https://proceedings.iclr.cc/paper_files/paper/2026), rather than inferred from arXiv venue notes. SHA-256 identifies the downloaded bytes; unequal hashes do not themselves imply different scientific content. Comparison used Poppler `pdftotext -layout` over full documents plus review of changed spans.

- **LPWM:** [official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/cf7ba4b2d14e0f6a0e8247af77745094-Paper-Conference.pdf), 49 pages, SHA-256 `1d800b2ace4a789791ff0d866c0deb9c2b869620a50c821068e3d5d1ac1a8403`. ArXiv v1 hash `571c8e3a42b9c35317a30910707f525af1ab1c98256cc513f0a98a2dff5677b7`. All extracted-word differences were the removed arXiv identifier or math glyph/placement extraction; no prose/result change was found. Full-word similarity was 0.99970.
- **Koopman:** [official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/f57ffe47d0b528fbb97901d16bd4eba2-Paper-Conference.pdf), 50 pages, SHA-256 `a535aa8bd1c06b37f7bd8cc1c50bf0aa719b59677cf25590b3b346a2f76dcaca`. ArXiv v2 hash `a39d888897adc06f6dd13c920bac8e575d32295012409266a65b736728348e4b`. The only extracted-word difference is the removed arXiv identifier. The preliminary Fact F.1 caution therefore also applies to the conference text.
- **World-in-World:** [official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/5b4263be85820683d78675cc18d2efc7-Paper-Conference.pdf), 40 pages, SHA-256 `b1270828c96094cd1c7a16ccfdedddc470def295148cf3c41af7e2ea1fb1c9cf`. ArXiv v2 has 43 pages and hash `a5849ea3308b8e1806ef62ebd35617c01433989af7d30fdb040adb6c3fb8e40d`. Layout and text materially differ. Accepted Tables 5–6 add LPIPS-versus-VLM revision and synthetic-scene transfer experiments; Table 1 adds Cosmos-P2.5; discussion explicitly addresses computational cost. Accepted pp. 7–10 and those tables were read, with Figure 5 visually rechecked on accepted p. 8. Prior high-level lessons and stated caveats remain supported. Do not describe the arXiv copy as the exact accepted version.

- **TD-JEPA:** [official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/3d158f054ff0cb83397367234899db07-Paper-Conference.pdf), 34 pages, SHA-256 `8153e49f01785ed0834475e2aa2f6cae72b8bfdf801e43fa49c2a62dc46d7264`. Conference main-text additions were read and Figure 2–3 inspected again on official p. 9. All 30 Table 1 numerical rows match the earlier arXiv copy. The prior conference-version-unverified caveat is resolved; the reading ledger above names the remaining textual deltas.

Official PDFs and full-text diff artifacts are retained under `/tmp/tinycraft-paper-reading/`, with TD-JEPA under `/tmp/td-jepa-conference.pdf` and `/tmp/td-jepa-main-diff.txt`.


Standalone archive: official PDFs, extracted text, inspected renders and source hashes are retained in [reading/](reading/) and [reading/manifest.json](reading/manifest.json). Original `/tmp` locations above record the reading process; the retained archive is the deliverable.
