# ICLR 2026 exemplars and ICLR 2027 revision checklist

Verified 19 September 2026. This is a source-backed writing/evidence review, not a certification of the Tinycraft results or an acceptance prediction.

## Verdict

The strongest route is a **diagnostic paper with a controlled explanation**: a prediction objective can look successful while losing information needed for decisions; establish exactly when this happens, separate representation loss from predictive averaging, and test one ordinary matched intervention. A complex benchmark motivates the question; simpler environments identify its cause. A new method is useful only if it makes this argument more convincing.

This recommendation is consistent with the official reviewer guide, which evaluates clear questions, motivation, correct and rigorous support, and new knowledge; leaderboard supremacy is not required. [ICLR 2027 reviewer guide](https://iclr.cc/Conferences/2027/ReviewerGuidelines).

There is no official ranked “top ten.” The reading set below contains **both ICLR 2026 Outstanding Papers, the Honorable Mention, and seven relevant official oral selections**. Selection of the seven orals is ours, chosen for relevance to world models and rigorous diagnosis. The committee recognized conceptual insight, careful evaluation, and principled methodology even when downstream gains were modest. [Official awards](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/), [official oral program](https://iclr.cc/virtual/2026/events/oral).

## Ten-paper reading ledger

### 1. LLMs Get Lost In Multi-Turn Conversation — Outstanding Paper

The scientific move is a controlled reformulation of an existing task: Full, Concat, and Sharded conditions separate information loss from the effect of interaction. The study decomposes average performance into aptitude and reliability, then tests remedial interventions and a translation case where the failure is absent. Figure 1 puts task setup, quantitative effect, and behavior together. Table 1 shades degradation relative to the Full control, making the scientific comparison visible. Main results span six tasks and fifteen models, but simulated conversations are not human interaction.

**Transfer:** build an equivalent confound control for JEPA; a lower prediction loss must not be mistaken for better state information or control. Show the condition in which the failure disappears. Borrow the table's semantic tinting, not its dense tiny labels or repeated broad rhetoric.

**Read:** original arXiv main text §§1–9; official conference main text, especially §§3–6 and limitations. Rendered official PDF pp. 1 and 7 inspected (Figure 1, Table 1, Figure 5). No reproduction.

[Official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/59f6421e64707225fdf5b28840679a07-Paper-Conference.pdf).

### 2. Transformers are Inherently Succinct — Outstanding Paper

One conceptual reframing, succinctness rather than expressivity alone, motivates a connected series of formal statements. Definitions distinguish existential witness-family lower bounds from universal conversion upper bounds. The main argument supplies proof sketches and moves detailed constructions to appendices. Its scope is a formal unique-hard-attention abstraction with specific precision assumptions; it does not establish that training discovers these constructions or explain all empirical transformer performance.

**Transfer:** distinguish “this counterexample exists,” “this objective has this optimum under these assumptions,” and “our trained networks exhibit this behavior.” A compact theorem should explain the measured failure, not provide a rhetorical universal guarantee. There is no need to invent an architectural innovation for conceptual value.

**Read:** main text §§1–6, definitions, theorem statements and proof sketches; official/arXiv versions cross-checked. Official PDF p. 2 visually inspected. No figures in the main text; typography and argument structure, rather than visual ornament, carry the paper. Appendix proofs not independently audited.

[Official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/5f7804e8855efe5554025217abc49315-Paper-Conference.pdf).

### 3. The Polar Express — Honorable Mention

The paper isolates one practical subproblem inside Muon and derives a polynomial composition method for the relevant cost/precision regime. It connects an explicit minimax objective, a greedy-optimality theorem, finite-precision modifications, and an implementable routine. Experiments use matched polynomial degrees/iteration counts, learning-rate sweeps, synthetic and real gradient matrices, actual training, and runtime/accuracy ablations. More accurate polar decomposition eventually stops improving model quality; this limitation is reported directly. Figure 1 pairs hyperparameter sensitivity with the learning curve.

**Transfer:** change one objective component at fixed capacity and data; report the compute contract and the null region. Mechanism accuracy and useful downstream behavior are separate outcomes. Prefer a modest repeatable difference over a spectacular comparison with mismatched baselines.

**Read:** main text §§1–4, theoretical formulation/statements, finite-precision discussion and all numerical experiments; official PDF p. 2/Figure 1 inspected. Appendix proofs not independently verified.

[Official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/df88b275bef31ac96c85f0c4013734fc-Paper-Conference.pdf).

### 4. World-In-World: World Models in a Closed-Loop World — Oral

The paper asks whether a simulated future improves the decision that is actually executed. It evaluates four task families with proposal/simulation/revision interfaces and separates visual quality, controllability proxies, data scale, and inference budget. Figure 5's paired scatterplots expose surrogate/outcome disagreement; lightly colored table headers group environments. The proposed controllability proxy is 1-LPIPS, and manipulation gains can depend heavily on the proposal policy.

**Transfer:** pair every important latent diagnostic with measured control utility; state planner/proposal policy and action interface. Avoid implying that perceptual metrics establish causality.

**Read:** HTML §§1–5, Tables 1–4, Figures 1–7 captions; PDF pp. 3 and 9 inspected. Initially read arXiv v2; then compared the official 40-page PDF, read accepted pp. 7–10 including added revision-policy/transfer Tables 5–6, and inspected official p. 8/Figures 5–7.

[Official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/5b4263be85820683d78675cc18d2efc7-Paper-Conference.pdf), [official oral](https://iclr.cc/virtual/2026/oral/10006575).

### 5. Latent Particle World Models — Oral

LPWM combines object-like particles and stochastic latent actions, evaluating video prediction, matched patch-based alternatives, ablations, and imitation learning. Its module explanations connect architecture choices to specific modeling problems. Figure 3 uses stable component colors and a zoomed context panel. It reconstructs observations and is not a JEPA. Limits include small camera motion, recurring scenarios, and substantial failures on harder control tasks.

**Transfer:** explain representation, dynamics, and action utility separately. Show matched imagined/executed trajectories; do not use attractive but unrelated videos as evidence. Cite the stochastic modeling result with its reconstruction and control scope intact.

**Read:** HTML §§1–8 and selected Appendix A.4; PDF pp. 5 and 7 inspected (architecture Figure 3).

[Official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/cf7ba4b2d14e0f6a0e8247af77745094-Paper-Conference.pdf), [official oral](https://iclr.cc/virtual/2026/oral/10007677).

### 6. TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning — Oral

Temporal-difference JEPA learns policy-conditioned, multi-step predictive representations from offline reward-free transitions, with separate state/task encoders. The exposition connects prediction, successor features, reward inference, and policy retrieval. Experiments cover 65 tasks in 13 datasets with matched architectures and tuning; ablations separate horizon, policy conditioning, and encoder asymmetry. Theory uses idealized optimization and structural assumptions. Figures 2–3 show uncertainty, signed per-domain effects, and regressions.

**Transfer:** a one-step JEPA failure does not refute all latent prediction or policy-conditioned successor representations. Use similarly matched tests, scoped claims, and visible negative outcomes.

**Read:** arXiv main text followed by a diff against the official conference main text pp. 1–10, §§1–7; all changed/additional passages read. Table 1 numerical rows unchanged. Official p. 9/Figures 2–3 visually inspected. The supplement records exact version differences.

[Official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/3d158f054ff0cb83397367234899db07-Paper-Conference.pdf), [official oral](https://iclr.cc/virtual/2026/oral/10009367).

### 7. Information Shapes Koopman Representation — Oral

The narrative repeatedly connects a mathematical requirement, an objective term, and a diagnostic: simple temporal dynamics must coexist with retained independent modes. It evaluates spectra, manifolds, forecasting, noise robustness, and loss ablations across several domains. Figure 1 uses mechanism colors; Figure 6 makes omitted objectives visible through changed pendulum manifolds. Non-asymptotic/sample-complexity analysis remains incomplete.

**Transfer:** non-collapse alone is not preservation of useful state. Pair a simple mechanism diagram with an intervention and its behavioral consequence; do not carry linear-Gaussian assumptions into nonlinear JEPA claims.

**Read:** arXiv v2 main text pp. 1–10, §§1–5; Tables 1–2, Figures 1–6, Appendix D and opening of F. PDF pp. 5 and 10 inspected. A preliminary concern about an entropy inference in Appendix F.1 is recorded in the supplemental notes; this is not a complete proof verdict.

[Official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/f57ffe47d0b528fbb97901d16bd4eba2-Paper-Conference.pdf), [official oral](https://iclr.cc/virtual/2026/oral/10009364).

### 8. Difficult Examples Hurt Unsupervised Contrastive Learning — Oral

A counterintuitive empirical observation leads to a controlled image-mixing intervention, a similarity-graph model, and testable objective changes. Removing selected examples, adjusting margin, and changing temperature are compared with broader interventions on all samples. Figure 3 links colored sample pairs to a four-point matrix. Theorems compare bounds under stylized assumptions; a looser bound is not proof that every actual error worsens.

**Transfer:** give each repair a control that addresses the hypothesized cause. The smallest environment can establish mechanism after the realistic failure establishes relevance. Do not hide assumptions behind a broad title.

**Read:** official main text pp. 1–10/§§2–5; Appendix A.2–A.5 pp. 15–17. Rendered pp. 1 and 4/Figures 1 and 3 inspected. Proofs not independently verified.

[Official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/a559a5a8aa5ae6682ced009ad97cdb16-Paper-Conference.pdf).

### 9. Train-before-Test Harmonizes Language Model Rankings — Oral

The contribution is an evaluation distinction: deployment performance and potential after standardized adaptation are different objects. Validation-selected finetuning across 61 models and 24 benchmarks produces much more consistent rankings, supported by all-pairs analyses and score-matrix structure. Figure 3 compares heatmaps with identical scales and printed values. Scope includes relatively small models, mainly multiple-choice tasks, and benchmark selection that excludes some problematic finetuning cases.

**Transfer:** define prediction accuracy, recoverable state, and control utility separately. State whether a claim concerns frozen models or retraining. Use a common visual scale for before/after comparisons and preserve the exclusions.

**Read:** official main text pp. 1–10, especially §§1.1, 3.1–3.4, 4. Rendered pp. 2 and 6/Figures 1 and 3 inspected; supplementary statistics not fully audited.

[Official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/79a97ba70ee6aff126f3090b10ad6702-Paper-Conference.pdf).

### 10. RealPDEBench — Oral

Paired physical measurements and simulations expose gaps hidden by simulated evaluation. The same real test sets support three training regimes, multiple predictive/physical metrics, and autoregressive rollout comparisons. Figure 1 aligns environment examples and data sources; the main table groups datasets with restrained color and explicit values. Some strong short-horizon models deteriorate during rollout. Domain coverage is limited, and aggregate pretraining gains do not hold in every cell.

**Transfer:** every environment should answer a named scientific question. State complete rollout length, feedback semantics, and the quantity each metric measures. Pair local prediction error with a behaviorally meaningful outcome. Use dark numerical text rather than the original's very pale numbers.

**Read:** official main text pp. 1–10/§§1–5; Appendix F p. 37. Rendered pp. 1 and 8/Figure 1 and Table 1 inspected. Dataset-specific implementations not audited.

[Official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/aed13d8a5ee83a9c6880285b69d0ba81-Paper-Conference.pdf).

## Concrete revision recipe (our synthesis)

1. **Name a bounded object.** “Jointly trained squared-error predictive representations in stochastic environments” is defensible if it describes the code. “JEPA cannot model the world” is not. State what is jointly trained, what is frozen, the target update, anti-collapse regularizer, action conditioning, and temporal horizon.
2. **Give the reader one question.** When can accurate latent prediction fail to preserve information needed for decisions? Separate encoder information loss from the conditional-mean predictor's treatment of uncertainty. Do not conflate either with decoder blur, planner optimization failures, or missing action information.
3. **Lead with relevance, then simplify.** Show matched Craftax/Tinycraft evidence first if it is auditable. Descend to the two-wall task and a minimal stochastic construction to identify the cause. Return to the complex environment only when the proposed intervention has valid evidence there. If Craftax comparisons are unmatched, explicitly label them motivating observations or move them to the appendix.
4. **Treat noise as an intervention.** Static appearance noise, time-varying observation noise, and stochastic transitions are scientifically different. Define which latent state/action is held fixed and which source of randomness changes. Recoloring walls may help a schematic but must not silently change trained/test observations. Label illustrative renderings. Do not add noise just to make a chosen method win.
5. **Use matched ablations.** Fix model capacity, latent dimension, dataset, augmentations, training updates, selection protocol, and planning budget where possible. A frozen encoder + alternative predictor isolates predictive aggregation; a jointly trained encoder ablation addresses information loss. Report exceptions, extra compute, and null results. Probe training must not leak evaluation trajectories or task labels into world-model training.
6. **Keep intervention ambitions small.** A PVE-style objective can test a hypothesis; it need not be branded a new architecture. Any task reward/value supervision must be declared and separated from reward-free world-model claims. Same-capacity comparisons are more useful than latent inflation.
7. **Bind every number to evidence.** Retain checkpoint identity, data revision, seed count, evaluation episodes, horizons, estimator, intervals, selection rule, and output path. Distinguish training-seed uncertainty from episode uncertainty. Trace the HTML table, PDF table, plot, and claim to one audited numeric source.
8. **Write conclusion-first paragraphs.** State the observation, show the controlled comparison, interpret only what it identifies, and state the remaining ambiguity. Three substantive contributions are enough: measured failure, explanation, and tested implication. Remove field-wide promises that the experiments do not establish.

### Suggested visual system

- Use a neutral white paper background and a small stable palette: deep blue for deterministic squared error, teal for a distributional/PVE intervention, amber for uncertainty, muted red only for a failure/distractor annotation. Color must map to a scientific meaning throughout.
- Figure 1: a left-to-right account of observation → representation → imagined outcome → action, with a compact measured discrepancy underneath. Make clear which elements are schematic and which come from artifacts.
- Core result: aligned panels for predictive loss, recoverable state information, and real control, using the same model order. Do not imply different metrics share a numerical scale.
- Mechanism figure: fixed-state samples and their conditional mean, followed by action rankings. A simple exact counterexample can be more convincing than a large architecture diagram.
- Results table: grouped environment headers with pale tints, exact values and uncertainty, bold only for supported comparisons, and footnotes exposing protocol differences. Mark unavailable results visibly rather than smoothing them away.
- Companion HTML: wider figures, hoverable provenance, direct artifact links, accessible color/labels, responsive layout. All essential conclusions must remain in the PDF and static view; no interaction should be needed to discover a caveat.
- Visually inspect desktop/mobile HTML, every PDF page, and grayscale figure exports. A high-resolution figure that becomes illegible at final paper width has failed QA.

## ICLR 2027 submission rules that govern the rewrite

- **Nine main-text pages** at submission; ten only during rebuttal/camera-ready. References and appendices are outside the main limit. Reviewer attention to appendices is optional, so essential evidence belongs in those nine pages.
- Use the unmodified official 2027 LaTeX style. Color figures are explicitly permitted if they remain interpretable in grayscale. Figure captions go below; table captions above. Light table tinting is a reasonable design interpretation, not an explicit rule in the guide.
- Main paper, supplementary material, and demo links must preserve anonymity. Use third-person self-citations and anonymous code; check HTML metadata, URLs, filenames, and embedded video credits too.
- **AI-use disclosure is required**, including substantive methodology, implementation, analysis, and interpretation performed in this project. Do not describe this work merely as proofreading. Author responsibility and any human verification must be factual.
- The abstract deadline was 18 September 2026, 23:59 AoE (19 September 13:59 Belgrade); it has passed. The paper requires an existing genuine abstract. The full-paper deadline is 25 September 23:59 AoE (**26 September 13:59 Belgrade**). Author membership already froze; title and abstract can change until the paper deadline within the substantive scope.

[Author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), [official style archive](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip), [AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors), [call and dates](https://iclr.cc/Conferences/2027/CallForPapers). The full rules audit and downloaded-style line references are in [the rules audit](iclr2027_rules.md).

## Independent review acceptance conditions

An independent critique should evaluate the submitted artifact without a target score in its instructions. It should reproduce the principal tables from retained outputs, inspect theorem assumptions and proof steps, challenge the strongest causal interpretation, verify baseline fairness, and identify the nearest 2026 work. A favorable agent review is not an external ICLR decision and does not certify 9/10. Deliver its actual rating, objections, evidence, and unresolved limitations without filtering dissent.

A directly relevant additional novelty check is **Temporal-Distance JEPA: Plan-Aware Representation Learning for Latent World Model Predictive Control** (July 2026), distinct from the ICLR oral temporal-difference TD-JEPA. Its abstract discusses latent Euclidean planning and LeWM/Two-Room evaluations. It was flagged for the broader literature reviewer; only its abstract was read in this workstream. [Primary record](https://arxiv.org/abs/2607.25337).

Detailed reading notes: [world-model orals](iclr2026_world_orals.md), [diagnostic orals](iclr2026_diagnostic_orals.md). All ten official PDFs and inspected renders are retained in [reading/](reading/), with hashes and sources in [the reading manifest](reading/manifest.json). This workstream did not modify the original repository or run experiments.
