# Three ICLR 2026 oral exemplars for a diagnostic world-model paper

Primary-source reading completed 2026-09-19. These are official oral selections, not a claimed numerical ranking of the conference's best papers. [Official oral program](https://iclr.cc/virtual/2026/events/oral).

## Difficult Examples Hurt Unsupervised Contrastive Learning: A Theoretical Perspective

[Official paper PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/a559a5a8aa5ae6682ced009ad97cdb16-Paper-Conference.pdf)

**Claim and evidence.** Some apparently informative training examples can impair unsupervised contrastive representation learning. The argument proceeds from a counterintuitive empirical observation, through a controlled image-mixing intervention, to a similarity-graph model and testable changes to the loss. Experiments separate removal, margin adjustment, temperature adjustment, and their combination. Crucially, the selected-pair interventions are compared with applying the same intervention to all samples; this is a mechanism control, not merely another leaderboard entry. Four image datasets and three-run averages anchor the main experiments, with further generalization and sensitivity checks in the appendix.

**Lessons for Tinycraft.** A strong diagnostic story can begin with the failure that requires explanation, then introduce the smallest environment that isolates it. Every proposed repair should have a matched control addressing the mechanism: for example, prediction-space changes must preserve architecture, data, budget, and evaluation. Figure 3 makes a formal construction understandable by linking colored sample pairs to a four-point matrix. This is more useful than a large decorative architecture graphic. Figure 1 is visually too small to emulate.

**Limits.** The theorems concern bounds under stylized spectral-contrastive assumptions; a worse upper bound alone does not prove worse actual error. Avoid inheriting the paper's broad language about universality.

**Reading ledger:** main text pp. 1-10, especially Sections 2-5; Appendix A.2-A.5, pp. 15-17. Visually inspected rendered pp. 1 and 4, Figures 1 and 3. Proof appendices were not independently verified.

## Train-before-Test Harmonizes Language Model Rankings

[Official paper PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/79a97ba70ee6aff126f3090b10ad6702-Paper-Conference.pdf)

**Claim and evidence.** Model rankings become substantially more consistent when models receive a standardized opportunity for task adaptation. The study evaluates 61 models on 24 benchmarks, selects finetuning checkpoints using separate validation data, and compares all benchmark pairs. Average Kendall correlation rises from 0.52 to 0.76; 274 of 276 benchmark pairs improve. Further analyses connect adaptation results to perplexity and examine the score matrix's dominant principal component. The contribution is an evaluation distinction with consequences, not a new complicated architecture.

**Lessons for Tinycraft.** Define exactly what each metric measures before presenting disagreement between metrics. Predictive fit, recoverable state, and useful control are different questions. A central figure should make the discrepancy visible, followed by a matched intervention and a quantitative summary. Figure 3's two heatmaps use identical scales, printed values, and a direct caption to make an aggregate claim immediately assessable. Figure 1's before/after ranking plot communicates the qualitative change strongly but its 61 labels are overloaded; adapt the paired logic with fewer categories. Separate conclusions about frozen models from conclusions after additional training.

**Limits.** Models are at most 14B parameters, most tasks use multiple-choice accuracy, and benchmark selection excludes several problematic finetuning cases. Ranking agreement remains imperfect; adaptation potential does not replace immediate deployment performance.

**Reading ledger:** main text pp. 1-10, particularly Sections 1.1, 3.1-3.4, and 4. Visually inspected rendered pp. 2 and 6, Figures 1 and 3. Supplementary statistical procedures were referenced but not fully audited.

## RealPDEBench: A Benchmark for Complex Physical Systems with Real-World Data

[Official paper PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/aed13d8a5ee83a9c6880285b69d0ba81-Paper-Conference.pdf)

**Claim and evidence.** Simulation-based evaluation misses important physical prediction failures. Five paired measured/simulated datasets, three training regimes, nine metrics, and ten baseline families expose this gap. All regimes share real-world evaluation sets; splits are at the physical-parameter level. The experiments distinguish simulation transfer, additional real-data training, local prediction error, physical summary statistics, and autoregressive error accumulation. Some strong short-horizon models degrade rapidly during rollout, and model rankings depend on the property being measured.

**Lessons for Tinycraft.** Define the scientific question for every environment rather than introducing environments as a list. Show a common sequence across tasks: observation, target, prediction, and the failure diagnostic. Figure 1 uses aligned rows and consistent environment columns to explain the data mismatch before equations. The colored main table keeps dataset groups distinct and prints exact values. For Tinycraft, use restrained background tints and dark text; the original's very pale colored numbers sacrifice contrast. Pair predictive error with a physically or behaviorally meaningful metric, and state complete rollout length and feedback semantics.

**Limits.** The benchmark is concentrated in fluid and combustion systems; strong out-of-distribution evaluation is missing. Simulated pretraining's average benefit is not a guarantee for every model-metric cell. Its physical metrics do not establish general intelligence.

**Reading ledger:** main text pp. 1-10, Sections 1-5; Appendix F, p. 37. Visually inspected rendered pp. 1 and 8, Figure 1 and Table 1. Dataset-specific appendices and all baseline implementations were not audited.

## Common recommendation

Use one falsifiable headline: **what a low world-model training loss fails to guarantee, the controlled reason, and which ordinary objective change repairs it under matched budgets**. Show the hard environment first to establish relevance, then descend to simpler environments to identify the mechanism. Return to the hard environment only when the evidence supports that return. Keep null results and limits visible. An oral paper's visual style is evidence organization, not a substitute for stronger experiments.

Local review artifacts, including PDFs, extracted text, and inspected PNGs: `/tmp/tinycraft-orals-diagnostic/`.


Standalone archive: official PDFs, extracted text, inspected renders and source hashes are retained in [reading/](reading/) and [reading/manifest.json](reading/manifest.json). Original `/tmp` locations above record the reading process; the retained archive is the deliverable.
