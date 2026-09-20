# Redcube story and novelty preflight

Review: 2026-09-19 23:22 UTC. Scope: the current `submission/v2/main.tex` and the six included main-text sections, with the local literature review and primary-source checks of Sobal et al. (2022) and Terver et al. (2026). Manuscript files were not edited. Pending goal-MPC, shared-CNN, oracle-variance, and inventory results were not treated as completed evidence. This is a story/novelty review, not a new empirical audit or an acceptance prediction.

## Verdict

**The rewrite is substantially clearer, but it has not yet raised the completed contribution above approximately 4/10 for novelty.** Conceptual clarity is approximately 7/10. The reader now understands two different ways a world model can fail. The reader still cannot identify a completed new result beyond the replicated noise-persistence × wind interaction. The planned behavioral comparisons could make that result substantially more consequential, but cannot be counted in advance. These scores are independent judgments, not target scores to print in the paper.

The strongest paper available here is a **minimal empirical test of when a particular predictive abstraction ceases to support decisions**, organized by exact known limitations. It is not a new impossibility theorem for JEPAs, and not yet evidence that optimized JEPA representations systematically choose nuisance over state. A skeptical reviewer can currently summarize it as “familiar counterexamples plus another moving-dot probe benchmark.” The revision must give that reviewer a concrete new behavioral fact that the earlier literature did not establish.

## The novelty boundary is closer than the current introduction makes clear

Sobal et al. already study action-conditioned moving objects, fixed versus changing background noise, real and predicted representation probes, a generative reference, and three training seeds. Their static Gaussian nuisance/identity-predictor example is almost exactly the nuisance construction used here. Thus neither “one moving object with nuisance” nor “non-collapse can ignore position” is a new contribution. Their observed failure is associated with persistent noise; our completed spatial-model result instead fails under **fresh noise plus wind**, while persistent gray noise preserves position. That difference is potentially useful and should be stated explicitly. [Sobal et al., Sections 2–3](https://arxiv.org/html/2211.10831).

Terver et al. explicitly acknowledge the conditional-mean limitation of deterministic MSE predictors under multimodal outcomes and note that their benchmark dynamics mitigate it. Our scalar examples give a particularly accessible account of its decision consequences, but should not be sold as discovering that limitation. [Terver et al., Section 6](https://arxiv.org/html/2512.24497v4#S6).

The draft correctly cites value equivalence and expectation models as alternatives. Keep that qualification: predicting a distribution is one solution, not a necessary property of every model that supports control. The genuinely unresolved practical question is whether the chosen representation and prediction operator preserve the expectations needed by the evaluated decisions.

## Highest-impact revisions, in order

### 1. Make the paper's contribution an answer, not an experimental agenda

`main.tex:51–55` gives three probe numbers and then describes what the experimental design pairs together. `intro.tex:41–46` describes evaluation machinery. Neither tells the reader the new behavioral finding.

In the final abstract, use the pattern **question → completed finding → practical consequence → scope**. The last two empirical sentences should contain completed outcomes with replication levels, not “we evaluate,” “the design pairs,” or pending pipelines. Until those outcomes exist, keep this as an internal draft.

The introduction should end with two or three contributions whose nouns are findings, not instruments. A defensible version of the first is:

> Across three independently trained spatial models in each of four conditions, fresh visual noise is tolerated under deterministic motion but produces a large loss of accessible position under stochastic wind; making that same noise persistent restores subpixel position readout.

The second must be filled from the actual paired control result. The third may be the exact learned-encoder variance measurement if it is informative, including a null. “We show paths, fit probes, and compare policies” is a method summary rather than a contribution.

### 2. Let the results determine the number and names of cases

The three cases currently read as an anticipated success/failure/failure progression. The first seed's supplied goal-MPC result is weak, and the remaining results are pending. A readable position representation is not a positive planning control.

If final MPC is reliable, the current Case 1 can report elementary goal-directed control. If it stays weak, rename it **“Readable state need not give a useful goal geometry”** and say so plainly. That outcome would itself fit the broader thesis, but it would invalidate a successful-planning panel. Do not search checkpoints, goals, or horizons until the desired opening success appears. Ordinary independent robustness checks remain legitimate if they are labeled as such and all outcomes are retained.

Likewise, `redcube_cases.tex:52` currently declares that a useful representation needs the right future distribution before the decoded-real control has established usefulness. Until qualified, use a question such as **“When does useful state fail in imagination?”** The learned-model comparison can locate a practical imagination deficit; the separate oracle test can isolate the omitted variance. They should not be merged into one causal claim.

### 3. Change the title from a mechanism claim to the phenomenon actually established

**Recommended title: “When Latent Prediction Is Not Enough for Control.”** An optional subtitle is “Minimal Tests of Joint-Embedding World Models.”

“Learning to Ignore What Matters” implies selective exclusion caused by the learning objective. The current replicated failure has high prediction loss and a failed readout, which does not demonstrate that mechanism. The title also omits the other half of the paper: a perfectly informative state representation whose mean prediction is inadequate for a downstream operation. “JEPAs in Stochastic Worlds” further invites a family-wide reading that the specific jointly trained, squared-loss recipe cannot support.

### 4. Make the surprising noise interaction the main empirical idea

The fact that persistent noise helps the spatial model under wind is the strongest completed distinction from the familiar nuisance story. It should be visible in the main figure and stated in the abstract or introduction, not treated as a caveat following the failure.

A small 2 × 2 panel is more informative than a long probe table: columns deterministic motion / wind; rows changing / persistent noise. Within each cell, show one fixed real frame, its reconstruction from real encoding, and the actual behavioral endpoint when available. A small position marker or one line of error can support the image; a large table of diagnostic variants need not lead the story.

Retain `redcube_cases.tex:46–47`: the approximately 0.733 prediction loss prevents describing this result as convergence to the static-nuisance optimum. A finite-budget failure can be an important finding, but the conclusion should be **a reproducible failure of the tested recipe**, not a proven global-objective attractor.

### 5. Connect the two mechanisms with one precise sentence, not another theorem detour

The main text currently places three useful examples in sequence, but the common principle remains abstract. A compact bridge is:

> Joint predictive learning chooses which uncertainty to represent; mean-state planning then chooses which uncertainty to discard. Neither choice is justified by prediction accuracy alone.

For precision, an optional two-line equation can explain the first clause: at an ideal predictor for a fixed encoder, squared prediction risk equals expected conditional latent variance. When the encoder is also optimized, both the targets and their conditioning information depend on that encoder. This can favor predictable features under a fixed marginal constraint. It does not prove that a finite neural fit reaches the nuisance optimum.

The downstream principle is equally concrete: a latent mean preserves expectations of affine functions of that latent. Squared goal costs additionally depend on second moments; nonlinear recursive dynamics need further closure conditions. Present this as an explanatory lens, not a new theorem. Keep the user's ±1 versus 0.1 example and the two-step nonlinear example. The horizon-three outcome enumeration is correct, but adds detail without strengthening the central point and can move to the caption or appendix.

Do not add the cropped-prefix construction to the main story merely to increase theorem count. It introduces another mechanism and is not the fresh-wind experiment. Its value depends on a completed, clearly related intervention.

### 6. Keep one fairness sentence in the main text and the audit machinery behind it

The user is right that heads and probe engineering should not become the topic. The current shared-policy paragraph is much improved. Reduce it to the minimum a reviewer needs to interpret the behavior: same CNN/GRU, native rewards/actions, matched policy budgets, and real-simulator controls using each model's decoded observations. Then state once that this compares complete recipes rather than isolating a single loss term.

The exact-simulator decoded-observation controls should be visually adjacent to imagined-policy outcomes. They decide whether “state retained but imagination fails” is actually supported. Their role can be clear without a main-text tour of reward-head losses. When the control does not qualify, state that limitation in the outcome sentence rather than burying it in an appendix.

### 7. Add Craftax only for the correct kind of evidence

Meaningful inventory acquisition could show that the minimal failure matters in a richer environment. If the new comparison produces such a result under the declared protocol, one compact transfer section is appropriate. Keep the specific achievement and model/policy replication explicit. Do not place a new large experiment between the minimal mechanism and its direct test.

If inventory results are incomplete or inconclusive, the redcube paper should stand without them. The existing few-step local-goal experiment should not return as a planning headline.

## Specific line-level instructions to the author

| Location | Change |
|---|---|
| `main.tex:36` | Replace the causal/family-wide title with the phenomenon title above. |
| `main.tex:47–51` | Call the exact examples explanatory consequences of conditional-mean prediction; do not let the abstract imply new general theory. |
| `main.tex:51–55` | Lead empirical contribution with the full noise-persistence × wind interaction; replace protocol language with completed behavioral outcomes when available. |
| `intro.tex:4–5` | Make irrelevance task-relative: in these tasks, texture does not affect the requested goal, dynamics, or reward. A general world model cannot universally declare all background texture irrelevant. |
| `intro.tex:23–32` | Name questions or observed cases, not predetermined outcomes. Update Case 1 after all admitted MPC results. |
| `intro.tex:41–46` | Replace most evaluation-inventory prose with the specific contributions and a single scope sentence. |
| `mechanisms.tex:38–52` | The two-step recursion example suffices. Reclaim space from horizon three for the empirical result. |
| `mechanisms.tex:63–75` | Keep the Sobal attribution immediate. Connect ideal risk to conditional variance only with the fixed-encoder/optimal-predictor qualification. |
| `worlds.tex:18–30` | State that this is the tested spatial LeWM-style objective and implementation, not a published-benchmark reproduction or the entire JEPA family. |
| `redcube_cases.tex:7–10` | Position accuracy establishes the motivation for a planning test, not its success. |
| `redcube_cases.tex:26–47` | Preserve the persistent-noise rescue and high-loss qualification prominently. |
| `redcube_cases.tex:75` | “Informative frozen encoder” must be linked to the verified position diagnostic; do not silently upgrade it to control sufficiency. |
| `related.tex:9–11` | Explain that Sobal already has the same minimal moving-object/nuisance design, then identify what stochastic dynamics and actual behavior add here. |
| `discussion.tex:10–15` | Replace conditional “would compare” wording only after results; end with the actual demonstrated boundary rather than reciting the planned apparatus. |

## What would change the novelty judgment

More polished prose or more variants of the same probe would not materially change 4/10. The existing small experiments can change it if they produce a reliable **new separation**: a readable observation interface with successful real control but failed learned imagination under the same policy package; a replicated dynamic-noise failure connected to actual behavior; or a practically meaningful learned-encoder difference between exact mean-goal scoring and exact expected-goal scoring. The last of these remains a latent-objective result unless the physical consequence is independently demonstrated.

A null can also be useful. For example, if the shared policy removes an apparent JEPA/flow gap, or an exact mean ranks actions correctly in the wind world, that is an important boundary on the original argument. A strong critical paper earns trust by reporting where its criticism does not apply. None of these outcomes automatically merits 9/10, and no current editorial change can independently certify strong acceptance.

## Reviewed source snapshot

SHA-256 at review completion:

```text
9bfc6cdb307129b5f06091fb153fa7bda0164fe0b42f80740a1727cf2c404f50  main.tex
ce91f01fe328da33e304a2f4dfade2732d008e36f801890f3e0b147f7e54e6a2  intro.tex
61ed05d02893ce7c24ded7ba876afb029e99757309c30a1692822a0985292d65  mechanisms.tex
f260c405a75355ce75f97eb9822ef232bcaf115370acd0e98508c516b5d13116  worlds.tex
e805bfd4206b2543fefe3be2d4abca1008ee2c8ddff3b6b3c01c91afe2f4c10a  redcube_cases.tex
5b44bb1b4d40d85dafa87bdbfc59454d9419fbaf9dd79f6b8abd56956611d36b  related.tex
361ad0e0b7d48ff0bd05abff070074888418671408ed16b1c71c50e83230d753  discussion.tex
```

An independent bounded abstract/introduction/discussion reader reached the same central assessment without reading the other reviews: the completed novelty is the noise × wind finding; the title overstates the mechanism; and the final abstract currently announces protocols rather than results.
