# Red-cube article: proposed reading flow

Status: layout and evidence contract only. Final case selection and scientific claims await root/author approval from completed evidence. The existing `submission/html` and `submission/html-v2` are archived drafts and are not edited here.

## One story, three concrete cases

Working title: **What a world model learns to ignore**.

Open with the red cube and its actual environment, not a taxonomy or benchmark dashboard. A 40–60-word introduction states the practical question: can a compact predictive model ignore visual noise while keeping the state and uncertainty needed to act? One short environment strip explains the four controls, boundary deaths, and whether noise/wind changes across each of the three selected cases. No broad JEPA-versus-generative conclusion precedes the evidence.

The main article should be roughly 1,200–1,600 words, with one primary visual and at most one compact secondary visual per case. Method colors remain JEPA rust, flow violet, simulator teal; native frame colors remain untouched. A fixed few actual examples replace large selectable benchmark dashboards. Optional scene switching changes among declared examples of the same case only.

### 1. First show the model working

**Reader question:** What useful abstraction can this JEPA learn?

Show the actual noisy observation, the real future, the JEPA future, and the flow future under exactly the same recorded action sequence. The prose explains what stays predictable and what can safely be ignored in this verified case. A compact numerical sentence reports its primary prediction/control result and replication scope; there is no multi-condition table.

The final case may be deterministic motion under fresh noise or another positively verified condition. Do not call high still-world survival evidence of using observations: state-blind alternating actions can achieve that score. If meaningful control is unavailable, call this a successful prediction case and say exactly what was measured.

**Visual:** synchronized 3-row future strip, rows Real / JEPA / Flow. A shared observed context and action strip sit above it. The reader can scrub the same recorded actions. A small current-frame/reconstruction inset is included only if it clarifies which nuisance has been removed.

### 2. Then show a representation failure

**Reader question:** What happens when the objective makes the wrong feature easiest to keep?

Change the relevant environment factor, show the real cube and actual visual nuisance, and then show the fitted representation's behavior. The final wording must follow the evidence: “the tested readout no longer recovers position” until stronger evidence justifies “drops position.” Decoder disappearance alone is never the proof. If a newly completed target-gradient intervention repairs or changes the effect, show the paired before/after result beside the scene rather than appending a separate audit section.

**Visual:** one observed frame, its frozen-real-latent reconstruction, and the model's own future, with flow shown on the identical context/actions where available. A tiny three-seed paired plot supplies the minimum quantitative support. Use original latent size; no amplified feature directions or resized latent rescue.

**Short analytical bridge:** the optional Gaussian state-plus-static-nuisance illustration. It explains why a globally noncollapsed marginal can favor predictable nuisance. Keep one-lag and single-snapshot assumptions explicit, label it analytical rather than experimental, credit the known Sobal shortcut, and do not claim it explains a high-loss neural optimization failure. It can be collapsed if the final mechanism differs.

### 3. Finally show where a usable representation stops supporting planning

**Reader question:** If current state is usable, what changes when the future branches?

Begin with evidence that the same fitted representation supports current-state readout or a qualified real-transition controller. Then contrast the actual stochastic futures, the model forecasts, and real-environment execution by the declared JEPA controller and flow-plus-PPO baseline. The positive flow outcome and negative JEPA outcome must be for the selected matched condition, not assembled from different scenes, checkpoints, objectives or best policy seeds.

**Visual A:** common observed context and recorded actions, followed by GT / JEPA / Flow futures. If multiple real continuations from one state/action sequence exist, show the spread; otherwise do not imply that a single realized future measures distribution quality. Sample versus sample-mean operators get distinct labels.

**Visual B:** short actual environment executions, labeled with each policy's own actions. This is separate from recorded-action model prediction. Beside the videos show one effect estimate or compact per-model dots, not a large table. Failed oracle/control baselines, if material, appear in the same short sentence; a planning-failure headline cannot survive an unqualified common controller.

**Two compact mathematical points:**

1. For a fixed encoder and squared loss, the optimum is the conditional mean. For squared goal cost, expected cost is the cost of the mean plus conditional variance. A planner that scores only the mean can rank actions differently. This is an analytical counterexample, not an attribution to the measured run.
2. Nonlinear recursive prediction need not preserve expectations. Use exactly `x(t+1)=x(t)^2+xi(t)`, independent fair `xi(t) in {-1,+1}`, and `x(0)=0`. The one-step conditional mean is zero, but `E[x(2)]=1`; recursively applying the mean dynamics at zero gives zero. A small native SVG diagram can show these two calculations directly. No fabricated model trajectory is presented as experimental evidence.

The connection to the actual third case is earned only by the completed intervention/readout/control evidence. If that evidence establishes only a system difference, state the measured system difference. If it does not establish the third case at all, leave the analytic example as a boundary and do not manufacture an empirical failure mode.

## Closing paragraph

State the two or three findings actually supported by these cases. Connect their practical consequence to the distinction between ignoring nuisance and losing useful state or uncertainty. Avoid a repeated list of audits, generic taxonomy, an anti-JEPA slogan, or a promise of general intelligence.

## Collapsed methods appendix

One expandable “How these examples were measured” section holds checkpoint/data/source identities, probe supervision, seed hierarchy, controller protocol, exact selection rule and links to the full research reports. Historical gate tables and the 66-endpoint explorer are not in the main reader flow. The full historical drafts stay separately available.

Craftax nearby-goal results do not appear in the main article. A later Craftax section is admitted only for matched, verified inventory acquisition such as pickaxe/sword; it must not relabel local displacement as that task.

## Layout and interaction contract

- Keep the existing warm paper/ink palette and editorial typography.
- Desktop: article column ~720 px, scientific figure width ~1,000 px. Mobile: stacked model rows with a shared scrubber and no page overflow.
- One interaction per figure: time scrub or scene selection, not a dashboard of unrelated filters.
- Recorded-action prediction and policy execution are visually and verbally separate.
- All frame assets retain native environment colors. No synthetic restyling of the measured environments.
- Analytical diagrams have permanent “Analytical illustration” labels, static equation fallbacks and no shared numerical display with experiment scores.
- No remote font, chart, math or script dependencies. Native MathML/SVG and local assets.

## Theory check completed

The theory reviewer verified both mathematical bridges. State their scope as follows:

- Conditional on context/action, the goal must be fixed or context-measurable and the next latent square-integrable. The squared-loss optimum is the conditional mean; expected squared goal cost equals mean-goal cost plus the covariance trace. **Action rankings change only if the variance contribution changes the comparison.** An action-independent variance cost adds the same constant and cannot by itself reverse a one-step ranking.
- Use the user’s exact independent fair **±1** noise in the `x^2+noise` illustration. Every finite-horizon distribution has finite support. With x0=0, x1=xi0 and x2=xi0²+xi1, so E[x2]=1 while repeated application of g(x)=x² gives0. The action example is exactly A=±1 equiprobably versus B=0.1 and goal0: mean-only costs0/.01; expected costs1/.01.
- These facts establish analytical possibilities. They are not an empirical diagnosis of the selected red-cube run, and a failure of a distributional sampler does not invalidate the algebra.
