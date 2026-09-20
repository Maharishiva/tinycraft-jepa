# Redcube raw-results audit (2026-09-19)

Read-only audit of the original `studies/redcube` experiment family and `runs/redcube` artifacts in `/Users/rldoeswork/Documents/rl/tinycraft-jepa`. No training or GPU evaluations were run. This is evidence for carefully scoped failure modes, not a blanket negative result about JEPA, and not yet a matched objective-only comparison.

## Evidence verdict

The strongest reproducible existing redcube result is the interaction in the **grid JEPA's four gray-noise conditions**, for which three training seeds are saved: independent per-frame gray noise plus stochastic wind produces poor position readout on real encoded observations; removing wind or making noise persistent preserves position readout. Separately, the class-token LeWM implementation exhibits a different failure on the easiest black/no-wind task: position is accurately recoverable from the real latent with whitening, but a probe trained on real latents fails on predicted latents. The latter is a predictor/readout-distribution mismatch, not evidence that the encoder erased position. Generative flow also has a substantial failure on the persistent color/no-wind condition; it must appear in any honest summary.

## Canonical result sources and machine-readable extraction

Use `runs/redcube/<condition>/allarms/bench.json` as the single source for a full architecture table. The top-level `bench.json` is a separate re-benchmark of only `tile_factored` and `flow`. `bench_allarms.json` is an older snapshot and lacks the modern `probe_error` and `pixel_probe` sections. Do not splice values from those versions without identifying this.

- `/tmp/tinycraft-redcube-summary.json` contains all **109 model-condition rows** from allarms/bench.json, including result-file SHA256, source path, all key position metrics, train seconds, logged final loss/step count, AF, gFVD, ridge R2 and episode 1-NN.
- `/tmp/tinycraft-redcube-table.md` contains a complete numerical table suitable for supplemental material.
- No checkpoint SHA256, checkpoint/source revision, actual sampled evaluation indices, or per-window residuals are recorded in the benchmark JSON. Result-file hashes establish the artifact read by this audit, not original model provenance.

Minor top-level/allarms rerun differences exist (including flow AF and probe errors). Example gray_frame_wind flow pixel H8 is 12.284 in top-level vs 12.302 in allarms. Some stochastic/recomputed probes differ slightly. Canonicalizing one source avoids false precision and mixed tables.

## Main numerical table

All values below are from `allarms/bench.json`. L1 error is |x error| + |y error| in pixels. `real` means probe error over all held-out eval frames, not rollout error. `pred8` applies the same real-latent-trained probe to an 8-step predicted latent. Flow pixel8 uses the shared pixel CNN on decoded flow predictions and is deliberately labeled separately.

| Background | Wind | Grid real | Grid pred8 | LeWM6 real | LeWM6 pred8 | Flow real latent | Flow pred8 latent | Flow pixel8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Black | off | 0.097 | 3.933 | 0.087 | 45.268 | 0.140 | 0.126 | 0.242 |
| Black | on | 0.126 | 11.732 | 0.049 | 8.957 | 0.153 | 11.632 | 11.622 |
| Color, frame | off | 0.172 | 5.570 | 24.133 | 19.428 | 1.088 | 1.059 | 0.395 |
| Color, frame | on | 0.253 | 11.823 | 25.185 | 19.317 | 1.344 | 12.311 | 11.985 |
| Color, episode | off | 3.273 | 6.594 | 20.804 | 21.567 | 18.563 | 21.954 | 22.423 |
| Color, episode | on | 1.451 | 13.342 | 23.431 | 21.807 | 7.010 | 18.960 | 16.937 |
| Gray, frame | off | 0.115 | 4.269 | 23.954 | 20.286 | 3.114 | 2.667 | 0.382 |
| Gray, frame | on | 21.354 | 19.696 | 25.288 | 19.567 | 3.690 | 14.392 | 12.302 |
| Gray, episode | off | 0.715 | 1.779 | 21.316 | 21.284 | 5.062 | 4.896 | 1.339 |
| Gray, episode | on | 0.717 | 11.553 | 23.880 | 23.386 | 3.066 | 15.150 | 13.583 |

Use the JSON extraction if more precision or recomputation is needed. Most individual differences in this table are one training seed and lack confidence intervals.

### Three-seed gray-noise evidence

Means ± sample standard deviations across saved grid `tile_factored`, `tile_factored_s1`, `tile_factored_s2`; these are three model initializations on the same dataset, not three independent datasets. The parent script sets seed 0 in COMMON and appends the arm-specific --seed 1/2; argparse uses the final occurrence.

| Condition | Real latent held-out L1 | Predicted latent H8 L1 | Last logged prediction losses, seeds 0 / 1 / 2 |
|---|---:|---:|---|
| gray_frame_wind | 21.3295 ± 0.0571 | 20.3878 ± 0.9248 | 0.7345 / 0.7336 / 0.7331 |
| gray_frame_still | 0.1151 ± 0.0115 | 4.8443 ± 1.1342 | 0.0007 / 0.0008 / 0.0005 |
| gray_episode_wind | 0.6676 ± 0.0486 | 10.9516 ± 0.5346 | 0.0031 / 0.0030 / 0.0028 |
| gray_episode_still | 0.7637 ± 0.0680 | 1.8293 ± 0.2196 | 0.0012 / 0.0016 / 0.0013 |

All use 8500 training updates. Data shape in training logs is (5001,100,63,63,3), i.e. 500,000 transitions. This is not simply a low-prediction-loss degenerate solution: the failing grid condition has high persistent prediction loss. That matters to the causal story and must be distinguished from low-loss nuisance solutions in some class-token/static-texture runs.

## What each diagnostic establishes

1. **Real-latent nonlinear held-out probe**: train an observation-to-position readout on training latents and test on independent eval frames. A low error establishes accessible position information. A high error establishes failure of this probe, not an information-theoretic absence claim. Implemented at `studies/redcube/bench.py:104-166`, `:233-246`, `:479-486`.
2. **Same probe on predicted and encoded true future latents**: separates readout on real observations from forecast/readout mismatch (`bench.py:472-486`). An accurate true-latent floor with bad predicted readout supports a forecast failure under this readout. It can also reflect predicted latents being off-manifold: a probe trained only on true latents may extrapolate badly. It does not establish no other readout could decode those predictions.
3. **Shared pixel CNN on decoded context and predictions**: context error diagnoses the combined encoder/decoder/readout bottleneck; rollout error adds dynamics. It is especially useful for rejecting misleading decoder-only visuals. The CNN is trained with synthetic background replacement using ground-truth cube masks (`bench.py:189-204`); this is a supervised diagnostic, not the training objective or an agent policy. Report this privileged annotation explicitly.
4. **Linear ridge R2**: an older supplementary probe fit on the first 80% of eval time rows and tested on later rows (`bench.py:286-309`, `:465-468`). It is not the same protocol as the train/eval-separated nonlinear probe, and cannot be used to claim position is absent. Example grid off_still ridge R2 is only about .24 while the nonlinear probe reaches .097 px. `r2_wind=1` in no-wind conditions is a degenerate zero-target result, not evidence for learning wind.
5. **Episode 1-NN identity**: evidence of persistent episode-specific information, consistent with texture encoding; does not mean 'only texture is encoded.' Split identities overlap across the time split. All successful persistent-texture representations often achieve the same .703 (still) or .29425 (wind), apparently limited by eval time-split identity overlap. The reported `chance_episode=1/number of fit identities` is not a calibrated baseline with unseen eval identities (`bench.py:291-309`).
6. **Redness argmax / mass**: simple interpretable image metrics but imperfect on color noise. The cube detector is handcrafted for the environment and should remain explicitly labeled diagnostic. Location based on learned pixel readout was added to avoid interpreting background redness as cube motion (`bench.py:51-79`).
7. **gFVD**: Fréchet distance in a fixed randomly initialized CNN feature space (`eval.py:88-128`), not pretrained I3D FVD and not a task-state metric. The background occupies most pixels, so score improvements can reflect texture quality rather than better object-state prediction. AF ranks salience-weighted pixel MSE with true versus random actions (`eval.py:131-140`); per-frame noise makes almost all pixels salient. Neither alone validates control or physical accuracy.

## Important mechanism distinctions

### Gray per-frame noise plus wind

Grid seed0 held-out real-latent readout is 21.354 px, true-latent H8 is 20.496 px, and predicted H8 19.696 px. The true representation readout is already bad. Three seeds repeat this, and linear position R2 is negative. Current evidence supports robust failure to expose position through the tested readouts under this recipe and budget. It does not distinguish failure of optimization from a global optimum that suppresses position, and final prediction loss near .733 counsels against calling this a successfully optimized low-loss shortcut.

### Black background without wind

LeWM6 has held-out real-latent error .087 px and encoded true-future H8 error .022 px, while predicted H1/H8/H16 are 44.302/45.268/46.463 px under the real-latent-trained whitened probe. Pixel context reconstruction is already poor (17.015 px); pixel H8 is 18.415 px. Thus the precise position is accessible but poorly served by decoder and predictor. Whitening sensitivity is direct evidence for poor conditioning. Separate diag logs show standardized MLP 18.53 px vs whitened .71 px for LeWM6, and 19.56 vs .68 for tile-class-token AR.

The rounded-prototype count displayed in the latent visualization (193 for LeWM6 off_still) is obtained by quantizing the first whitened dimensions. It is not an exact number of distinct latent vectors; avoid 'collapses to exactly 193 states.' See `/tmp/tinycraft-redcube-diag.md` for detailed source anchors. That audit also finds diagnostic-source/log drift (current `diag_cls.py` prints a raw-MLP field absent from saved logs), and its old standardization uses fit plus held-out feature statistics. Treat those old numbers as supporting exploratory diagnostics. The modern benchmark separately fits normalization using its training split, so the old diagnostic leakage should not be attributed to the main benchmark.

### Architecture matters substantially

Under black/no-wind, tile AR, tile AR history8/no-dropout, and tile AR no-dropout achieve H8 errors .270/.307/.253 px, much better than grid factored 3.933 px and class-token variants 26–45 px. Under color_frame_still, tile_ar8 succeeds (.651 px H8) while tile_ar and tile_ar_nodrop fail (~19–20 px). This defeats a broad claim that the objective always fails under independent noise, and shows history, dropout, and encoder/predictor organization cannot be collapsed into one explanation. The history8/no-dropout change bundles two factors relative to tile_ar, though tile_ar_nodrop helps partly separate them.

### Generative flow is not uniformly successful

Color_episode_still flow representation error 18.563 px, decoded-context error 18.403 px and pixel H8 error 22.423 px indicate a representation/reconstruction failure before dynamics. Grid JEPA retains substantially more usable position there (real 3.273 px, latent H8 6.594). Include this adverse result. Flow has low pixel error in deterministic frame-noise conditions; however its latent position probe is worse than its decoded pixel probe in several conditions, which further shows why probe architecture/readout is not neutral.

## Design, data, fairness, and inference limitations

- Environment: one red 6x6 object on 63x63 RGB, four cardinal actions of 3px, stochastic hidden integer wind in [-2,2], uniform iid pixel background redrawn per frame or held per episode, red boundary, max128 steps (`studies/redcube/env.py:26-126`). Original redcube has no internal two-wall gap layout. It is a controlled diagnostic, not evidence of physical general intelligence.
- Black/color datasets have 100,000 transitions; gray datasets have 500,000 (`runs/redcube/*/data.log`). Compare wind/static/dynamic gray conditions within their matched data size. Do not interpret differences between gray and color as color-noise effects alone.
- Only grid JEPA's four gray conditions have three saved training seeds. Flow and LeWM6 evidence is one seed per condition; full 12-layer LeWM is saved only for gray_frame_still. Seed0 labels for other variants are based on launcher defaults, not newly inspected checkpoint metadata.
- Current benchmark defaults are 2048 windows for gFVD/AF, first 512 for cube/probe trajectories, context8 and horizon16 (`bench.py:400-403`, `:439`; `eval.py:19`). These are code-derived defaults, not embedded JSON provenance; exact original CLI must be recovered for a formal claim.
- Windows are sampled **with replacement**, overlap, and are conditioned on no reset/death across 24 frames (`eval.py:58-76`). They are not 2048 independent episodes. Survival filtering strongly changes the wind sample population: only about 37% of eval wind episodes reach length25, versus 82% without wind (data logs). State error on these windows does not establish death prediction or closed-loop control.
- `oracle.json` uses every `T.valid_starts` window: 4864 wind / 16376 still (`oracle.py:29-41`). Primary metric sample uses 512 selected windows. Thus the visual oracle reference is not evaluated on exactly the same window set. Held-last-wind extrapolation is an interpretable reference, not a Bayes-optimal lower bound. Wind H8 error is 8.874 for this reference; persistence13.647 and action-only12.352. No-wind action-only is exact0.
- Training is approximately a ten-minute **wall time recipe**, not exact equal compute: reported main grid train times 576–619s, flow 533–710s; one LeWM6 run636s. Calibration excluded/off-budget supervised decoders and heads add extra work. Gray_frame_still flow709.9s vs grid609.1s is material. Report actual times and number of steps.
- Flow and JEPA use different representation learning, optimization, decoder, target, predictor/inference procedures. Matching latent dimensions and parts of the architecture does not isolate deterministic vs stochastic prediction objectives. Fixed reconstruction target vs jointly learned representation also matters. Describe this as comparing recipes with ablations, not an objective-only causal experiment.
- `vit_factored` uses 81x32=2592 latent scalars, tile grid512, class token192 or512. Large latent-capacity changes are exploratory architecture ablations, not a clean main comparison. Avoid introducing them into the headline as a fair objective baseline; the user explicitly opposes arbitrary latent inflation.
- Existing plots of scenes select the windows with greatest object travel, not random examples (`bench.py:501-518`). This is fine if disclosed; include a fixed random sample or selection-independent montage for a paper.
- Probe preprocessing was developed after inspecting observed failures. It improves the diagnosis, but makes confirmatory new seeds/eval split valuable. Probe training uses GT positions and a shared spatial inductive bias; do not call it entirely task-agnostic.

## Efficient follow-up priorities

1. Export durable per-window residuals, episode IDs, seeds, checkpoint hashes, dataset hashes and exact code/CLI. Recompute oracle references on the exact same windows. Bootstrap by episode, not raw overlapping windows.
2. Repeat LeWM6 black/no-wind and flow color-episode/no-wind with independent training seeds, because these counterintuitive failures currently carry substantial explanatory weight from one run.
3. Fit a separate identically budgeted readout on **predicted** latents with train-only data, and evaluate on independent eval rollouts. If it recovers position where the real-latent probe fails, the diagnosis is alignment/conditioning rather than destroyed predictive information. Add low-dimensional covariance spectrum and train-vs-predicted projection residuals.
4. For the core gray interaction, keep the existing model architecture and 512-scalar latent fixed. Test any objective modification under the same split, update budget, seeds and diagnostic probes. Do not change latent capacity to obtain success.
5. A simple wider budget check on the failed grid condition would distinguish a short-budget optimization failure from a durable recipe failure; it still would not prove global optimality.

A defensible manuscript sentence: “Across three initializations, the spatial JEPA recipe exposed position with subpixel error when either dynamics were deterministic or gray noise was persistent. Combining per-frame gray noise with stochastic wind raised held-out position readout error to 21.33±0.06 pixels. Other architectures failed through a different mechanism: in a deterministic black-background task, LeWM6 retained accurately decodable position in true latents but its predicted latents were poorly aligned with the same readout.”
