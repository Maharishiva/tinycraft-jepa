# Prospective target-gradient experiment

Status: design and implementation independently approved on 2026-09-19; staged for launch after the existing matched-PPO campaigns release GPU0. No target-gradient fits launched yet. This protocol fixes the scientific comparison before viewing any of its fitted-model outcomes. The launcher still makes its declared timing-only admission decision before full fits.

## Question and scope

Does removing the direct target-side gradient of squared latent prediction loss preserve more accessible position information, particularly when controlled dynamics contain fresh stochastic innovations?

This is a targeted mechanism test, not a proposed universal JEPA replacement. The comparison changes a gradient path while leaving the forward scalar objective, model architecture, data, SIGReg, initialization and optimizer schedule identical. Stop-target gradients do not make the encoder fixed: predictor-input gradients and all SIGReg gradients remain. Frames shared by predictor input and target retain their input-path gradients. Batch-normalization and overlapping history can couple frame contributions.

## Fixed matrix and training

Twelve fresh fits: environments `gray_frame_still` and `gray_frame_wind`; seeds 0, 1 and 2; modes `joint` and `stop_target`. The two environments share independent per-frame gray backgrounds. Their difference is wind dynamics. In the wind world, each wind coordinate receives a fresh increment -1, 0 or +1 with probabilities 0.1, 0.8 and 0.1, clipped to [-2,2]; conditional innovation variance is positive even if the previous wind is known. At the boundary the variance is 0.09, and away from clipping it is 0.2.

Use the existing 500,000-transition archives, without recollection or reward/trajectory filtering. Each mode in a world/seed pair uses the same initialization, minibatch indices, random projections and number of updates. Every fit uses exactly 8,500 updates, batch 128, AdamW learning rate 0.001, weight decay 0.001, 10% warm-up then cosine decay to zero, and global-gradient clipping at 1.0. The existing grid architecture is fixed: tile encoder width128/depth6/heads8, 16 tokens ×32 channels (512 coordinates), factored predictor width256/depth6/heads8, history8, and the original projector/predictor defaults. SIGReg remains weight0.09, 17 knots and1024 projections. No stop-gradient applies to SIGReg.

The sole mode difference is:

```
target = embeddings[:, num_preds:]
if mode == "stop_target":
    target = stop_gradient(target)
prediction_loss = mean((prediction - target) ** 2)
loss = prediction_loss + 0.09 * SIGReg(embeddings)
```

No reward, position, wall, wind, success, or privileged simulator label enters model training. Labels are used only for the separately fitted evaluation readouts below. There is no latent resizing, coordinate amplification, feature replication, reward modification, or objective selection based on downstream success.

Save the full per-update prediction/SIGReg loss trace and checkpoints at updates0,500,1000,2000,4000,4250 and8500. Checkpoint8500 is the sole primary endpoint; earlier checkpoints are descriptive trajectory evidence, not candidates for selection. Save configuration, input archive/source/checkpoint hashes, initialization identity and sampled-window sequence identity.

## Fixed readouts

Fit both of the following readouts for every model using the same deterministically sampled50,000 training frames and10,000 frames from the independently collected evaluation archive. Frame selection uses seed19417 and no outcome-based filtering. Archive metadata must confirm the different collection seeds and matching environment specification. Frozen encoders are evaluated in inference mode. These readouts introduce no gradient into the world model.

1. **Primary generic nonlinear readout:** raw512-coordinate latent →256 GELU →256 GELU →2 position outputs; 4,000 updates, batch1024, AdamW learning rate0.001, fixed initialization/minibatch seed0 for every fit. Position units may be divided by the known image width63 for optimization and converted back to pixels for all metrics. Latent coordinates are not rescaled, whitened or augmented.
2. **Secondary linear accessibility:** centered raw latent ridge regression with an unpenalized intercept and penalty0.001 ×trace(latent covariance)/512. This global penalty convention avoids an arbitrary absolute regularizer scale while leaving all latent coordinates unchanged. It is fixed prospectively, with no test-set selection.

The primary endpoint is final-checkpoint nonlinear-probe **mean absolute error per coordinate in pixels**. Report ridge MAE separately, and RMSE/R², per-coordinate errors, the training-position-mean baseline, latent variance and effective rank as supporting diagnostics. Save selected indices, raw per-frame predictions/errors, sample hashes and probe configuration. A failed readout is not proof that information is absent.

All12 final checkpoints receive both readouts regardless of outcome. Checkpoint4250 receives the same two readouts for all12 as a predeclared secondary learning-trajectory check only if a timing-only smoke projects at most five minutes per checkpoint for both readouts and the complete campaign including all midpoints fits the remaining deadline with a50% runtime margin; otherwise the secondary midpoint analysis is omitted for all12 and the final-only campaign is considered for admission. Make this timing decision before fitted-model readout outcomes. No selective probing of promising models is allowed. Checkpoint4250 belongs to the8500-step cosine schedule, not a separately optimized4250-step run.

## Primary comparison and interpretation

For each model seed, compute the stop-target benefit in each world:

`benefit(world, seed) = MAE(joint, world, seed) − MAE(stop_target, world, seed)`.

The primary interaction is `benefit(wind, seed) − benefit(still, seed)`. Show all three paired seed values, their mean and spread, along with all12 raw endpoint values and both within-world paired benefits. The interaction is a **relative effect**. It can be positive solely because stopping the target gradient harms the still world while doing nothing in the wind world; it therefore does not establish an absolute improvement or a rescue. Describe any absolute wind improvement from its own paired raw errors, separately from the interaction. A null, reversed, or harmful effect is reported as such. Do not replace the primary endpoint with whichever readout gives a favorable result.

Wind also changes state visitation, temporal complexity and episode lengths; the interaction is therefore not a unique causal proof of an innovation-variance mechanism. Matched mode comparisons within each world are cleaner than a raw cross-world score difference. Any empirical finding must be paired with this limitation and the distinction between an objective-level construction and finite-budget neural optimization.

No positive outcome is assumed. Removing target-side gradients could impede the nuisance compression needed to learn position while SIGReg continues to maintain nuisance variation; this is one theoretical possibility, not an empirical prediction. The EMA-free intervention tested here is not equivalent to V-JEPA2, TD-JEPA, or other teacher-based systems with different objectives and training recipes. A negative result remains part of the fixed experimental record rather than a reason to introduce another method under the same label.

Do not regard50,000 training frames or10,000 evaluation frames as independent model fits. Model initialization/data-order seeds are the replication units. Three seeds constrain the strength of statistical conclusions. Aggregate prediction loss is descriptive because the representation itself changes; low loss alone does not certify state preservation or control.

## Optional descriptive sensitivity

A natural-state sensitivity diagnostic can compare the same rendered background with cube position moved by one valid action-sized displacement against the same position with an independent nuisance background. Any such check must use the original renderer and valid states, report within-model state/noise distance ratios with both numerator and denominator, and never edit model inputs/latents during training or control. It is not the primary endpoint and will not be used to choose hyperparameters.

## Admission and validation

Before full launch: independently review the minimal copied loop; numerically verify `joint` one-step losses, parameter updates and batch statistics against the original trainer; verify the stop-target derivative on a toy objective while its forward loss stays equal; assert paired initialization/window/SIGReg RNG hashes; smoke checkpoint save/load and readout array alignment. Check data identities and independent evaluation seeds.

Completed independent validation: all six training checks and six probe checks pass on CPU; original and copied joint training give bit-for-bit equal losses and complete parameter/BatchNorm hashes through two updates including a nonzero optimizer step. Target stopping changes the intended derivative and final weights, while SIGReg still differentiates target embeddings. Checkpoint0 and final reload exactly. Readout tests verify ridge mathematics, metric units, fixed sampling, split separation, deterministic MLP fitting, frozen encoder parameters/statistics and invariance to tested encoding batch/padding changes. The queue's actual CLI was tested with mocked subprocesses: an expired deadline launches zero jobs, a timeout records `deadline_incomplete`, and a timestamp without an explicit UTC offset is rejected. The independent review is GREEN for the staged sources once GPU exclusivity is confirmed.

The proposed matrix costs about two GPU hours at existing grid-training timings, plus fixed readouts. A timing-only smoke must project every8500-step fit below20minutes and the full admitted work within the remaining deadline. If rejected on runtime, report the experiment as not run; do not shorten selected arms or pick successful partial checkpoints. Once admitted, run all12 fixed budgets. Unexpected numerical failures are retained and investigated without replacing seeds.

The queue pins all four input archives and core source files before admission. Every job has an explicit UTC deadline guard and subprocess timeout; a deadline stop is labeled incomplete and its partial artifacts retained. The two500-step full-architecture timing fixtures and their readout are stored under `smoke/` and never counted as registered8500-step results.

Implementation ownership: `studies/revision/target_gradient.py` (training), `studies/revision/target_gradient_probe.py` (readouts), and this protocol. These are new files and do not modify the frozen matched-PPO campaign sources.
