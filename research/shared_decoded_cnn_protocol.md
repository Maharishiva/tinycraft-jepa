# Shared decoded-RGB control comparison

Prospectively fixed on2026-09-19 after the user's red-cube scope revision, before these new policy outcomes. This replaces further queued H48, raw-pixel and cold-start variants as the main control priority. Completed earlier experiments remain in the record.

## Fixed matrix

Four worlds: `gray_frame_still`, `gray_frame_wind`, `gray_episode_wind`, `gate`. For each, run the existing grid JEPA and actual TinyCraft Flow model with three policy seeds0,1,2:24 imagined-training policies. Additionally run three decoded-real gate policies per family:6 positive controls. All30 final outcomes are included.

Gray JEPA checkpoints are the existing `tile_factored` modelseed0 under `/dev/shm/redcube/<world>/model_probe.npz` (actual subdirectory `tile_factored/model_probe.npz`). Gate JEPA is preselected modelseed1, using prior development evidence that its ordinary decoded-real CNN could discover the strategy. This is favorable selection for JEPA, not an independent randomly chosen encoder. Every Flow checkpoint is the existing seed0 fit. No new policy is selected by performance, and policy seeds are not independent world-model fits.

Each policy is the unmodified original `studies.stoch.rl.AC(obs="pixels")`:1,702,980 parameters, native63×63 RGB, original Impala CNN and GRU. Both policies receive their own model's decoded posterior images on real observations, and decoded model predictions in imagination. Neither family gets raw real images as the actor input. The same decoder is used in burn-in, imagined interaction and final real evaluation. Raw real images remain inputs to the frozen codecs and are saved for trajectory displays.

The same optimizer, recurrent PPO updater, learning rate0.00025, four epochs, minibatch128,16-step backpropagation chunks, gamma0.99, GAE lambda0.9, value coefficient0.5, entropy coefficient0.01, clipping0.2 and gradient clipping1.0 apply to both families. Batch512 and horizon48 are fixed. The three nongate worlds use50 updates (1,228,800 scheduled positions and2,400 optimizer steps). **Every gate policy**, including both dream and decoded-real families, uses150 updates (3,686,400 positions and7,200 optimizer steps). There are no method-specific extensions.

## Initial observations and evaluation

Training starts are sampled uniformly from actual valid eight-frame archive histories. They exclude resets through the history and its incoming action; no repeated-frame reset starts are used. Policy burn-in processes the first seven decoded posterior frames with their real incoming actions. The final context observation is also decoded posterior RGB.

Flow's temporal codec uses the true preceding real image during live reconstruction. The first context frame uses its native beginning-of-sequence convention; later frames use their actual predecessor. JEPA's encoder remains framewise. Both codecs use fixed512-frame chunks with padding discarded before actor/dynamics access. No latent coordinate is resized, duplicated, amplified, whitened or otherwise changed. Native decoder quantization is retained.

Final evaluation uses256 fixed fresh reset episodes, each with seven uniform-random actions to obtain eight real observations. Admission conditions **only on survival of that past prefix**, never on subsequent policy performance. The same accepted prefix IDs and all action/environment keys are reused across families and policy seeds. The actor then controls121 steps under the original128-step cap. Report prefix attempts/exclusions and explicitly label scores conditional on prefix survival. Every policy uses its final checkpoint; there is no intermediate evaluation selection.

Save all per-episode outcomes plus native-RGB trajectories and actual policy actions for the first three accepted episode IDs. These IDs are fixed irrespective of outcome. They are real agent traces, not recorded-action forecast movies.

## Frozen models and task signals

Latents remain the native4×4×32 interface. Model encoders, dynamics and decoders stay frozen. Flow uses its native sampled dynamics at4 function evaluations and its original jointly fitted reward/termination heads. JEPA uses its deterministic predictor and the existing decoder; its original head architecture is refitted on **all500,000 available training transition labels**, with the original6000-update recipe and no reweighting or clipping. Decoder fitting remains the original80,000-frame posthoc fit. The new head fit never updates the encoder, predictor or decoder.

The decoded-real gate controls replace only the training transitions and task signals with the actual simulator and native rewards/termination; the actor still sees its own frozen decoded posterior images. They test whether this controller and observation interface can learn the gate strategy. They are not a parameter-matched comparison to raw-pixel or latent-only controls.

This is a **frozen model-recipe comparison**, not an isolated squared-loss ablation. Native model architecture, temporal receptive field, reconstruction training, head architecture and inference compute differ. All data/checkpoint identities and auxiliary training costs are disclosed. The equal500k label opportunity prevents the known80k-versus500k supervision mismatch from being presented as an objective effect.

Live metadata verification found the same native Flow settings in all four actual checkpoints: reward and termination loss weights0.1, `head_warm=0.5`, `head_trunk=0.1`, and16,000 dynamics updates. Those task losses send full-strength gradients into the dynamics trunk and action embeddings for the first8,000 steps, then scale those gradient paths by0.1. The tokenizer/decoder remain frozen after their9,000-step pixel reconstruction fit. JEPA world-model fitting is reward-free; its task labels only train posthoc heads with the model frozen. Equal available labels therefore do not equalize supervision stage or isolate stochastic sampling.

## Validation and admission

Before scientific runs, require independent source/CPU checks, full-checkpoint GPU codec parity (context versus real step,256/512 batches, and head-cache encoding), unchanged frozen state hashes, exact gate-state reconstruction and shared initial-context/evaluation fingerprints. Both families use the native Flow precision convention (`tensorfloat32`, non-Triton GEMM flag) explicitly, rather than inheriting a family-dependent global import side effect.

Use two-update computational smokes for each family and dream/real gate path. Whole-process timing includes loading, hashing, compilation and evaluation. Admit the full30 matrix using those timings and the remaining07:41UTC deadline; no score from a smoke determines admission. Run the six gate positive controls early, but retain and report all their outcomes and all24 planned dream policies. If the decoded-real controls fail, do not attribute imagined-policy failure to learned dynamics. If no clean retained-representation/stochastic-control third case emerges, say so rather than force the requested storyline.

Scientific code: `decoded_models.py`, `decoded_dream.py`, `warm_context_eval.py`, `shared_decoded_ppo.py`, `refit_full_label_heads.py`. Earlier scientific sources are not modified.
