# Supervision and interface differences in the native-inventory comparison

Verified from the frozen checkpoints and their current matching source bytes. This is a comparison of two frozen model recipes with a common downstream policy learner, not an isolated loss-function ablation.

| Property | Frozen Flow world model | Frozen JEPA world model |
|---|---|---|
| Model training seed | 0 | 0 |
| Training data | Same 500,000-transition archive, SHA-256 09f2cb41… | Same archive |
| Observation/dynamics history | Temporal two-frame codec; native 32-slot dynamics window | Single-frame tile encoder; native 8-frame dynamics history |
| Model objective | Generative latent dynamics plus learned reward/done losses and the checkpoint's other declared losses | Joint embedding prediction MSE + 0.09 SIGReg; no reward/done objective |
| Reward/done supervision | Joint with the shared dynamics trunk during WM fitting | Separate readouts fitted after freezing the encoder and predictor |
| Current readout fit | Existing native checkpoint heads | Fresh decoder 8,000 updates and reward/done heads 6,000 updates over every training transition; no value head |
| Policy interface | Generated RGB, native learned reward/done | Decoded predicted native 512-coordinate latent, native learned reward/done |
| Policy architecture/budget | Same 6,092,177-parameter no-GRU pixel actor and 4,998,400 imagined transitions | Identical |

The actual Flow metadata has `reward_weight=0.1`, `done_weight=0.1`, `head_trunk=0.1`, `head_warm=0.5`, and 16,000 WM updates. Source `studies/revision/inventory_source/wm_generic.py:414` implements strength 1.0 while `step < 8000`, then strength 0.1. Thus head-loss gradients reach the shared dynamics trunk at full strength in the first half and are reduced in the second half; there is no initial zero-gradient phase. The 0.1 reward/done coefficients additionally scale the respective losses. Rewards use two-hot symlog cross-entropy; termination uses binary cross-entropy. The done path also applies this gradient scaling to action embeddings. See lines 59,169–175,409–429.

The Flow tokenizer is learned first, after which embeddings are cached and normalized by their training mean and standard deviation. Only the dynamics module is optimized during the WM phase (`wm_generic.py:434–459`). This intrinsic preprocessing is preserved exactly as part of the existing checkpoint; no new latent transform is introduced for either arm.

The current JEPA readout fitting uses the entire 500k transition label bank with current-frame/action alignment. Its base encoder and predictor arrays are verified byte-identical before and after fitting. Sharing the same label archive therefore provides equal label availability, **not identical gradient paths, objectives, head architectures, or training compute**. A policy-transfer gap cannot by itself be attributed solely to the JEPA prediction loss.

Both policy families receive the same corrected genuine eight-frame episode-start prefix bank and actual seven outgoing context actions. Their native history representations remain different. The actor starts a new control episode with previous_action=NOOP and sees raw final-prefix RGB initially, decoded model predictions during imagination, and raw native RGB from ordinary fresh resets during final evaluation. Policy starts may already contain resources acquired during the recorded prefix; no inventory/success filtering is applied. All native final achievements are measured afresh in ordinary evaluation episodes.
