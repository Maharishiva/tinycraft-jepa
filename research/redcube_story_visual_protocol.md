# Fixed-window red-cube visual evidence

This is an illustrative export from frozen checkpoints, not a new control experiment or a quantitative sample of prediction quality.

## Selection fixed before model loading

For each of `gray_frame_still`, `gray_frame_wind`, `gray_episode_wind`, and `gate`, enumerate `(start_time, environment_index)` lexicographically with `start_time ≥ 1`, retaining the recorded incoming action at `start_time - 1`. Retain the first qualifying window from each distinct `(environment_index, episode_id)` until three are selected. A qualifying window has 8 context and 16 future frames with no episode reset. For gate, a wall must already exist in the last context frame. The continuity condition selects future-surviving windows; it is disclosed and must not become a success estimate. Selection uses neither movement amount, RGB appearance, nor any model output.

Pinned IDs are still `[[1,0],[1,2],[1,3]]`; frame-wind and episode-wind both `[[1,1],[1,2],[1,6]]`; gate `[[15,12],[15,21],[15,24]]`. Equal IDs do not imply equal observations across conditions.

`selection.json` and `inputs.npz` were written before any model forward pass. They contain dataset and checkpoint SHA-256, archive collection metadata, exact time/environment indices, recorded outgoing actions, reset masks, and recorded state labels. The dataset archives do not store every simulator transition key; the export does not invent these missing keys.

## Frame and action contract

The five rows are recorded RGB future; reconstruction of each actual future through the frozen JEPA encoder and fitted decoder; autoregressive JEPA forecast decoded by that same decoder; reconstruction through the flow checkpoint's actual codec; and one sampled flow forecast. All forecast rows receive only the same 8-frame context and 16 recorded future actions. Reconstructions are diagnostic reference rows and are never fed back as future observations into a forecast.

JEPA is the original 512-dimensional `tile_factored` seed-0 checkpoint. Its posthoc decoder used 80,000 real training frames and 8,000 decoder updates; full stored decoder metadata is retained. A failed decoder is not proof that the representation contains no state information.

Flow uses the corrected reward-head recipe checkpoints in `~/personal/stoch/<condition>/flow/model.npz`. These are not the older `/dev/shm/redcube/<condition>/flow/model.npz` files used by some historical benchmarks. The figures must not silently identify the two checkpoint families. The export uses 8 flow evaluations per step and sample seed `20260920`, fixed in selection metadata. It shows one sample, with no best-of selection, and is not expected to match the archive's particular wind realization.

## Numerical semantics

The JEPA encoder and predictor use the original public rollout implementation with batch 64 and history 8, matching the frozen benchmark context shape. The three selected inputs are padded by repeating the last selected input. Every real future frame is encoded as the last frame of an 8-frame rolling history at the same batch shape. Public-rollout context and direct context encodings must agree bit for bit; export fails otherwise.

The temporal flow codec receives the actual preceding/current RGB pair when constructing the diagnostic real-future row. Forecasts start from the original 8-frame context. No future frame enters a forecast. All JEPA decoder calls use 512 latent rows, with repeated-row padding where necessary. Latents retain checkpoint values and are saved in float32; conversion from BF16 to float32 is exact. No scaling, whitening, probe overlay, manual recoloring, environment change, or new model fitting is used.

## Deliverables and checks

`frames.npz` preserves original RGB arrays, real and predicted latents, actions, indices, state labels, masks, and flow PRNG keys. `case_manifest.json` records exact hashes, stored model metadata, code hashes, numerical batch shapes, array hashes, dependency versions, and context parity. `browser_packet.json` maps each selected scene to exact native-resolution PNGs and a five-column filmstrip. The HTML may hide a diagnostic row behind a control but must preserve its identity and availability.

Selection ordering, distinct episodes, reset exclusion, and the current-context wall predicate were checked with synthetic metadata before export. The original inputs and checkpoints remain read-only. All three selected scenes are preserved, whether favorable or unfavorable.
