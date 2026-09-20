# Assets needed for the three selected red-cube cases

The author/root selects cases from verified evidence. HTML does not select favorable outcomes. Existing archived assets can be used if their identity and row contract are known; otherwise export from frozen checkpoints without retraining or altering the environment.

## Minimal handoff per case

| Field | Required information |
|---|---|
| Claim | Exact supported sentence, including whether this is prediction, representation accessibility, or real control. |
| Environment | Actual condition name; background color/type and redraw schedule; wind; wall geometry if present; renderer version. |
| Models | JEPA and flow checkpoint paths + SHA-256; architecture/latent shape; model seeds; frozen decoder identities and fitting scope. |
| Shared input | A saved evaluation episode/context ID, RGB context, recorded outgoing actions, reset mask, data archive hash. |
| Real future | Original RGB frames and frame/episode time indices; position/state labels may be retained in provenance, not fed to a controller. |
| JEPA future | Decoded original predicted latents for those exact inputs; native shape/units; predictor variant and horizon. |
| Flow future | One explicitly seeded sample for the same inputs; NFE; decoded-real reconstruction if used. Sampling does not have to match GT's particular noise realization. |
| Representation panel | Decoded real encodings and the smallest verified readout/intervention result needed for the claim, with probe training domain. A real-trained decoder alone cannot establish erasure. |
| Control panel | Actual environment RGB trajectories from final saved policies; each policy's own action sequence and outcomes; common evaluation episode/noise keys where declared. This is distinct from model rollout under recorded actions. |
| Main scalar | One predeclared primary metric, per-model seed values and policy/evaluation counts. Never use best checkpoint or best policy as aggregate evidence. |
| Selection | A rule fixed independently of which model looks better. Prefer the first three eligible evaluation episodes or an explicit metadata-only rule; preserve all selected examples. |

## Suggested file schema

One `case_manifest.json` plus one NPZ or exported PNG sprite set per case:

```
case_id
claim_status: verified | pending
claim_text
source_paths_and_sha256
model_checkpoint_sha256
scene_ids
selection_rule
context: uint8 [scene, context_time, H, W, 3]
actions: int [scene, future_time]
real_future: uint8 [scene, future_time, H, W, 3]
jepa_encoded_real: uint8 [scene, future_time, H, W, 3]  # optional diagnostic
jepa_future: uint8 [scene, future_time, H, W, 3]
flow_encoded_real: uint8 [scene, future_time, H, W, 3]  # optional diagnostic
flow_future: uint8 [scene, future_time, H, W, 3]
valid_mask / episode_reset_mask
controller_videos: separate assets + own actions/outcomes
```

This schema is a display contract, not a demand to create new evaluations. Existing exact exports may have different keys; provide the mapping. Do not infer missing actions or provenance from a GIF.

## Already available / pending inventory

- The original redcube blog has per-condition GIFs under `blog/redcube/assets/`; their row contract and selected-window provenance require inspection before reuse.
- Current V1 gate latent-check NPZs have exact context/GT/decoded-real/JEPA/flow frames but are not automatically evidence for a selected red-cube case.
- Existing `allarms/bench.json` provides prediction/readout metrics, not a matched policy execution panel and not an information-theoretic absence test.
- Existing Gaussian analytical component can be copied from the archived V2 without changing that archive; formulas were independently verified. Mean/variance and `x^2+noise` diagrams need theory-review verification before publication.
- Native environment rendering and original colors are required. No ImageGen scene substitutes, recoloring, or invented model forecasts.

## Publication boundary

No final claim text is filled until the author hands off the three-case evidence. If a requested case is not supported, use the strongest truthful statement or omit the case rather than making the visualization imply it.

## Confirmed local inventory (read-only scout)

- All ten conditions have `runs/redcube/<condition>/scene{0,1,2}.gif`, corresponding MP4/strips, and allarms/probe variants. Main three-column GIFs are 804×322; four-column probe-arm GIFs are1068×322. They show8 context frames (model reconstruction) followed by16 forecast frames.
- These GIFs provide matched GT/JEPA/flow prediction views. They do **not** provide a separate decoded-real-future row. The original source code selects high-travel windows; this must be disclosed unless a fresh fixed-index export is substituted.
- Full local `runs/redcube` NPZ inventory comprises27 loss-log archives, not raw model-frame packets. No local `probe_scene_raw.npz`, model binaries, or data evaluation archive was found there.
- Single-model decoder studies exist in the original checkout at `output/decoder-study/off_still_lewm6/visuals.npz` and `output/decoder-study/color_episode_still_flow/visuals.npz`. Each stores GT `[3,24,63,63,3]` and decoder variants' actual-real-future and predicted-future frames `[3,16,63,63,3]`, plus positions/readouts. They do not store actions. They are not a matched JEPA/flow packet for one condition.
- The decoder-study `cache.npz` format includes indices/actions in its generating code, but that cache is not local. Fetching it or another existing remote raw export is preferable to recollecting data.
- Do not assume sibling `scene*_frames/` matches a GIF: `color_frame_wind/scene0_frames/000.png` is1596×322 (six columns), while `scene0.gif` is804×322 (three columns). The frame directory is stale relative to the current GIF. Inspect/use the actual selected artifact, and record its hash.
- Checked GIF/strip/probe/allarms files are byte-identical between the current worktree and original checkout. This establishes exported-asset identity, not the missing checkpoint/action provenance.
