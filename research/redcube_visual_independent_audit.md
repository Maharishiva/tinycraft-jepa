# Independent red-cube visual packet audit

**Pass with two documentation/provenance caveats.** No action-alignment, row-identity, pixel-integrity, checkpoint-identity, or future-observation leakage defect was found in the four original packets. Scope is `gray_frame_still`, `gray_frame_wind`, `gray_episode_wind`, and `gate`, all with JEPA and flow model seed 0. The newly appearing `gate_s1` packet is outside this audit.

## Independently verified evidence

Read-only NumPy checks on `multigpu-vm` independently rehashed all 12 original evaluation archives and checkpoint files; every hash matched its manifest. All 50 selected input-array hashes were recomputed directly from the original evaluation archives and matched. Full-archive metadata enumeration reproduced the selected distinct-episode windows and eligible counts:

| Case | First three eligible windows, start ≥ 1 | Eligible windows |
|---|---|---:|
| Gray frame, still | (1,0), (1,2), (1,3) | 16,287 |
| Gray frame, wind | (1,1), (1,2), (1,6) | 4,827 |
| Gray episode, wind | (1,1), (1,2), (1,6) | 4,827 |
| Gate | (15,12), (15,21), (15,24) | 738 |

The independent byte subaudit checked 102 exported-array hashes, 50 input-array hashes, 16 packet-hash crossreferences, 46 mirrored input arrays, and 60 scene-metadata comparisons. **All 1,056 native PNGs exactly match their referenced arrays.** All 12 filmstrips were reconstructed pixel-exactly, including their 300 image cells and annotations. There were no missing, extra, or duplicated image references. Prepare/export source snapshots and the separate annotation renderer match their declared hashes.

The five rows are correctly distinguished: recorded future; JEPA reconstruction of actual future frames; JEPA forecast; flow reconstruction of actual future frames; one flow forecast sample. Future actions begin at outgoing action index 7 and contain 16 transitions. Incoming actions shifted by one equal outgoing actions. Recorded positions equal previous position plus recorded action displacement plus next recorded wind in every selected transition. Frames are consecutive, episodes are distinct, reset masks are clear, and the gate wall is already present in every last context frame.

Source inspection confirms that forecasts receive only the common eight-frame context and recorded future actions. Real-frame diagnostic reconstruction is separate; the temporal flow codec receives each actual preceding/current pair. Encoder calls use batch 64 and history 8; JEPA decoding uses 512 latent vectors per call. Padding repeats a selected scene or vector and applies no latent gain, normalization, or image recoloring. Stored direct-versus-rollout context difference is exactly zero in every manifest. This recorded numerical result was not independently rerun on a GPU.

Flow files were rehashed at the corrected `/home/rldoeswork/personal/stoch/<case>/flow/model.npz` paths, not historical redcube flow paths. Visual sampling is fixed at seed 20260920 and 8 evaluations per future step. Future-survival conditioning is explicit in selection, manifest, browser metadata, and protocol. Representative gate and dynamic-noise filmstrips were also visually inspected.

## Caveats to retain

1. **Specify start ≥ 1 in the selection description.** The source excludes archive frame 0 to retain an incoming action. The current prose only says first lexicographic window. Allowing start 0 selects different first three windows in the noisy cases: still `(0,0),(0,1),(0,2)`; both wind cases `(0,1),(0,2),(0,4)`. This is a deterministic eligibility-boundary omission, not evidence of selection by model outputs. Keep the frozen packets and clarify the description.
2. **Source provenance is not full dependency closure.** `source_modules` omits `flow_recipe/kernels.py`, which supplies inference attention, normalization, and activation operations. Listed hashes and retained source snapshots are valid, but do not claim every transitive inference dependency was hashed. No numerical mismatch was observed.

These illustrations cannot estimate unbiased prediction or policy performance, prove information-theoretic state erasure, or identify conditional averaging as the sole cause of forecast failures. The flow sample need not reproduce the archive's particular wind realization. Gate seed-0 figures must remain distinct from new seed-1 control results.

No fitting, model forward pass, GPU job, environment change, or production edit was performed during this audit. The machine-readable companion records exact scope and verified identities.
