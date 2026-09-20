# Gate model-seed-1 visual export: completed and verified

The new control comparison uses JEPA model seed 1. Its image export completed in 84.4 seconds in a new directory, preserving the complete seed-0 packet. MG1 was released immediately to the queued goal-distance evaluation.

- Checkpoint: `/home/rldoeswork/personal/stoch/gate/tile_factored_s1/model_probe.npz`
- Checkpoint SHA-256: `7190a8c231596043b62b6750f807e1c9fa542287dc161b7ca72182f2e68e625a`
- The prepared input packet is byte-identical to seed 0: `8ce0a4f5c0397617a7832d5a4a0db79ee656d64787485cc12dd51f5a582a6d89`.
- All input-array hashes, selected IDs `[[15,12],[15,21],[15,24]]`, flow checkpoint, flow sampling seed `20260920`, and visual NFE `8` match the seed-0 export exactly.
- The original remote export source is unchanged; SHA-256 `7b33126b957b1ffc7ae213285b4649880bb17ece9bbec488326ac4a39cf20f2d`. It is preserved locally as `artifacts/revision/redcube_story/export_source.py`.
- Export packet SHA-256: `856139a0492aa3aa273370f3822246fd0d5b2dc3acb8e5bb4c40215e791235fa`. All 27 array hashes, 264 native PNGs, and exact-zero public/direct context encoding parity were verified.

Executed command, from the isolated remote checkout on `multigpu-vm`:

```sh
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/rldoeswork/personal/tinywmcraft-slim/.venv/bin/python \
  -m studies.revision.export_redcube_story export \
  --config gate \
  --jepa /home/rldoeswork/personal/stoch/gate/tile_factored_s1/model_probe.npz \
  --flow /home/rldoeswork/personal/stoch/gate/flow/model.npz \
  --flow-code studies/stoch/flow_recipe \
  --out artifacts/revision/redcube_story/gate_s1
```

The packet was fetched and rendered locally with the manuscript annotation palette and explicit model-seed/NFE footer. All three selected scenes and all sixteen future steps were visually reviewed. The real-frame JEPA decoder preserves the wall; forecast images often omit it, with a late fragment in the first example and cube loss in the third. This remains a learned forecast failure, not proof that conditional averaging explains the common wall’s disappearance.

The flow rerun is **not bit-identical** despite identical checkpoint, inputs, batching, and sampling keys. Four of 571,536 output RGB values differ by one intensity unit; flow predicted latents have maximum absolute difference 0.03125 and mean absolute difference 0.0009988104. All other non-JEPA arrays match seed 0 exactly. No causal source of the small numerical difference was established, and no output array was replaced. See `gate_seed1_visual_verification.json`.

## Reported annotation issue

The exact seed-0 `gate/scene0_filmstrip.png` at 29,094 bytes was opened and visually verified again: all five row labels and both footer lines are present in its 954×834 pixels. SHA-256 is `a9e4ba90a1792f953de15e9a27ef414e09e17db2994d416bbfdc8fa7bc132bde`. A byte-identical `scene0_filmstrip_labeled.png` copy provides a fresh display/cache key. No raw image or annotation repair was needed.
