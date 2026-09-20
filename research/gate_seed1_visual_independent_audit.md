# Independent gate seed-1 visual audit

**Pass, with the observed flow replay difference preserved.** This extends the earlier four-packet audit to `artifacts/revision/redcube_story/gate_s1`.

The JEPA checkpoint was independently rehashed on `multigpu-vm` and its NPZ metadata read without loading the model. SHA256 `7190a8c231596043b62b6750f807e1c9fa542287dc161b7ca72182f2e68e625a` matches the manifest and identifies model seed 1, tile/factored architecture, 16 × 32 native latent coordinates, history 8. The flow checkpoint remains seed 0.

The input packet is byte-identical to the original gate packet; selected IDs remain `(15,12), (15,21), (15,24)`. Actions, frame times, states, validity/reset masks, flow keys, and flow checkpoint are unchanged. Both prepare and export hashes match the retained frozen `export_source.py`; inference source-module hashes match the original gate export. The separate kernel receipt documents the omitted dependency without rewriting original manifests.

Independent byte checks pass for all 27 exported-array hashes, 264 native PNGs, 15 scene-metadata comparisons, and three complete filmstrips, including all 75 image cells and annotations. The five row identities are correct. Browser metadata and visible footers identify JEPA seed 1, flow seed 0, eight flow evaluations per step, and candidate starts at t ≥ 1. Scene 0 was also visually inspected.

Comparing the actual packets confirms **20 non-JEPA arrays are identical**. The two differing flow arrays are:

| Array | Different values | Maximum absolute difference | Mean absolute difference |
|---|---:|---:|---:|
| Flow future RGB | 4 / 571,536 | 1 intensity unit | 0.000006998684247 |
| Flow predicted latents | 6,040 / 24,576 | 0.03125 | 0.000998810377 |

These numbers reproduce the exporter's report independently. The flow checkpoint, input packet, sampling keys, NFE and source-module hashes match; the two executions are still not bit-identical. The audit establishes no causal source of this numerical difference. Both original results remain intact; no array was replaced to force equality.

Packet SHA256: `856139a0492aa3aa273370f3822246fd0d5b2dc3acb8e5bb4c40215e791235fa`. Current annotation renderer SHA256: `9c4e3ad2ee367640d58632f5160e31e193f3c2891a2feb323e72dcd4e87f143e`.

This audit ran no model forward pass or GPU job and changed no production artifact. Future-survival conditioning and the distinction between diagnostic reconstruction and forecast remain necessary when interpreting these illustrations.
