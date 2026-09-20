# Independent audit of the completed native-grid sensitivity

**PASS.** All 27 planned final policies completed 150 updates and 1,228,800 nominal transitions. All policy binary hashes, three model checkpoint identities, nine three-arm start-array pairs, 256-episode evaluation metadata, and every scalar endpoint were independently verified from raw files. The frozen matched harness and native-grid policy source hashes match. The actor-critic has 583,236 parameters; this sensitivity uses the existing 4×4×32 latent topology.

All 18 model/contrast/endpoint means match the publication JSON. Exact enumeration of the 27 equally probable ordered bootstrap resamples, using inverse-CDF quantiles, reproduces every declared 10,000-draw interval endpoint. With only three policy seeds, these are coarse conditional intervals; they do not justify family-level inference.

| Model seed | Learned dynamics/heads | Exact transitions/learned heads | Exact transitions/true signals |
|---|---:|---:|---:|
| 0 | 0.000 walls; 0.000% survival | 0.431; 2.083% | 0.616; 1.172% |
| 1 | 0.284 walls; 0.651% survival | 0.738; 2.344% | 0.367; 0.521% |
| 2 | 0.202 walls; 0.000% survival | 0.229; 0.391% | 0.452; 0.651% |

Each cell averages three policy seeds for one fixed model. There are no discoveries among these 27 policies, including all nine trained using exact transitions and true signals. Changing the policy to respect native spatial topology does not remove the tested short-rollout optimization limitation.

Exact transitions with learned heads have positive model-level mean wall differences under this torso, unlike the MLP primary result. The effect therefore depends on the policy class and stated terminal extension. Keep the two campaigns separate; do not pool their conditional intervals or interpret their combined policy count as independent models. Across both separately reported campaigns, none of 72 policies crosses the discovery threshold, including 24 exact-transition/true-signal policies.

Sources: `runs/revision/matched_grid/`, `output/revision/native-grid-analysis/publication.json`, `research/native_grid_final_audit.json`. No run artifacts, analysis code or published primary estimand were changed.
