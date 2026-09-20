# Independent audit of the completed matched primary experiment

**PASS.** All 45 planned policies completed 150 updates (1,228,800 nominal transitions). Their final policy binary hashes, evaluation metadata, scalar metrics and within-block training starts were independently checked against raw files. All 15 three-arm seed blocks share exactly the same start arrays. Evaluation uses seed 20260919 and 256 episodes per policy. Three distinct frozen model checkpoints and the approved harness hash were verified.

All 18 model/contrast/endpoint means were reproduced directly from episode arrays without importing the analyzer. Exact enumeration of all 3,125 paired-policy bootstrap resamples agrees with the declared 10,000-draw Monte Carlo intervals up to small sampling differences. The publication keeps its prospectively declared intervals. The machine audit records both.

| Model seed | Learned dynamics/heads | Exact transitions/learned heads | Exact transitions/true signals |
|---|---:|---:|---:|
| 0 | 0.285 walls; 0.938% survival | 0.230; 0.625% | 0.452; 0.469% |
| 1 | 0.217 walls; 0.781% survival | 0.068; 0.156% | 0.341; 0.391% |
| 2 | 0.320 walls; 0.391% survival | 0.096; 0.000% | 0.543; 1.406% |

Each cell averages five policy seeds for one fixed checkpoint. No arm produces a discovery: 0/45 policies exceed 50% survival. The complete exact pipeline increases mean walls by 0.166, 0.123 and 0.223 relative to the learned pipeline, but still does not learn reliable survival. The MLP interface and short-rollout policy optimization remain substantial limitations; these results do not support attributing failure solely to learned latent dynamics.

Replacing dynamics while retaining the fitted heads lowers mean walls in every model. That comparison includes the prospectively disclosed terminal-state extension, which carries appreciable continuation-weighted mass after true death. The separately authorized termination sensitivity will clarify that contribution; its future results must not overwrite the completed primary estimand.

Sources: `runs/revision/matched/`, `output/revision/matched-analysis/publication.json`, `research/matched_final_audit.json`. No training artifacts or primary analysis code were changed.
