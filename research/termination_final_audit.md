# Independent audit of the completed termination sensitivity

**PASS.** All 24 additional policies completed 150 updates. Their final policy binary hashes, raw episode endpoints, fixed evaluation metadata, model/data identities and source lineage were independently checked. All 3,600 training updates have finite required diagnostics and exactly zero continuation-weighted postterminal mass and reward. Each of the 24 new runs shares its exact saved start-index array with both corresponding frozen reference arms. The 72 parent policies were not retrained or re-audited.

Parent harness SHA256: `6a3460b2db2e47713b7af604454a4b04066c96bfd7748b5c991cfbc433176269`. Additional harness SHA256: `1c003dd4e019b2179981877310e131304d13a7150ac7da93e11d5164d011feb1`. After removing the approved extra harness entry, every dependency hash matches its reference.

## Final outcomes

`oracle_reward` uses exact transitions, true termination, and the original learned current-state/outgoing-action reward head. All cells are means over paired policy seeds for one fixed checkpoint.

| Policy torso | Model seed | Policies | oracle_reward walls | oracle_reward survival | True termination gain in walls | True reward gain in walls |
|---|---:|---:|---:|---:|---:|---:|
| mlp | 0 | 5 | 0.1641 | 0.625% | -0.0656 | +0.2875 |
| mlp | 1 | 5 | 0.2164 | 0.859% | +0.1484 | +0.1242 |
| mlp | 2 | 5 | 0.0781 | 0.156% | -0.0180 | +0.4648 |
| native_grid | 0 | 3 | 0.1016 | 0.651% | -0.3294 | +0.5143 |
| native_grid | 1 | 3 | 0.0000 | 0.000% | -0.7383 | +0.3672 |
| native_grid | 2 | 3 | 0.1810 | 0.651% | -0.0482 | +0.2708 |

The true-termination gain compares oracle_reward minus oracle_heads. It includes continuation weighting and removal of the after-death extension. The true-reward gain compares oracle_true minus oracle_reward, with exact transitions and true termination fixed.

No new policy exceeds the discovery threshold: **0/24**. True reward raises the mean wall count in all six fixed model/torso cells, but survival changes are heterogeneous and remain small. This supports a limitation of the fitted reward interface within the tested training protocol; it does not establish reliable control, broad reward-model causality, or superiority across model families. Correcting termination alone does not consistently improve control and lowers the mean wall count in all three native-grid cells.

All 24 model/contrast/endpoint means reproduce the publication JSON directly from paired episode arrays. The audit retains an exact finite-bootstrap cross-check alongside the prospectively declared 10,000-draw Monte Carlo intervals. The two policy torsos and three model checkpoints remain separate; intervals condition on each saved checkpoint and policy protocol.

This sensitivity was triggered by inspecting the prospectively monitored terminal diagnostics. It does not replace the completed three-arm primary experiment or retroactively change its design. Together, the separately reported H16 campaigns contain 96 policies and no discoveries; this is a descriptive count, not 96 independent model replications.

Sources: `runs/revision/termination-mlp`, `runs/revision/termination-native-grid`, `output/revision/termination-analysis/publication.json`, and `research/termination_final_audit.json`. No active scientific source, training artifact, policy or plot was edited.
