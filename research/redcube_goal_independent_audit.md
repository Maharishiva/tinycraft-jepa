# Independent red-cube goal-control result audit

**PASS.** The auditor imports no evaluator, environment, JAX, or learned-model code.

Every native transition is reconstructed independently from the four three-pixel action vectors and the documented border. Prefix positions are recovered directly from the red square in raw RGB, excluding the red border.

| Model | Checks | Native executed transitions | JEPA H1 exact hits | JEPA H3 exact hits | Random exact hits | Oracle exact hits |
|---|---:|---:|---:|---:|---:|---:|
| m0 | 1107 | 12861 | 0/128 | 5/128 | 3/128 | 128/128 |
| m1 | 1107 | 12903 | 0/128 | 5/128 | 3/128 | 128/128 |
| m2 | 1107 | 12646 | 0/128 | 9/128 | 3/128 | 128/128 |

Verified all 1,024 prefix frames per model; every warmup action; safe start/goal geometry and 6–12-action lattice reachability; exact goal RNG draws; first-hit and first-death absorption; all success, death, distance and time summaries; Wilson intervals; visual episodes 0/1/2; terminal padding and forecast-valid masks; selected-plan first actions; observed shared nuisance pixels across jointly active visual trajectories; and recorded artifact/source hashes.

Across models, every cohort array and every raw random/fixed-cycle/oracle trajectory is bit-identical. Checkpoint hashes and training seeds are distinct; training configurations differ only in seed and name the same training-data digest.

The oracle reaches every goal in its exact shortest-path number of actions. This protocol measures empty-arena goal-directed control; an audited execution does not imply that JEPA achieved a successful positive control. The observed JEPA success counts must determine that interpretation. The task does not establish a requirement for deep planning.

Limitations:

- Fresh reset and renderer RNG keys are verified from the reviewed source; JAX PRNG streams are not replayed by this NumPy-only audit.
- The supplied model SHA identifies the launch checkpoint; this audit does not load checkpoint weights.
- Encoder batch shape is established by the hashed source and manifest, rather than runtime instrumentation.
- Decoded forecast pixels are illustrations; this audit verifies their policy selection and validity masks, not learned decoder fidelity.
