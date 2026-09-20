# Conditional H48 training-package sensitivity

H48 status: **incomplete**; 0/36 runs in validated four-arm blocks. Cross-H status: **unresolved**.

Walls/episode and survival are primary. Discovery (>50% survival) is secondary. Incomplete results are provisional.

| Model | Arm | Policy seeds | Walls/episode | Survival (%) | Discovery |
|---|---|---|---:|---:|---:|

All differences below are target minus reference. Intervals resample paired policy seeds within a fixed model.

| Comparison | Model | Paired seeds | Delta walls [95%] | Delta survival, pp [95%] |
|---|---|---:|---:|---:|

Equal-model means, model ranges, individual policy outcomes, paired evaluation-slot intervals, terminal diagnostics and every missing/invalid run are retained in publication.json.

- H48 was conditionally admitted after the audited H16 oracle_true discovery count was zero; this is an adaptive sensitivity, not independent confirmation or a preplanned primary result.
- H48 minus H16 changes a training package, not only GAE: start-context count, uninterrupted rollout length, collection-refresh delay and advantage normalization change while nominal positions and optimizer steps remain equal. Recurrent BPTT remains 16.
- Cross-horizon start prefixes and policy initialization constructors are matched; training action-noise keys and policy trajectories are not guaranteed identical.
- The exact-transition contrast includes absorbing terminal-support exposure. The termination contrast changes continuation, bootstrap and postterminal weighting together, and is not a pure reward intervention.
- Policy-seed intervals resample paired seeds within one fixed model and condition on the fixed evaluation set. With three policy seeds they are coarse descriptive intervals.
- Paired evaluation-slot intervals hold fitted policies fixed and resample shared slots jointly across policies; they do not measure training variability.
- All three fixed-model means, their equal-model descriptive mean and range are reported. No model-family confidence interval or pooled-nine-policy replication claim is justified.
- Intervals are unadjusted. No seed, arm, checkpoint, horizon or discovery-based subset is selected. A change does not identify a unique cause of H16 failure; failure of both packages bounds only these tested controller settings.
- The frozen learned collector does not record true-terminal/postterminal diagnostics; these remain unavailable, not zero. All arms record mean alive weight.
