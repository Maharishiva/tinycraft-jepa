# Target-gradient ablation

Status: **complete**; publication eligible: **True**.

Each row is one fitted model. Errors are mean absolute error per coordinate in pixels; lower is better. Ridge is secondary.

| World | Model seed | Mode | Final MLP MAE | Final ridge MAE | Train-mean MAE | Eval effective rank |
|---|---:|---|---:|---:|---:|---:|
| gray_frame_still | 0 | joint | 0.4195 | 4.4580 | 9.3549 | 242.44 |
| gray_frame_still | 0 | stop_target | 13.6986 | 9.3396 | 9.3549 | 491.99 |
| gray_frame_still | 1 | joint | 0.3734 | 3.2469 | 9.3549 | 237.47 |
| gray_frame_still | 1 | stop_target | 13.8117 | 9.3688 | 9.3549 | 492.97 |
| gray_frame_still | 2 | joint | 0.3161 | 4.6921 | 9.3549 | 265.41 |
| gray_frame_still | 2 | stop_target | 5.2661 | 5.7384 | 9.3549 | 469.52 |
| gray_frame_wind | 0 | joint | 14.4062 | 10.0027 | 9.9658 | 480.06 |
| gray_frame_wind | 0 | stop_target | 14.5892 | 9.9637 | 9.9658 | 492.41 |
| gray_frame_wind | 1 | joint | 14.6318 | 9.9979 | 9.9658 | 484.04 |
| gray_frame_wind | 1 | stop_target | 14.4619 | 9.9763 | 9.9658 | 493.93 |
| gray_frame_wind | 2 | joint | 14.4779 | 9.9951 | 9.9658 | 483.08 |
| gray_frame_wind | 2 | stop_target | 14.4906 | 10.0031 | 9.9658 | 493.20 |

Primary paired MLP interaction. Benefit = joint MAE minus stop-target MAE; interaction = wind benefit minus still benefit.

| Model seed | Still benefit (px) | Wind benefit (px) | Interaction (px) |
|---:|---:|---:|---:|
| 0 | -13.2791 | -0.1830 | +13.0961 |
| 1 | -13.4382 | +0.1699 | +13.6081 |
| 2 | -4.9499 | -0.0126 | +4.9373 |

Primary MLP interaction: +10.5472 px; seed SD 4.8650; range [+4.9373, +13.6081].

Secondary ridge interaction: +4.0341 px; seed SD 2.6655; range [+1.0384, +6.1435].

Midpoint status: complete_all_twelve_from_8500_schedule.

- The replication units are three fixed model initialization/data-order seeds, conditional on one training and one evaluation archive per world.
- The 50,000 readout-training frames and 10,000 evaluation frames are not independent model fits; no frame-based model-family confidence interval or p-value is reported.
- Wind changes visitation, temporal complexity and episode lengths as well as innovation variance; the interaction is not unique causal identification of an innovation-variance mechanism.
- The MLP is the sole primary readout. Ridge, representation rank, training loss and state/nuisance sensitivity are separate descriptive diagnostics, without winner selection.
- Readout failure is not proof of information absence; readout success is not proof of successful control.
- All saved checkpoints belong to the 8,500-update schedule. The final checkpoint is fixed prospectively; intermediate losses or readouts never select it.
- A positive interaction is a relative effect. It can result from worse still-world errors without any wind-world improvement; both within-world benefits and all twelve raw errors must accompany it.
- This NumPy audit verifies saved predictions and their metrics, not a fresh encoder/readout execution. Raw-feature hashes cannot be independently recomputed because features are not saved; independent re-encoding is a separate check.
