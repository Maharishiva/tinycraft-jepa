# Forecast extraction: numerical audit and pre-fit correction

The first forecast-readout smoke stopped at its strict parity guard, before fitting a readout. It cached frames in batches of 32 while comparing with the public rollout's encoder call on 16 contexts of eight frames, flattened internally to 128 frames. The maximum coordinate difference was 0.0625. This was investigated rather than accepted by increasing a tolerance.

All measurements below use the same saved seed-0 checkpoint, the same 128 frames from 16 sampled training contexts, inference mode with `train=False`, and unchanged parameter/BatchNorm-statistic hashes. The frames were exactly reproduced by the simulator. Of these frames, 50 contain a wall. Their matched wall-free counterfactuals were rendered from the same positions and other state variables. This is a numerical diagnostic on a small fixed sample, not a new control experiment or an independent population estimate.

## Measured scope

The reference encoder call has shape 16×8 frames. Its mean squared latent norm is 511.72. The mean wall-removal squared distance on wall-containing frames is 0.03261.

| Flattened frame batch | Maximum coordinate difference | Mean numerical squared L2 over frames | Relative L2 RMS | Mean cosine |
|---:|---:|---:|---:|---:|
| 16 | 0.0625 | 0.017606 | 0.005866 | 0.9999831 |
| 32 | 0.0625 | 0.007005 | 0.003700 | 0.9999934 |
| 64 | 0.0625 | 0.002113 | 0.002032 | 0.9999980 |
| 128 | 0 | 0 | 0 | 1 |
| 256, padded | 0.0000610 | 3.37e-11 | 2.56e-7 | effectively 1 |

At the matched 128-frame shape, the cached encoder and public context encoder are bit-exact. The public combined encoder/rollout and standalone encoder are also bit-exact at the same shape. The refactored predictor and public rollout are bit-exact through horizons 1, 2, 4 and 8 when supplied the same latent context and the same context batch shape.

Changing only predictor batch size from 16 to 32 produced maximum coordinate differences of 0.0000305, 0.03125, 0.046875 and 0.046875 at horizons 1, 2, 4 and 8. Changing from 16 to 64 was bit-exact on this sample. This illustrates why parity should compare the same execution shape rather than presume smooth or monotonic rounding behavior across batch sizes.

## Relation to the wall contrast

Total rounding error can point in directions unrelated to the wall contrast. For frame-batch 32, only 10.2% of squared numerical error on the wall-containing frames projects onto the paired wall-present minus wall-absent latent direction. Its signed coordinate shift along that direction averages −0.118, with RMS 0.162; the coordinate's endpoints are 0 for the wall-free counterfactual and 1 for the reference wall encoding. For frame-batch 16, the corresponding fraction is 14.0%, signed mean −0.265 and RMS 0.280.

These projections quantify an execution-shape effect along one chosen counterfactual direction. They do not establish information deletion, a change in decoded pixels, a failure of a trained readout, or control causality. The large-batch recipes used by existing results were not reevaluated in this check. In particular, this audit must not be presented as proof that the original paper's wall diagnostic or policies were affected by the same magnitude of error.

## Approved correction, before successful readout fitting

The revised extractor caches real frames in fixed padded batches of `B * history`, and checks against the public function on `B` padded contexts. Predictor batch size is `B` in both paths. Every real/copy/future encoder feature uses that same canonical frame batch. No checkpoint parameter, compute dtype, parity tolerance, data selection, label, readout normalization or fit objective changed.

The old source is retained as `runs/revision/forecast/encoder_parity/forecast_readout.f35.py` with SHA256 `f35fbb0726d87c761ae664034a9e89f0204e9ab4bcf51a816753e792c9ba05dc`. The corrected source SHA256 is `37bbc37594579188613fd785670850d593caf62596f9d2f282d873c497f50b18`.

Both subsequent smoke checks passed strict parity with maximum differences of exactly 0.0 for encoded contexts and predictions in training and evaluation splits:

- Context batch 32, encoder frame batch 256: 87.7 seconds total.
- Production context batch 256, encoder frame batch 2048: 105.0 seconds total, completed on the 40 GB A100.

Both verified unchanged model parameters/statistics and identical common/copy versus per-source/copy predictions. These small smoke fits are excluded from final evidence. The corrected production shape is fixed for all three encoder-seed evaluations.

## Artifacts

- `runs/revision/forecast/encoder_parity/results.json`: detailed relative/absolute errors and frozen-state hashes.
- `runs/revision/forecast/encoder_parity/per_frame.npz`: frame IDs, reference/counterfactual latents and per-frame errors.
- `runs/revision/forecast/encoder_parity/wall_axis_projection.json`: directional summaries.
- `runs/revision/forecast/encoder_parity/tinycraft_encoder_parity.py`: diagnostic implementation.
- `runs/revision/forecast/encoder_parity/failed_smoke_f35.log`: original strict-guard traceback.
- `runs/revision/forecast/smoke_m0_canonical` and remote `smoke_m0_production_shape`: runtime checks only.
