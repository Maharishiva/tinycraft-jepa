# Frozen positive-control protocol: visual goal-directed red-cube control

This protocol was fixed before any new controller outcomes. It is a positive
control for a frozen JEPA's ability to navigate to a visually specified position,
not evidence of long-horizon strategic planning. The primary condition is
`gray_frame_still`; `off_still` is an optional separately reported condition.

- **Existing model:** original `tile_factored/model_probe.npz`, seed 0, native
  512-dimensional latent. No retraining, latent scaling, head, decoder or
  task-specific latent features enter action selection. The model hash and all
  configuration fields are recorded before evaluation. Additional existing
  training seeds, if evaluated, use precisely the same cases and settings.
- **Fresh starts:** evaluation seed 20260920. Draw native resets and seven random
  actions; retain the first 128 candidates alive throughout all seven steps.
  Reject only past terminal prefixes. Record candidate IDs and rejection count.
  Every controller receives the same eight genuine observations and actions.
- **Goal distribution:** uniformly sample among safe integer positions sharing
  both coordinate residues modulo the native 3-pixel action step with the start,
  with Manhattan distance between 6 and 12 minimum actions inclusive. These are
  reachable goals, not successful model trajectories selected retrospectively.
  Render one independent gray-noise RGB goal per case and keep it fixed. Only
  that image is given to JEPA; coordinates are reserved for metrics/reference.
- **JEPA controller:** enumerate all 64 three-action sequences, recursively apply
  the frozen predictor, and minimize the sum of three mean squared latent
  distances to the encoded goal. Execute the first action and replan from the
  new real history. This standard summed-stage-cost MPC variant differs from the
  historical terminal-only planner. No subsequent planner tuning is allowed.
  A prespecified horizon-one JEPA comparison measures how much the depth adds.
- **References:** uniform random actions; fixed, state/goal-blind left-right-up-
  down cycle; privileged exact-transition oracle with the same three-step search
  and summed physical squared-distance cost. Oracle rejects any plan with an
  intermediate native death. It is a sanity check, not a learned competitor or
  an isolation of the predictor alone, because its cost has state information.
- **Budget:** 24 actions after the valid prefix. Native dynamics and border death
  remain unchanged. Stop externally on the first exact goal hit or first native
  death; no post-death recovery or requirement to hold position is counted.
  There is no reward learning or modification of a training reward.
- **Primary endpoint:** first exact goal-position hit before death, count/128
  with Wilson 95% interval. Prespecified secondaries: minimum and final Manhattan
  distance, ever within one native action (L1 ≤ 3 pixels), native deaths, and
  first-hit step among successes. All 128 cases remain in denominators.
- **Pairing and numerics:** separate fixed RNG namespaces for reset, warmup,
  goal, random actions and execution noise. Execution noise is identical across
  policies for the same case/time. Both current history and repeated RGB goal
  use the canonical encoder shape `[16,8,63,63,3]`. Only the framewise encoder
  sees repeated goals; the dynamics receives actual eight-frame histories.
  Predictor history uses outgoing actions `[a0,...,a6,candidate a7]`.
- **Visual selection:** episodes 0, 1 and 2, specified before outcomes. Save all
  policies' raw RGB trajectory, exact actions/positions and goal images. Save
  decoded selected model futures solely as an illustrative readout, never as a
  controller input. Preserve failures and all horizon-one results.

Expected GPU cost: under 15 minutes including compilation, approximately 590k
predictor transitions for the horizon-three arm. No GPU launch without the
parent's resource coordination. E2E is the proposed host.

```sh
python -m studies.revision.redcube_goal_eval \
  --model MODEL.npz --config gray_frame_still \
  --out runs/revision/redcube_goal/gray_frame_still/m0
```

Independent protocol review checked lattice reachability, cumulative warmup
survival, action alignment, first-death/first-hit masking, and the privileged
oracle's intermediate death exclusion. The empty arena can be solved reactively;
any success must be described as goal-directed control rather than necessity of
deep planning.

Before launch, a separate agent independently reviewed the completed source and
found no blocking correctness error. Local self-check and Python compilation
passed. An independent CPU check of the actual goal sampler and oracle action
function reached 1,000/1,000 independently drawn eligible goals in exactly the
Manhattan shortest-path number of steps, with no unsafe pre-hit position.
The visual packet explicitly marks `forecast_valid`; forecasts after the first
terminal event are discarded padding and must be suppressed by visualizers.

Before seed-0 outcomes, the parent approved evaluation of all three original
`gray_frame_still` training seeds with identical 128 cases and settings. Run
seed 0 first to measure runtime. If all three fit the initial 15-minute GPU slot,
finish them in that slot; otherwise release the GPU and schedule seeds 1 and 2
later. The scheduling decision depends only on runtime/resources, never result
direction. Preserve the three model results separately.

A CPU runtime smoke test also passed before launch: replace the network with an
exact two-dimensional position model, then require the planner's selected action
to match the exact oracle, every recursively predicted position to equal the
selected action rollout, and the returned cost to equal the independently
computed summed mean squared distance. This verifies the executed JIT path and
recursive outgoing-action convention without measuring a scientific checkpoint.
