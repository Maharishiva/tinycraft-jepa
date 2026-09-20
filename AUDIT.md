# Harness audit: can the numbers in this repository be trusted?

Written 2026-09-20 on branch `paper/iclr-2027` for the repository owner, who will rerun training and
evaluation himself. Every claim below points at a file and line in this checkout. The question is
narrow: does any part of the training, evaluation or reporting pipeline reward the model or the
policy for something other than the task, select on the reported metric, leak privileged state, or
present a single lucky run as a family result.

## Verdict

- The **revision harness** (`studies/revision/`) is clean. Budgets are fixed in code and refused on
  mismatch, endpoints are the final checkpoint only, evaluation is paired on stored keys, admission of
  evaluation episodes never looks at the policy, rewards and terminations come from the models' own
  heads, and every run writes source, model and data hashes.
- The **original stochastic-worlds drivers** (`studies/stoch/`) have three reporting weaknesses that
  the revision exposed and worked around: the endpoint file records the best evaluation, real and
  imagined arms did not share evaluation keys, and the headline flow success was one policy of six.
  None of them is hidden; all are documented in `research/evidence-audit.md` and none reaches the
  paper's main claims after the rewrite.
- Two **selection choices** must stay disclosed wherever the gate appears: the gate JEPA encoder in
  the shared-CNN campaign is model seed 1, chosen because it was the only seed on which decoded
  controls had ever succeeded, and every flow gate policy shares one dynamics checkpoint.

## 1. Evaluation core

`studies/revision/warm_context_eval.py`

- Episodes are admitted by surviving a seven-action uniformly random prefix; the policy never
  participates in admission (module docstring, lines 1-6; `validate`, lines 97-120, checks that the
  attempt count equals admitted plus rejected).
- Every simulator, action-sampling and texture key is stored in the packet (dataclass fields, lines
  64-70). JEPA and flow policies of all seeds read the same packet, so comparisons are paired.
- Survival is derived from return only, `alive = ret > -0.5` (`studies/stoch/rl.py:169-176`, reused
  by the revision code). Death on the last capped step is a death.

`studies/stoch/rl.py:178-234` (`make_eval`): finished episodes are frozen and no hidden simulator
state reaches the actor. The policy observes pixels or the frozen model's latent, nothing else.

## 2. Policy training inside the models

`studies/revision/shared_decoded_ppo.py`

- Scientific budgets are hard-coded and refused on mismatch: 150 updates for the gate, 50 for the
  gray worlds, batch 512, horizon 48, chunk 16, minibatch 128, four epochs (lines 101-104); a smoke
  must use exactly two updates (105-106); the shared optimizer constants are checked (109-111); the
  flow recipe path is pinned (112-114); a JEPA dream requires the full-label head refit (115-116).
- Provenance records "fixed final update; no intermediate score selection" (line 172), and the final
  record is written from the last update with the policy file's hash (228-238).
- Rewards and terminations: flow reads its own jointly trained heads on its window
  (`studies/stoch/rl.py:307`); the JEPA reads its post-hoc heads on the current latent and action
  (`rl.py:348`). No true reward enters any imagined rollout.

`studies/revision/matched_ppo.py`

- The exact-simulator arms are admitted only after 1,024 stored states re-render to byte-identical
  pixels (`validate_reconstruction`, lines 88-107); the training loop then evaluates the final policy
  once on a fixed key and stores per-episode arrays (295-308). The docstring's promise of no reward
  clipping, shaping, latent scaling, privileged action choice or best-checkpoint selection matches the
  code.
- One design choice to keep in mind: in the exact-transition arm with learned heads, the learned
  continuation can remain positive after a true terminal; the run records that mass explicitly
  (lines 153-157, 285-287) instead of silently masking it.

## 3. Craftax inventory harness

- `studies/revision/inventory_ppo.py`: no behaviour cloning, anchor, action mask, reward alteration
  or checkpoint selection (line 5; provenance strings at 112-114); non-finite losses fail the run
  instead of being skipped (211).
- `studies/revision/inventory_jepa.py`: the once-per-episode reward modification is refused outright
  (225-226); the frozen model's weight digest must match the fitting manifest (107-109); the refit
  recipe is fixed (128-131).
- `studies/revision/inventory_flow.py`: native rollout, raw heads, no task hook (docstring, lines 1-7).
- `studies/revision/inventory_eval.py`: the reportable contract is frozen to 256 episodes, the
  native horizon and evaluation seed 123 (line 212); the policy sees native RGB and its previous
  action only (295, 326); the primary endpoint is the wood-pickaxe achievement (161-162).

## 4. What the original study got wrong, and how it was handled

All three items come from Codex's own `research/evidence-audit.md`, verified here against the code.

1. **Best-row endpoint file.** `studies/stoch/rl.py` keeps the evaluation with the highest survival
   and writes it as `final.json` (476-478 and 503 for real PPO; 581-583 and 610 for dream PPO). The
   last evaluated checkpoint is still saved as `model.pkl` (475, 580) and every evaluation is appended
   to `ledger.jsonl`. The historical tables were regenerated from the **last ledger row**
   (`studies/stoch/summarize.py:30-38`, `research/evidence-audit.md:18`), so no published number is
   a best-of-evals pick. Anyone reading a `final.json` from `runs/stoch/` directly must know this.
2. **Unpaired real-versus-imagined keys.** The original driver draws evaluation keys from the
   evolving training RNG (`rl.py:469, 574`), and real and dream runs consume different numbers of
   splits. Historical real-versus-dream contrasts are therefore unpaired
   (`research/evidence-audit.md:65`). The revision evaluates from stored packets.
3. **One lucky flow policy.** The 98 % flow gate survival was policy seed 0 of six sampled policies;
   the other five and all three mean-of-eight policies were far weaker, and all nine share the same
   dynamics checkpoint, SHA-256 `72b829ea…` (`research/evidence-audit.md:43`). The rewrite does not
   cite a family-level flow advantage from it.

Also worth knowing:

- **Gate JEPA seed 1 is preselected.** The shared-CNN queue uses `tile_factored_s1` for the gate
  (`studies/revision/queue_shared_decoded.py:112`, `model_seed=1` at 126) and records
  "favorably selected using prior development; not an independent encoder replication" (226). The
  reason is `research/evidence-audit.md:43`: both decoded-CNN discoveries in the original study
  occurred on encoder seed 1.
- **Supervision-stage asymmetry between recipes.** Flow's reward and termination losses send
  gradients into its dynamics trunk (full strength for the first half of training, then 0.1); the
  JEPA is reward-free and its heads are fitted afterwards on the frozen latent. The shared-CNN
  campaign refits the JEPA heads on all 500k training labels (`refit_full_label_heads.py`,
  `prep_shared_decoded.sh`) so the label count is equal, but the stage is not
  (`research/shared_decoded_cnn_protocol.md:27-33`). Any JEPA-versus-flow gap is a recipe gap.
- **Decoders are supervised and off-budget.** The pixel decoder used for decoded-latent controls and
  for every reconstruction figure is trained on 80,000 frames with a reconstruction loss after the
  world model is frozen. A decoded-latent result is a statement about recoverable information under
  that read-out, not about the unsupervised model alone.
- **Still-world survival is not a control certificate.** A state-blind alternating policy survives
  the still worlds at 100 %, so survival there is reported only alongside walls passed or position
  error. `studies/revision/reference_policies_eval.py` runs two state-blind rules (uniform random,
  left-right alternation) through the exact paired evaluator on the same packets; their rows sit
  under `runs/revision/shared_decoded/reference/` and in the shared-CNN table. Both rules score 0 %
  on the wind worlds and the gate.
- **Gate codec parity was red once.** The first gate parity files
  (`runs/revision/shared_decoded/validation/gate_{jepa,flow}_codec.json`, 23:47-23:58 UTC on
  2026-09-19) report `passed: false` (JEPA head-cache latents off by 0.0625; flow batch-256 pixels
  off by one level). Both were precision-convention mismatches between callers; the sealed
  re-checks written minutes later (`gate_jepa_codec_sealed.json`, `gate_flow_codec_actualpath.json`)
  are exact, and every scientific gate policy started after them (first at 00:31 UTC). The gray
  worlds passed on the first run (`prep_20260920.log`).
- **Imagined termination is a model property, not a harness one.**
  `studies/revision/imagined_termination_probe.py` rolls each frozen model forward under the recorded
  actions and reads its own termination head on the predicted latent and on the encoder latent of
  the real frame. In the per-frame-noise wind world the JEPA head reads 0.91 on every predicted
  latent from the first imagined step (0.03-0.06 on encoder latents, matching the label rate), so
  PPO inside that model sees death independent of action. The still world tracks the data; the
  persistent-noise wind world is over-optimistic (imagined survival 0.40 after 48 steps against
  0.008 recorded). Outputs in `runs/revision/shared_decoded/termination_probe/`.

## 5. Provenance limits

- The remote source directories used for the original study have no `.git`; identity is by file
  hash only (`research/gpu_inventory.md`, "Execution environments"). The e2e copy of `rl.py` differed
  from the local original at audit time; the differing files are preserved under
  `research/gpu_inventory/original_e2e_source/`.
- The gate training archives on the two VMs have different container bytes but identical array
  hashes (`research/gpu_inventory.md`, "Exact gate artifacts").
- Machine paths (`/dev/shm/redcube`, `/home/rldoeswork/personal/stoch`) are defaults in the revision
  entry points and are overridable by flag or environment variable; see the README for the list.

## 6. Independent checks the owner can run

Each of these reproduces a reported number from stored artifacts without training.

```bash
# paired shared-CNN campaign table from status.json and per-policy final.json
python -m studies.revision.analyze_shared_decoded --root runs/revision/shared_decoded

# codec parity: decoded_dream and warm_context_eval see the same encode/decode path
python -m studies.revision.check_shared_decoded --family jepa --config gate ...

# regenerate the historical gate summary from ledgers (last row, not best)
python -m studies.stoch.summarize

# matched-PPO analysis from saved per-episode arrays
python -m studies.revision.analyze_matched --root runs/revision/matched
```

Retraining anything means using the budgets above; the scripts refuse other values without `--smoke`.

## 7. Tests

The harness ships CPU-only tests in two styles. Run on 2026-09-20 on the e2e VM with
`JAX_PLATFORMS=cpu`: 30 unittest cases passed with 1 skipped, and all 14 function-style tests
passed, once the mirror had every file the provenance hashes refer to (the target-gradient test
hashes `studies/stoch/train_arms.sh` and fails with a missing-file error otherwise).

```bash
export JAX_PLATFORMS=cpu
python -m unittest studies.revision.test_inventory_eval studies.revision.test_inventory_queue \
    studies.revision.test_inventory_flow_numpy studies.revision.test_inventory_numerical_review \
    studies.revision.test_warm_context_eval studies.revision.test_inventory_jepa
python -m pytest tests/test_port.py studies/revision/test_target_gradient.py tests/test_target_gradient_probe.py
```

What the tests cover: the paired evaluation packet and its admission rule; the native Craftax
aggregation invariants (monotone achievements, telescoping reward identity, no post-terminal
updates); the JEPA and flow inventory adapters; the fixed policy-training matrix; SIGReg against the
reference implementation and predictor causality; and that the detached-target trainer reproduces
the original trainer bit for bit in joint mode. They do not cover the original `studies/stoch`
drivers.
