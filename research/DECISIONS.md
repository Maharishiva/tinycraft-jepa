# ICLR revision decisions

Started 2026-09-19 19:41:45 UTC. Deadline 2026-09-20 07:41:45 UTC.

- Original checkout is preserved. Separate Git worktree: `/private/tmp/tinycraft-jepa-iclr-20260919`, branch `codex/iclr-strong-paper`.
- V1 uses audited pre-existing results only; V2 incorporates completed preregistered follow-up results.
- Submission target is ICLR 2027; official ICLR 2026 awards/orals provide exemplars.
- No reward shaping, privileged controller inputs, latent inflation or cosmetic environment changes used to improve scores.
- Primary follow-up: matched short-rollout PPO with learned transitions/heads, true transitions/learned heads, true transitions/true heads.
- Common final evaluation keys, episode-level outcomes and exact model/policy seed hierarchy required.
- Gate colors and rendering remain unchanged for checkpoint compatibility. Static/dynamic nuisance already has separate controlled-world evidence.
- Craftax initially motivates an audit; old goal results are open-loop and exploratory. New common-harness evaluation considered.
- PVE acronym requested from user; no unverified paper identity will be asserted.
- Independent reviews must report earned ratings and unresolved weaknesses; no preassigned score.
