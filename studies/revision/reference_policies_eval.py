"""State-blind reference rows on the shared-CNN evaluation packets.

Two fixed rules replace the trained policy in the exact paired evaluator of
warm_context_eval (same packet, same episode keys, same admission):

  random     uniform over the four moves
  alternate  left, right, left, right ... (the blind survival strategy of the still worlds)

Neither sees an observation; the identity adapter hands the evaluator raw frames,
which these rules ignore. The rows give the trained policies' survival a scale and
are not part of the paired JEPA/flow contrast.

  JAX_PLATFORMS=cpu python -m studies.revision.reference_policies_eval \
      --packets runs/revision/shared_decoded/packets --out runs/revision/shared_decoded/reference
"""
import argparse
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from studies.revision import warm_context_eval as W
from studies.stoch import env as S

LEFT, RIGHT = 0, 2


class IdentityAdapter:
    def posterior(self, raw):
        return raw

    def posterior_step(self, real, frame):
        return frame


class RandomRule(nnx.Module):
    def step(self, obs, prev_a, g):
        n = obs.shape[0]
        return jnp.zeros((n, S.N_ACTIONS)), jnp.zeros((n, 1)), g


class AlternateRule(nnx.Module):
    """Parity lives in the first GRU slot; the evaluator threads g for us."""

    def step(self, obs, prev_a, g):
        n = obs.shape[0]
        go_right = g[:, 0] > 0.5
        action = jnp.where(go_right, RIGHT, LEFT)
        logits = jax.nn.one_hot(action, S.N_ACTIONS) * 40.0
        g2 = g.at[:, 0].set(jnp.where(go_right, 0.0, 1.0))
        return logits, jnp.zeros((n, 1)), g2


RULES = {"random": RandomRule, "alternate": AlternateRule}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worlds", nargs="*", default=["gray_frame_still", "gray_frame_wind", "gray_episode_wind", "gate"])
    a = p.parse_args()
    for world in a.worlds:
        packet = W.load_packet(a.packets / f"{world}.npz")
        spec = S.CONFIGS[packet.metadata["config"]]
        evaluator = W.make_evaluator(spec, IdentityAdapter(), packet)
        for name, rule in RULES.items():
            out = a.out / world / name
            if (out / "summary.json").exists():
                continue
            result = evaluator(rule())
            summary = W.save_evaluation(result, packet, out, policy_seed=None,
                                        provenance={"reference_rule": name, "observation": "ignored"})
            print(json.dumps({"world": world, "rule": name, **{k: round(summary[k], 4) for k in
                              ("survival", "walls_passed", "control_length", "return")}}), flush=True)


if __name__ == "__main__":
    main()
