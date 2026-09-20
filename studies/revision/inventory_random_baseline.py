"""Uniform-random and no-op reference rows for the native Craftax inventory evaluation.

Same environment, episode count, horizon, key construction and aggregation as
studies.revision.inventory_eval, with the policy replaced by a fixed rule. The
rows exist to give the trained policies' achievement rates a scale; they are not
part of the paired JEPA/flow contrast and use no checkpoint.

  JAX_PLATFORMS=cpu python -m studies.revision.inventory_random_baseline --policy random --out OUT
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from studies.revision import inventory_eval as E


def run(policy, out, eval_seed=123, episodes=E.EPISODES, horizon=E.HORIZON, chunk=64):
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    import jax
    import jax.numpy as jnp
    import craftax
    from craftax.craftax_env import make_craftax_env_from_name
    from craftax.craftax_classic.constants import BlockType

    started = time.time()
    source_root = Path(craftax.__file__).resolve().parent
    assert {name: E.sha256(source_root / name) for name in E.PINNED_SOURCES} == E.PINNED_SOURCES
    env = make_craftax_env_from_name(E.ENV_NAME, auto_reset=False)
    ep = env.default_params
    assert int(ep.max_timesteps) == E.HORIZON and env.num_actions == 17
    keys = E.make_keys(jax, episodes, horizon, eval_seed)
    vreset = jax.jit(jax.vmap(lambda k: env.reset(k, ep)))
    vstep = jax.vmap(lambda k, s, a: env.step(k, s, a, ep))

    def extract(state):
        lava = state.map[jnp.arange(episodes), state.player_position[:, 0], state.player_position[:, 1]] == BlockType.LAVA.value
        return {"achievements": state.achievements,
                "inventory": jnp.stack([getattr(state.inventory, name) for name in E.INVENTORY], -1),
                "health": state.player_health, "in_lava": lava, "timestep": state.timestep}

    obs, state = vreset(jnp.asarray(keys["reset_keys"]))
    initial = {k: np.asarray(v) for k, v in extract(state).items()}
    carry = (state, jnp.ones(episodes, bool), jnp.full((episodes,), jnp.nan))

    @jax.jit
    def rollout_chunk(carry, env_keys, action_keys):
        def body(c, ks):
            state, active, terminal_score = c
            if policy == "random":
                action = jax.vmap(lambda k: jax.random.randint(k, (), 0, 17))(ks[1]).astype(jnp.int32)
            else:
                action = jnp.zeros((episodes,), jnp.int32)
            _, candidate, reward, done, info = vstep(ks[0], state, action)
            state2 = jax.tree.map(lambda old, new: E.freeze_inactive(old, new, active, jnp), state, candidate)
            done2 = active & done
            score2 = jnp.where(done2, info["score"], terminal_score)
            fields = extract(state2)
            record = {"actions": jnp.where(active, action, -1).astype(jnp.int8),
                      "rewards": jnp.where(active, reward, 0.), "native_done": done2,
                      "active_before": active, "alive_after": (fields["health"] > 0) & ~fields["in_lava"],
                      "policy_logits_finite": jnp.ones((episodes,), bool),
                      "achievements": fields["achievements"], "health": fields["health"].astype(jnp.int16),
                      "in_lava": fields["in_lava"]}
            return (state2, active & ~done, score2), record
        return jax.lax.scan(body, carry, (env_keys, action_keys))

    records = []
    for start in range(0, horizon, chunk):
        stop = min(start + chunk, horizon)
        carry, record = rollout_chunk(carry,
            jnp.asarray(keys["environment_step_keys"][:, start:stop].swapaxes(0, 1)),
            jnp.asarray(keys["policy_action_keys"][:, start:stop].swapaxes(0, 1)))
        records.append({k: np.asarray(v).swapaxes(0, 1) for k, v in record.items()})
        remaining = int(np.asarray(carry[1]).sum())
        if start == 0 or stop % 512 == 0 or remaining == 0:
            print(json.dumps({"native_eval_step": stop, "active_episodes": remaining,
                              "elapsed_s": round(time.time() - started, 1)}), flush=True)
        if not remaining:
            break
    raw = {k: np.concatenate([r[k] for r in records], axis=1) for k in records[0]}
    endpoint = {k: np.asarray(v) for k, v in extract(carry[0]).items()}
    endpoint["native_terminal_info_score"] = np.asarray(carry[2])
    rows, summary, first = E.aggregate_traces(raw, initial, endpoint, horizon=horizon)
    summary.update(status="complete", reportable=True, policy=policy, eval_seed=eval_seed, horizon=horizon,
                   elapsed_seconds=time.time() - started,
                   note="reference row: fixed rule instead of a trained policy; same keys, environment and aggregation as inventory_eval")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("primary", "achievement_count", "native_return", "executed_steps")}), flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=("random", "noop"), default="random")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--eval-seed", type=int, default=123)
    p.add_argument("--chunk", type=int, default=64)
    a = p.parse_args()
    run(a.policy, a.out, a.eval_seed, chunk=a.chunk)
