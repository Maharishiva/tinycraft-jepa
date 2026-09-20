"""Fixed-final native Craftax inventory evaluation; imports NumPy only until run.

Every policy uses 256 ordinary resets, sampled actions, identical episode/time
keys, and the first native terminal transition (at most 10,000 actions). No
world model, state-based policy, reward modification, or checkpoint selection.
Run: python studies/revision/inventory_eval.py --checkpoint RUN/model.pkl --out OUT
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ENV_NAME = "Craftax-Classic-Pixels-v1"
EPISODES = 256
HORIZON = 10000
ACHIEVEMENTS = (
    "COLLECT_WOOD", "PLACE_TABLE", "EAT_COW", "COLLECT_SAPLING", "COLLECT_DRINK",
    "MAKE_WOOD_PICKAXE", "MAKE_WOOD_SWORD", "PLACE_PLANT", "DEFEAT_ZOMBIE",
    "COLLECT_STONE", "PLACE_STONE", "EAT_PLANT", "DEFEAT_SKELETON",
    "MAKE_STONE_PICKAXE", "MAKE_STONE_SWORD", "WAKE_UP", "PLACE_FURNACE",
    "COLLECT_COAL", "COLLECT_IRON", "COLLECT_DIAMOND", "MAKE_IRON_PICKAXE",
    "MAKE_IRON_SWORD",
)
INVENTORY = ("wood", "stone", "coal", "iron", "diamond", "sapling", "wood_pickaxe",
             "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword")
TOOLS = {"MAKE_WOOD_PICKAXE": "wood_pickaxe", "MAKE_STONE_PICKAXE": "stone_pickaxe",
         "MAKE_WOOD_SWORD": "wood_sword", "MAKE_STONE_SWORD": "stone_sword"}
PINNED_SOURCES = {
    "craftax_classic/constants.py": "c1e78c1c1dc9e6d24f70ea80da194aba1471bac49df9d673c620dddfe77cd090",
    "craftax_classic/game_logic.py": "3d1d61066d71fa6730d93fa4c874ca9b862de0e404102ff1fe5b23ccec60da1b",
    "craftax_classic/envs/craftax_state.py": "cee0dfe2d0be17803125e366180967dac9485e02d8fdbf10a10856e3cc6e082c",
    "craftax_classic/envs/common.py": "51f89ae914892e5b316695d93dec86319ad39c528ca3c1c40f7c80ebd40f8f20",
    "craftax_classic/envs/craftax_pixels_env.py": "6fa6a620748326aaeda8415e16aa18e7e0151096a839959b58264d0d7f465cd1",
    "craftax_classic/world_gen.py": "cf0b42999d675ef722f74c2e356e8625515948ed35f8e0392d5ce5285c583f4f",
    "environment_base/environment_bases.py": "c96af3e0690c0546fef0b6f7b4d605b4c22f6599f2299e1cd71ac9a36ba0ca53",
    "craftax_env.py": "f74d828dbc9802984e026ace29293527dfd3901cc945d3c7792d199dc92affa3",
}
AGENT_SHA256 = "1bfeb69f0a6098959a409c4601e8a92219b1967f297b1e35beb96440edceb132"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_hash(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def freeze_inactive(old, candidate, active, xp=np):
    """Use the PRE-transition activity, preserving a new terminal state."""
    mask = active.reshape(active.shape + (1,) * (candidate.ndim - active.ndim))
    return xp.where(mask, candidate, old)


def scalar_summary(values):
    x = np.asarray(values, np.float64)
    sd = float(x.std(ddof=1)) if len(x) > 1 else None
    return {"n": len(x), "mean": float(x.mean()), "sd": sd,
            "sem": sd / math.sqrt(len(x)) if sd is not None else None}


def binary_summary(values):
    x = np.asarray(values, bool)
    n, k = len(x), int(x.sum())
    p, z = k / n, 1.959963984540054
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"n": n, "successes": k, "rate": p,
            "wilson_95": [max(0., mid - half), min(1., mid + half)]}


def aggregate_traces(raw, initial, endpoint, horizon=HORIZON):
    """Audit episode-major raw traces and return rows plus fixed endpoints.

    Pure NumPy: tests exercise the same aggregation used after GPU execution.
    Inactive action=-1, reward=0 and done=False are explicit padding, never data.
    """
    active = np.asarray(raw["active_before"], bool)
    done = np.asarray(raw["native_done"], bool)
    actions = np.asarray(raw["actions"])
    rewards = np.asarray(raw["rewards"], np.float64)
    ach = np.asarray(raw["achievements"], bool)
    n, steps = active.shape
    assert done.shape == actions.shape == rewards.shape == (n, steps)
    assert ach.shape == (n, steps, len(ACHIEVEMENTS))
    assert 0 < steps <= horizon
    assert active[:, 0].all(), "Every admitted reset must receive an action"
    expected = np.concatenate([np.ones((n, 1), bool), ~np.maximum.accumulate(done[:, :-1], axis=1)], axis=1)
    assert np.array_equal(active, expected), "Post-terminal actions or missing active steps"
    assert np.all((actions[active] >= 0) & (actions[active] < 17))
    assert np.all(actions[~active] == -1) and np.all(rewards[~active] == 0)
    assert not done[~active].any() and np.all(done.sum(1) <= 1)
    assert np.isfinite(rewards).all()
    assert np.asarray(raw["policy_logits_finite"])[active].all(), "Nonfinite actor logits"
    assert not np.asarray(initial["achievements"]).any()
    assert not np.asarray(initial["inventory"]).any()
    assert np.all(np.asarray(initial["health"]) == 9)
    previous = np.concatenate([np.asarray(initial["achievements"], bool)[:, None], ach[:, :-1]], axis=1)
    assert not ((previous & ~ach) & active[:, :, None]).any(), "Achievements are not monotone"
    lengths = active.sum(1)
    native_done = done.any(1)
    final_ach = ach[np.arange(n), lengths - 1]
    assert np.array_equal(final_ach, endpoint["achievements"])
    assert np.array_equal(np.asarray(raw["health"])[np.arange(n), lengths - 1], endpoint["health"])
    assert np.array_equal(np.asarray(raw["in_lava"])[np.arange(n), lengths - 1], endpoint["in_lava"])
    assert np.array_equal(lengths, endpoint["timestep"]), "Reset or timestep discontinuity"
    return_ = rewards.sum(1)
    identity = final_ach.sum(1) - np.asarray(initial["achievements"]).sum(1) + .1 * (
        np.asarray(endpoint["health"]) - np.asarray(initial["health"]))
    assert np.allclose(return_, identity, rtol=0, atol=2e-4), "Native reward telescoping identity failed"
    dead = np.asarray(endpoint["health"]) <= 0
    lava = np.asarray(endpoint["in_lava"], bool)
    time_limit = np.asarray(endpoint["timestep"]) >= HORIZON
    assert np.array_equal(native_done, dead | lava | time_limit)
    truncated = ~native_done & (lengths >= horizon)
    assert np.all(native_done | truncated), "Incomplete evaluation"
    survival = ~dead & ~lava
    raw_alive = (np.asarray(raw["health"]) > 0) & ~np.asarray(raw["in_lava"], bool)
    assert np.array_equal(raw_alive, raw["alive_after"])
    first = np.where((ach & active[:, :, None]).any(1),
                     (ach & active[:, :, None]).argmax(1) + 1, -1)
    inventory = np.asarray(endpoint["inventory"])
    assert inventory.shape == (n, len(INVENTORY))
    assert np.all((inventory >= 0) & (inventory <= 9))
    for name, field in TOOLS.items():
        assert np.array_equal(final_ach[:, ACHIEVEMENTS.index(name)], inventory[:, INVENTORY.index(field)] > 0), name
    endpoint_score = np.expm1(np.log(101.) * final_ach.mean(1))
    scores = np.asarray(endpoint["native_terminal_info_score"])
    assert np.allclose(scores[native_done], endpoint_score[native_done], rtol=2e-5, atol=2e-5)
    assert np.isnan(scores[~native_done]).all()
    rows = []
    for i in range(n):
        rows.append({"episode_id": i, "executed_steps": int(lengths[i]),
                     "native_done": bool(native_done[i]), "health_nonpositive": bool(dead[i]),
                     "in_lava": bool(lava[i]), "native_time_limit": bool(time_limit[i]),
                     "evaluator_truncated": bool(truncated[i]), "survived_endpoint": bool(survival[i]),
                     "native_return": float(return_[i]), "initial_health": int(initial["health"][i]),
                     "final_health": int(endpoint["health"][i]),
                     "achievements": {name: bool(final_ach[i, j]) for j, name in enumerate(ACHIEVEMENTS)},
                     "first_acquisition_step": {name: int(first[i, j]) if first[i, j] > 0 else None for j, name in enumerate(ACHIEVEMENTS)},
                     "final_inventory": {name: int(inventory[i, j]) for j, name in enumerate(INVENTORY)},
                     "native_terminal_info_score": float(scores[i]) if native_done[i] else None})
    rates = final_ach.mean(0)
    tool_flags = final_ach[:, [ACHIEVEMENTS.index(name) for name in TOOLS]]
    summary = {
        "episodes": n, "primary_endpoint": "MAKE_WOOD_PICKAXE",
        "primary": binary_summary(final_ach[:, ACHIEVEMENTS.index("MAKE_WOOD_PICKAXE")]),
        "four_tool_success": {name: binary_summary(tool_flags[:, j]) for j, name in enumerate(TOOLS)},
        "four_tool_count": scalar_summary(tool_flags.sum(1)), "all_four_tools": binary_summary(tool_flags.all(1)),
        "all_22_achievement_rates": {name: binary_summary(final_ach[:, j]) for j, name in enumerate(ACHIEVEMENTS)},
        "native_return": scalar_summary(return_), "executed_steps": scalar_summary(lengths),
        "achievement_count": scalar_summary(final_ach.sum(1)), "survived_endpoint": binary_summary(survival),
        "native_done": binary_summary(native_done), "native_time_limit": binary_summary(time_limit),
        "evaluator_truncated": binary_summary(truncated),
        "achievement_rate_geometric_aggregate": float(np.expm1(np.log1p(100 * rates).mean())),
        "achievement_rate_geometric_aggregate_formula": "exp(mean_j(log(1+100*mean_episode(A_j))))-1",
        "mean_native_terminal_info_score": float(scores[native_done].mean()) if native_done.any() else None,
        "native_return_identity_max_abs_error": float(np.abs(return_ - identity).max()),
        "uncertainty_scope": "Across evaluation episodes for one fixed trained policy; not model-seed replication",
    }
    return rows, summary, first


def make_keys(jax, episodes, horizon, seed):
    """Episode keys are independent of policy, batch size and termination time."""
    import jax.numpy as jnp
    root = jax.random.PRNGKey(seed)
    episode_keys = jax.vmap(lambda i: jax.random.fold_in(root, i))(jnp.arange(episodes, dtype=jnp.uint32))
    reset = jax.vmap(lambda k: jax.random.fold_in(k, 0))(episode_keys)
    times = jnp.arange(horizon, dtype=jnp.uint32)
    streams = [jax.vmap(lambda k: jax.vmap(lambda t: jax.random.fold_in(jax.random.fold_in(k, stream), t))(times))(episode_keys)
               for stream in (1, 2)]
    return {"episode_ids": np.arange(episodes, dtype=np.int32), "reset_keys": np.asarray(reset),
            "environment_step_keys": np.asarray(streams[0]), "policy_action_keys": np.asarray(streams[1])}


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "dtype"):
        return np.asarray(value).tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def evaluate_checkpoint(checkpoint, out, *, eval_seed=123, chunk=64, keys_path=None,
                        smoke=False, episodes=EPISODES, horizon=HORIZON):
    """Evaluate exactly one prospective fixed-final policy, with no training."""
    checkpoint, out = Path(checkpoint).resolve(), Path(out).resolve()
    if not smoke:
        assert checkpoint.name == "model.pkl", "Only the fixed FINAL model.pkl is admitted"
        assert (episodes, horizon, eval_seed) == (EPISODES, HORIZON, 123), "The reportable evaluation contract is frozen"
    assert 0 < episodes <= EPISODES and 0 < horizon <= HORIZON
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing nonempty evaluation output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    assert chunk > 0
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    actor_source = Path(__file__).with_name("inventory_source") / "agent.py"
    assert sha256(actor_source) == AGENT_SHA256, "Shared historical actor source changed"
    checkpoint_hash = sha256(checkpoint)
    final_path = checkpoint.parent / "final.json"
    final_manifest = json.loads(final_path.read_text()) if final_path.exists() else None
    if not smoke:
        assert final_manifest is not None, "Missing prospective training final.json"
        assert final_manifest["checkpoint_sha256"] == checkpoint_hash, "Checkpoint differs from training final.json"
        assert final_manifest["budget_complete"] is True and final_manifest["env_steps"] == 4998400
        for field in ("source_hashes_unchanged", "world_model_bytes_unchanged", "adapter_state_unchanged"):
            assert final_manifest[field] is True, field
        # Pure provenance helper; does not import the world-model trainer.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from inventory_common import source_hashes
        assert final_manifest["source_hashes"] == source_hashes(), "Sources changed since final policy training"
    sys.path.insert(0, str(actor_source.parent))
    import agent
    assert Path(agent.__file__).resolve() == actor_source.resolve(), "Conflicting imported agent module"
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import craftax
    from craftax.craftax_env import make_craftax_env_from_name
    from craftax.craftax_classic.constants import Achievement, BlockType
    from craftax.environment_base.environment_bases import EnvironmentNoAutoReset

    started = time.time()
    source_root = Path(craftax.__file__).resolve().parent
    actual_sources = {name: sha256(source_root / name) for name in PINNED_SOURCES}
    assert actual_sources == PINNED_SOURCES, "Native environment differs from the audited source"
    # The core files above were inspected before the run. Also retain the
    # observation/reset dependencies and texture bytes for exact provenance.
    package_sources = {str(p.relative_to(source_root)): sha256(p) for p in sorted(source_root.rglob("*.py"))}
    texture_sources = {str(p.relative_to(source_root)): sha256(p)
                       for p in sorted((source_root / "craftax_classic" / "assets").rglob("*")) if p.is_file()}
    assert importlib.metadata.version("craftax") == "1.5.0"
    assert tuple(a.name for a in sorted(Achievement, key=lambda x: x.value)) == ACHIEVEMENTS
    env = make_craftax_env_from_name(ENV_NAME, auto_reset=False)
    ep = env.default_params
    assert isinstance(env, EnvironmentNoAutoReset)
    assert type(env).__name__ == "CraftaxClassicPixelsEnvNoAutoReset"
    assert int(ep.max_timesteps) == HORIZON and env.num_actions == 17
    assert env.static_env_params == env.default_static_params()
    ac, meta = agent.load_ac(checkpoint)
    assert ac.recurrent is False and ac.gru is None and ac.cond_dim == 0 and ac.q is None
    assert meta.get("cond", "none") == "none"
    if not smoke:
        assert meta.get("budget_complete") is True, "Checkpoint lacks final-budget completion"
        assert meta.get("env_steps", meta.get("steps")) == 4998400, "Wrong prospective policy budget"
        for field in ("arm", "family", "model_seed", "policy_seed", "model_sha256", "env_steps", "budget_complete"):
            assert meta[field] == final_manifest[field], f"Final manifest/checkpoint metadata disagree: {field}"
    keys = make_keys(jax, episodes, horizon, eval_seed)
    keys_path = Path(keys_path).resolve() if keys_path else out / "eval_keys.npz"
    if keys_path.exists():
        with np.load(keys_path, allow_pickle=False) as saved:
            assert set(saved.files) == set(keys)
            assert all(np.array_equal(saved[k], v) for k, v in keys.items()), "Evaluation keys differ"
    else:
        keys_path.parent.mkdir(parents=True, exist_ok=True)
        with keys_path.open("xb") as f:
            np.savez_compressed(f, **keys)
    provenance = {
        "checkpoint": str(checkpoint), "policy_checkpoint_sha256": checkpoint_hash, "checkpoint_meta": _jsonable(meta),
        "training_final_manifest_sha256": sha256(final_path) if final_path.exists() else None,
        "policy_training_seed": meta.get("policy_seed", meta.get("seed")), "world_model_training_seed": meta.get("model_seed"),
        "world_model_checkpoint_sha256": meta.get("world_model_checkpoint_sha256", meta.get("model_sha256", meta.get("wm_sha256"))),
        "agent_source_sha256": sha256(actor_source), "evaluator_source_sha256": sha256(__file__),
        "python_version": sys.version,
        "environment_source_sha256": actual_sources,
        "environment_package_python_sha256": package_sources, "environment_texture_sha256": texture_sources,
        "versions": {p: importlib.metadata.version(p) for p in ("craftax", "gymnax", "jax", "jaxlib", "flax", "numpy")},
        "env_name": ENV_NAME, "auto_reset": False, "env_params": _jsonable(ep),
        "static_params": _jsonable(env.static_env_params), "episodes": episodes, "horizon": horizon,
        "execution_chunk": chunk,
        "smoke": smoke, "reportable": not smoke,
        "eval_seed": eval_seed, "policy_actions": "categorical samples from unchanged 17-way actor logits",
        "policy_observation": "native RGB rounded to uint8 with original agent.to_u8; previous action starts at native NOOP=0",
        "keys_path": str(keys_path), "keys_sha256": sha256(keys_path),
        "key_array_sha256": {k: array_hash(v) for k, v in keys.items()},
        "key_rule": "E=fold_in(PRNGKey(eval_seed),episode_id); reset=fold_in(E,0); env_t=fold_in(fold_in(E,1),t); action_t=fold_in(fold_in(E,2),t); t starts at 0",
        "checkpoint_selection": ("NONREPORTABLE pipeline smoke" if smoke else "Fixed final model.pkl after 4,998,400 imagined transitions; no evaluation selection"),
        "policy_params": agent.n_params(ac), "devices": [str(d) for d in jax.devices()],
        "backend": jax.default_backend(),
        "jax_config": {name: getattr(jax.config, name) for name in (
            "jax_default_matmul_precision", "jax_default_prng_impl", "jax_enable_x64", "jax_threefry_partitionable")},
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    vreset = jax.jit(jax.vmap(lambda k: env.reset(k, ep)))
    vstep = jax.vmap(lambda k, s, a: env.step(k, s, a, ep))

    def extract(state):
        lava = state.map[jnp.arange(episodes), state.player_position[:, 0], state.player_position[:, 1]] == BlockType.LAVA.value
        return {"achievements": state.achievements, "inventory": jnp.stack([getattr(state.inventory, name) for name in INVENTORY], -1),
                "health": state.player_health, "in_lava": lava, "timestep": state.timestep}

    obs, state = vreset(jnp.asarray(keys["reset_keys"]))
    initial = {k: np.asarray(v) for k, v in extract(state).items()}
    assert not initial["achievements"].any() and not initial["inventory"].any()
    assert np.all(initial["health"] == 9) and np.all(initial["timestep"] == 0) and not initial["in_lava"].any()
    assert obs.shape == (episodes, 63, 63, 3)
    carry = (state, agent.to_u8(obs), jnp.zeros((episodes, agent.PHID)),
             jnp.zeros(episodes, jnp.int32), jnp.ones(episodes, bool), jnp.full((episodes,), jnp.nan))

    @nnx.jit
    def rollout_chunk(policy, carry, env_keys, action_keys):
        def body(c, ks):
            state, pixels, g, prev_a, active, terminal_score = c
            logits, _, next_g = policy.step(pixels, prev_a, g, jnp.zeros((episodes, 0)))
            action = jax.vmap(lambda k, l: jax.random.categorical(k, l))(ks[1], logits).astype(jnp.int32)
            next_obs, candidate, reward, done, info = vstep(ks[0], state, action)
            state2 = jax.tree.map(lambda old, new: freeze_inactive(old, new, active, jnp), state, candidate)
            obs2 = freeze_inactive(pixels, agent.to_u8(next_obs), active, jnp)
            g2 = freeze_inactive(g, next_g, active, jnp)
            pa2 = jnp.where(active, action, prev_a)
            done2 = active & done
            score2 = jnp.where(done2, info["score"], terminal_score)
            fields = extract(state2)
            record = {"actions": jnp.where(active, action, -1).astype(jnp.int8),
                      "rewards": jnp.where(active, reward, 0.), "native_done": done2,
                      "active_before": active, "alive_after": (fields["health"] > 0) & ~fields["in_lava"],
                      "policy_logits_finite": jnp.isfinite(logits).all(-1),
                      "achievements": fields["achievements"], "health": fields["health"].astype(jnp.int16),
                      "in_lava": fields["in_lava"]}
            return (state2, obs2, g2, pa2, active & ~done, score2), record
        return jax.lax.scan(body, carry, (env_keys, action_keys))

    records = []
    for start in range(0, horizon, chunk):
        stop = min(start + chunk, horizon)
        carry, record = rollout_chunk(ac, carry,
            jnp.asarray(keys["environment_step_keys"][:, start:stop].swapaxes(0, 1)),
            jnp.asarray(keys["policy_action_keys"][:, start:stop].swapaxes(0, 1)))
        record = {k: np.asarray(v).swapaxes(0, 1) for k, v in record.items()}
        records.append(record)
        remaining = int(np.asarray(carry[4]).sum())
        if start == 0 or stop % 512 == 0 or remaining == 0:
            print(json.dumps({"native_eval_step": stop, "active_episodes": remaining, "elapsed_s": round(time.time() - started, 1)}), flush=True)
        if not remaining:
            break
    raw = {k: np.concatenate([r[k] for r in records], axis=1) for k in records[0]}
    endpoint = {k: np.asarray(v) for k, v in extract(carry[0]).items()}
    endpoint["native_terminal_info_score"] = np.asarray(carry[5])
    # Save evidence before auditing: a failed invariant must remain diagnosable.
    arrays = {**raw, **{"initial_" + k: v for k, v in initial.items()},
              **{"endpoint_" + k: v for k, v in endpoint.items()}}
    np.savez_compressed(out / "episodes_raw.npz", **arrays)
    try:
        rows, summary, first = aggregate_traces(raw, initial, endpoint, horizon=horizon)
    except Exception as error:
        (out / "audit_failure.json").write_text(json.dumps({"reportable": False, "error": repr(error)}) + "\n")
        raise
    np.save(out / "first_acquisition_steps.npy", first, allow_pickle=False)
    with (out / "episodes.jsonl").open("x") as stream:
        for i, row in enumerate(rows):
            row.update(policy_checkpoint_sha256=provenance["policy_checkpoint_sha256"],
                       policy_training_seed=provenance["policy_training_seed"],
                       world_model_training_seed=provenance["world_model_training_seed"],
                       world_model_checkpoint_sha256=provenance["world_model_checkpoint_sha256"],
                       eval_seed=eval_seed, reset_key=keys["reset_keys"][i].tolist(), provenance_file="provenance.json")
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    summary.update(status="smoke_complete" if smoke else "complete", reportable=not smoke,
                   policy_checkpoint_sha256=provenance["policy_checkpoint_sha256"],
                   checkpoint_meta=provenance["checkpoint_meta"], eval_seed=eval_seed, horizon=horizon,
                   elapsed_seconds=time.time() - started,
                   artifacts_sha256={name: sha256(out / name) for name in (
                       "provenance.json", "episodes_raw.npz", "episodes.jsonl", "first_acquisition_steps.npy")})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, allow_nan=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--chunk", type=int, default=64, help="Execution chunk only; does not change RNG or episode horizon")
    parser.add_argument("--keys", type=Path, help="Optional shared exact key archive, validated before reuse")
    parser.add_argument("--smoke", action="store_true", help="Explicitly nonreportable pipeline smoke; permits partial budgets and smaller evaluation")
    parser.add_argument("--episodes", type=int, default=EPISODES, help="Frozen at 256 unless --smoke")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="Frozen at 10000 unless --smoke; native max_timesteps always remains 10000")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.out, eval_seed=args.eval_seed, chunk=args.chunk, keys_path=args.keys,
                        smoke=args.smoke, episodes=args.episodes, horizon=args.horizon)


if __name__ == "__main__":
    main()
