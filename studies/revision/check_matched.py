"""Independent data/alignment and updater parity checks before matched PPO runs."""
import argparse
import ast
import inspect
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import train as T
from studies.revision import matched_ppo as M
from studies.stoch import env as S
from studies.stoch import rl as P


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.data) as archive:
        data = {name: archive[name] for name in
                ("frames", "actions", "rewards", "dones", "pos", "wind", "episode", "wall_x", "gap")}
    data["actions"] = data["actions"].astype(np.int32)
    steps = M.episode_steps(data["episode"], data["dones"])
    spec = S.CONFIGS["gate"]
    result = {"reconstruction": M.validate_reconstruction(spec, data, steps)}
    n, e = data["actions"].shape
    resets, uniform = P.start_tables(data["dones"], n, e, P.CTX)
    old_rng, new_rng = np.random.default_rng(11), np.random.default_rng(11)
    for _ in range(5):
        old = P.sample_starts(old_rng, data["frames"], data["actions"], resets, uniform, 32, 0.5)
        new = M.sample_starts(new_rng, data, resets, uniform, steps, 32, 0.5)
        for original, candidate in zip(old, new[:3]):
            np.testing.assert_array_equal(np.asarray(original), np.asarray(candidate))
    result["start_sampler"] = {"identical_batches": 5, "batch": 32}
    _, _, _, state, ids = M.sample_starts(new_rng, data, resets, uniform, steps, 1024, 0.5)
    t, env = ids[:, 2], ids[:, 1]
    actions = jnp.asarray(data["actions"][t, env])
    observed, next_state, rewards, dones = jax.jit(jax.vmap(S.make(spec)[1]))(
        state, actions, jax.random.split(jax.random.PRNGKey(89), len(t)))
    # Existing walls move deterministically. A newly spawned random gap is the
    # only transition randomness in this texture-free, wind-free gate.
    no_spawn = (np.asarray(state.wall_x) != S.NO_WALL) | ((np.asarray(state.step) + 1) % S.WALL_PERIOD != 0)
    np.testing.assert_array_equal(np.asarray(rewards)[no_spawn], data["rewards"][t, env][no_spawn])
    np.testing.assert_array_equal(np.asarray(dones)[no_spawn], data["dones"][t, env][no_spawn])
    continuing = no_spawn & ~np.asarray(dones)
    np.testing.assert_array_equal(np.asarray(observed)[continuing], data["frames"][t + 1, env][continuing])
    for field in ("pos", "wind", "episode", "wall_x", "gap"):
        np.testing.assert_array_equal(np.asarray(getattr(next_state, field))[continuing],
                                      data[field][t + 1, env][continuing])
    np.testing.assert_array_equal(np.asarray(next_state.step)[continuing], steps[t + 1, env][continuing])
    result["deterministic_replay"] = {"signals_checked": int(no_spawn.sum()), "next_frames_and_states": int(continuing.sum())}
    bundle, _ = T.load_model(args.model)
    cfg = M.parser().parse_args(["--arm", "learned", "--model", str(args.model), "--data", str(args.data),
                                "--out", str(args.out.parent / "unused"), "--model-seed", "0",
                                "--batch", "32", "--hor", "4", "--chunk", "4", "--mb", "16"])
    ac1 = P.AC(nnx.Rngs(2), "latent", int(bundle.model.latent_dim))
    ac2 = P.AC(nnx.Rngs(2), "latent", int(bundle.model.latent_dim))
    optimizer = lambda ac: nnx.Optimizer(ac, optax.chain(optax.clip_by_global_norm(1.0),
                                                       optax.adamw(cfg.lr, weight_decay=0.01)), wrt=nnx.Param)
    opt1, opt2 = optimizer(ac1), optimizer(ac2)
    frames, incoming, use_reset, _, _ = M.sample_starts(new_rng, data, resets, uniform, steps, 32, 0.5)
    collector = P.make_jepa_dream(bundle, 32, 4, cfg.gamma, cfg.lam)
    batch, _ = collector(ac1, frames, incoming, use_reset, jax.random.PRNGKey(29))
    # Execute only the existing local nested updater definition, supplying its
    # lexical arguments; this avoids creating a second handwritten reference.
    outer = ast.parse(inspect.getsource(P.ppo_dream)).body[0]
    original_node = next(node for node in outer.body if isinstance(node, ast.FunctionDef) and node.name == "ppo_step")
    globals_ = dict(vars(P), a=cfg, NCH=1, NB=2)
    exec(compile(ast.Module(body=[original_node], type_ignores=[]), str(Path(P.__file__)), "exec"), globals_)
    old_update = globals_["ppo_step"]
    new_update = M.make_update(cfg)
    old_loss = np.asarray(old_update(ac1, opt1, batch, jnp.asarray(0), jnp.asarray(0)))
    new_loss = np.asarray(new_update(ac2, opt2, batch, jnp.asarray(0), jnp.asarray(0)))
    np.testing.assert_allclose(new_loss, old_loss, rtol=1e-6, atol=1e-7)
    max_parameter_error = 0.0
    for old, new in zip(jax.tree.leaves(nnx.state(ac1, nnx.Param)), jax.tree.leaves(nnx.state(ac2, nnx.Param))):
        max_parameter_error = max(max_parameter_error, float(np.max(np.abs(np.asarray(old) - np.asarray(new)))))
        np.testing.assert_allclose(np.asarray(new), np.asarray(old), rtol=1e-6, atol=1e-7)
    result["ppo_update"] = {"old_loss": old_loss.tolist(), "new_loss": new_loss.tolist(),
                            "max_parameter_absolute_error": max_parameter_error}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
