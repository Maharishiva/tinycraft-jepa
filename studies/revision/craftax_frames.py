"""Save genuine Craftax start/goal frames from the frozen evaluation protocol.

CPU-only illustration, with no model loading or inference. Select the first
episodes whose target displacement reaches a predeclared threshold; never use
model success/failure for scene selection. Regenerated coordinates must match
the completed quantitative evaluation before any PNG is written.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--min-displacement", type=int, default=2)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("output directory exists; use a fresh illustration path")
    import imageio.v3 as iio
    import jax
    import jax.numpy as jnp
    import numpy as np
    from craftax.craftax_env import make_craftax_env_from_name

    report = json.loads(args.eval_report.read_text())
    run = next(r for r in report["goal_runs"] if r["rows"]["noop"][0]["eval_seed"] == args.seed)
    selected = [i for i, row in enumerate(run["rows"]["noop"])
                if row["goal_displacement_l1"] >= args.min_displacement][:args.count]
    if len(selected) < args.count:
        raise ValueError("not enough qualifying episodes for the declared selection")
    n, warmup, horizon = (report["arguments"][k] for k in ("envs", "warmup", "horizon"))
    env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
    params = env.default_params
    reset = jax.jit(jax.vmap(lambda k: env.reset(k, params)))
    step = jax.jit(jax.vmap(lambda k, s, a: env.step(k, s, a, params)))

    def key(stream, t=0):
        return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(args.seed), stream), t)

    obs, state = reset(jax.random.split(key(0), n))
    reset_frames = np.asarray(jnp.clip(obs * 255.0, 0, 255).astype(jnp.uint8))
    for t in range(warmup - 1):
        actions = jax.random.randint(key(1, t), (n,), 0, 17)
        obs, state, _, _, _ = step(jax.random.split(key(2, t), n), state, actions)
    start_positions = np.asarray(state.player_position)
    start_frames = np.asarray(jnp.clip(obs * 255.0, 0, 255).astype(jnp.uint8))
    reference = jax.random.randint(key(3), (n, horizon), 0, 17)
    for t in range(horizon):
        obs, state, _, _, _ = step(jax.random.split(key(4, t), n), state, reference[:, t])
    goal_positions = np.asarray(state.player_position)
    goal_frames = np.asarray(jnp.clip(obs * 255.0, 0, 255).astype(jnp.uint8))
    np.testing.assert_array_equal(start_positions, run["start_positions"])
    np.testing.assert_array_equal(goal_positions, run["target_positions"])
    np.testing.assert_array_equal(np.asarray(reference), run["reference_actions"])
    args.out_dir.mkdir(parents=True)
    meta = {"purpose": "actual environment illustration, not additional quantitative evaluation",
            "eval_report": str(args.eval_report),
            "eval_report_sha256": hashlib.sha256(args.eval_report.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "selection": "first episode indices with goal displacement >= threshold, no model outcomes used",
            "seed": args.seed, "min_displacement": args.min_displacement, "selected_episodes": selected,
            "warmup_frames": warmup, "horizon": horizon, "device": [str(d) for d in jax.devices()],
            "craftax_version": importlib.metadata.version("craftax"),
            "coordinate_and_action_regeneration": "exact match to completed GPU evaluation",
            "scenes": []}
    for episode in selected:
        scene = {"episode": episode, "start_position": start_positions[episode].tolist(),
                 "goal_position": goal_positions[episode].tolist(),
                 "displacement_l1": int(np.abs(goal_positions[episode] - start_positions[episode]).sum()),
                 "goal_generating_actions": np.asarray(reference[episode]).tolist(), "frames": {}}
        for label, frames in (("reset", reset_frames), ("start", start_frames), ("goal", goal_frames)):
            path = args.out_dir / f"seed{args.seed}_episode{episode:03d}_{label}.png"
            iio.imwrite(path, frames[episode])
            scene["frames"][label] = {"path": path.name, "shape": list(frames[episode].shape),
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        meta["scenes"].append(scene)
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
