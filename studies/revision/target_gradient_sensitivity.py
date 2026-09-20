"""Descriptive state/nuisance sensitivity of frozen encoders on valid renders.

No training or policy input is modified. Each comparison uses an original-world
render: a valid action-sized position displacement with the same gray background,
or the same position with an independent gray background. These are controlled
pairs of valid images, not consecutive trajectories. Distances are compared
only within one encoder and are not an information measure.
"""
import argparse
import hashlib
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import train as T
from studies.redcube import env as R


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--eval-data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=1024)
    p.add_argument("--seed", type=int, default=19418)
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()
    if args.count <= 0 or args.batch <= 0:
        raise ValueError("count and batch must be positive")
    if args.out.exists() or args.out.with_suffix(".npz").exists():
        raise ValueError("refusing to overwrite an existing sensitivity artifact")
    with np.load(args.eval_data) as archive:
        meta = json.loads(str(archive["meta"]))
        positions, winds = archive["pos"], archive["wind"]
    if meta["config"] not in ("gray_frame_still", "gray_frame_wind"):
        raise ValueError("this prospective sensitivity only covers the two declared gray-frame worlds")
    expected_spec = {"noise": "gray", "per_episode": False, "wind": meta["config"] == "gray_frame_wind"}
    if meta.get("split") != "eval" or meta.get("env") != "redcube" or meta.get("spec") != expected_spec:
        raise ValueError("the declared sensitivity requires the original independent redcube evaluation archive")
    spec = R.Spec(**meta["spec"])
    rng = np.random.default_rng(args.seed)
    n, e = positions.shape[:2]
    if args.count > n * e:
        raise ValueError("more requested contexts than archived frames")
    flat_ids = rng.choice(n * e, args.count, replace=False)
    ids = np.stack(np.unravel_index(flat_ids, (n, e)), axis=1)
    pos = positions[ids[:, 0], ids[:, 1]].astype(np.int32)
    wind = winds[ids[:, 0], ids[:, 1]].astype(np.int32)
    axes = rng.integers(0, 2, args.count)
    signed_steps = rng.choice(np.array([-R.STEP, R.STEP], np.int32), args.count)
    moved_coordinate = pos[np.arange(args.count), axes] + signed_steps
    signed_steps = np.where((moved_coordinate < R.SAFE_LO) | (moved_coordinate > R.SAFE_HI),
                            -signed_steps, signed_steps)
    delta = np.zeros_like(pos)
    delta[np.arange(args.count), axes] = signed_steps
    shifted = pos + delta
    if np.any((shifted < R.SAFE_LO) | (shifted > R.SAFE_HI)):
        raise ValueError("natural displacement left the live-state domain")
    keys = jax.random.split(jax.random.PRNGKey(args.seed), 2 * args.count).reshape(2, args.count, 2)
    state = R.State(jnp.asarray(pos), jnp.asarray(wind), jnp.zeros((args.count, 2), jnp.uint32),
                    jnp.zeros(args.count, jnp.int32), jnp.zeros(args.count, jnp.int32))
    render = jax.jit(jax.vmap(lambda s, key: R.render(spec, s, key)))
    frames = (render(state, keys[0]), render(state._replace(pos=jnp.asarray(shifted)), keys[0]),
              render(state, keys[1]))
    bundle, model_meta = T.load_model(args.model)
    if model_meta.get("config_label") != meta["config"]:
        raise ValueError("checkpoint world label differs from the evaluation archive")
    if int(bundle.model.latent_dim) != 512:
        raise ValueError("the declared diagnostic requires the original 512-coordinate model")

    @nnx.jit
    def encode(model, x):
        return model.encode(x[:, None])[:, 0]

    latents = [np.concatenate([np.asarray(encode(bundle.model, frame[i:i + args.batch]))
                               for i in range(0, args.count, args.batch)]) for frame in frames]
    state_distance = np.square(latents[1].astype(np.float64) - latents[0]).sum(axis=1)
    nuisance_distance = np.square(latents[2].astype(np.float64) - latents[0]).sum(axis=1)
    numerator, denominator = float(state_distance.mean()), float(nuisance_distance.mean())
    report = {
        "config": meta["config"], "count": args.count, "seed": args.seed,
        "position_displacement_pixels": R.STEP, "mean_state_distance_sq": numerator,
        "mean_nuisance_distance_sq": denominator,
        "state_to_nuisance_distance_ratio": numerator / denominator if denominator > 1e-20 else None,
        "denominator_too_small": denominator <= 1e-20,
        "interpretation": "within-encoder descriptive sensitivity, not information or control sufficiency",
        "model_cfg": model_meta.get("cfg"), "model_sha256": sha256(args.model),
        "eval_data_sha256": sha256(args.eval_data),
        "source_sha256": {str(path): sha256(path) for path in (Path(__file__), Path(T.__file__), Path(R.__file__))},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(args.out.with_suffix(".npz"), indices=ids, positions=pos, displacements=delta,
                        background_keys=np.asarray(keys), state_distance_sq=state_distance,
                        nuisance_distance_sq=nuisance_distance,
                        latent_original=latents[0], latent_position_shift=latents[1], latent_nuisance_shift=latents[2])
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
