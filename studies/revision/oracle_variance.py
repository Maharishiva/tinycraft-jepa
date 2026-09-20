"""Exact finite-branch latent-goal variance diagnostic; never trains a model.

See research/oracle_variance_design.md. Oracle state is used only to enumerate
counterfactual observations. The learned predictor, decoder, heads and policy
are never evaluated. A ranking difference concerns expected latent goal cost,
not reward, survival or physical-control superiority.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np


MOVES = np.asarray([[-3, 0], [0, -3], [3, 0], [0, 3]], np.int32)
GOALS = np.asarray(list(itertools.product([8, 18, 28, 38, 48], repeat=2)), np.int32)
INNOVATIONS = np.asarray(list(itertools.product([-1, 0, 1], repeat=2)), np.int32)
WIND_PROBABILITIES = np.outer([0.1, 0.8, 0.1], [0.1, 0.8, 0.1]).reshape(9)
CONDITIONS = ("off_wind", "gray_episode_wind")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def branch_law(wind, pos, moves=MOVES):
    """Return wind[N,9,2], position[N,A,9,2], and original nine probabilities."""
    wind, pos, moves = np.asarray(wind), np.asarray(pos), np.asarray(moves)
    if wind.shape != pos.shape or wind.ndim != 2 or wind.shape[1] != 2:
        raise ValueError("position and wind must have shape (N,2)")
    if moves.ndim != 2 or moves.shape[1] != 2 or np.any(np.abs(wind) > 2):
        raise ValueError("invalid moves or current wind")
    next_wind = np.clip(wind[:, None] + INNOVATIONS[None], -2, 2)
    next_pos = pos[:, None, None] + moves[None, :, None] + next_wind[:, None]
    return next_wind, next_pos, WIND_PROBABILITIES.copy()


def safe_all_branches(next_pos, step, safe_lo=2, safe_hi=55, max_steps=128):
    next_pos, step = np.asarray(next_pos), np.asarray(step)
    if next_pos.ndim != 4 or next_pos.shape[-1] != 2 or step.shape != (len(next_pos),):
        raise ValueError("invalid branch position/time shapes")
    border_safe = ((next_pos >= safe_lo) & (next_pos <= safe_hi)).all(axis=(1, 2, 3))
    return border_safe & (step + 1 < max_steps)


def infer_wind(history_pos, outgoing_actions, moves=MOVES):
    """Current wind from a caller-verified no-reset history and its last action."""
    history_pos, outgoing_actions = np.asarray(history_pos), np.asarray(outgoing_actions)
    if history_pos.ndim != 3 or history_pos.shape[1] < 2 or history_pos.shape[-1] != 2:
        raise ValueError("history positions must have shape (N,H>=2,2)")
    if outgoing_actions.shape != (history_pos.shape[0], history_pos.shape[1] - 1):
        raise ValueError("one outgoing action is needed between each pair of frames")
    return history_pos[:, -1] - history_pos[:, -2] - np.asarray(moves)[outgoing_actions[:, -1]]


def costs(branch_z, goal_z, weights, block=32):
    """Exact branch costs and centered variance, in coordinate-mean units."""
    z, goal, q = (np.asarray(x, np.float64) for x in (branch_z, goal_z, weights))
    if z.ndim != 4 or goal.ndim != 3 or goal.shape[0] not in (1, len(z)) or goal.shape[-1] != z.shape[-1]:
        raise ValueError("expected branch (N,A,K,D) and goal (N or1,G,D)")
    if q.shape != (z.shape[2],) or np.any(q < 0) or not np.isclose(q.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("branch probabilities must be nonnegative and sum to one")
    if not all(np.isfinite(x).all() for x in (z, goal, q)) or block < 1:
        raise ValueError("nonfinite features or invalid block size")
    q = q / q.sum()  # remove only floating-point normalization error
    origin = z[:, :, 0]
    mu = origin + np.einsum("k,nakd->nad", q, z - origin[:, :, None])
    variance = np.einsum("k,nak->na", q, np.square(z - mu[:, :, None]).mean(axis=-1))
    shape = (len(z), z.shape[1], goal.shape[1])
    mean_cost, exact_cost = np.empty(shape), np.empty(shape)
    for begin in range(0, len(z), block):
        end = min(begin + block, len(z))
        g = goal if len(goal) == 1 else goal[begin:end]
        mean_cost[begin:end] = np.square(mu[begin:end, :, None] - g[:, None]).mean(axis=-1)
        direct = np.zeros_like(mean_cost[begin:end])
        for k, probability in enumerate(q):
            direct += probability * np.square(z[begin:end, :, k, None] - g[:, None]).mean(axis=-1)
        exact_cost[begin:end] = direct
    residual = exact_cost - (mean_cost + variance[:, :, None])
    return {"mean": mean_cost, "variance": variance, "exact": exact_cost,
            "mean_latent": mu, "identity_residual": residual}


def decision_metrics(mean, exact, tolerance_factor=1e-10):
    mean, exact = np.asarray(mean, np.float64), np.asarray(exact, np.float64)
    if mean.shape != exact.shape or mean.ndim != 3 or not np.isfinite(mean).all() or not np.isfinite(exact).all():
        raise ValueError("costs must be finite paired arrays (N,A,G)")
    if tolerance_factor < 0:
        raise ValueError("negative tolerance")
    scale = np.maximum(1.0, np.maximum(np.abs(mean).max(axis=1), np.abs(exact).max(axis=1)))
    tolerance = tolerance_factor * scale
    mean_best, exact_best = mean.min(axis=1), exact.min(axis=1)
    mean_set = mean <= mean_best[:, None] + tolerance[:, None]
    exact_set = exact <= exact_best[:, None] + tolerance[:, None]
    regret = exact - exact_best[:, None]
    count = mean_set.sum(axis=1)
    return {"mean_optimal": mean_set, "exact_optimal": exact_set,
            "robust_disagreement": ~(mean_set & exact_set).any(axis=1),
            "regret_min": np.where(mean_set, regret, np.inf).min(axis=1),
            "regret_max": np.where(mean_set, regret, -np.inf).max(axis=1),
            "regret_uniform": np.where(mean_set, regret, 0).sum(axis=1) / count,
            "raw_argmin_disagreement": mean.argmin(axis=1) != exact.argmin(axis=1),
            "mean_gap": np.sort(mean, axis=1)[:, 1] - mean_best,
            "exact_gap": np.sort(exact, axis=1)[:, 1] - exact_best,
            "tolerance": tolerance}


def select_contexts(dones, history, count, seed):
    dones = np.asarray(dones, bool)
    if dones.ndim != 2 or len(dones) < history - 1 or count < 1:
        raise ValueError("invalid collection size")
    times = np.arange(history - 1, len(dones) + 1)
    prefix = np.concatenate([np.zeros((1, dones.shape[1]), np.int32), np.cumsum(dones, axis=0)], axis=0)
    valid = (prefix[times] - prefix[times - history + 1]) == 0
    row, env = np.nonzero(valid)
    ids = np.stack([times[row], env], axis=1).astype(np.int32)
    if not len(ids):
        raise ValueError("no valid eight-frame contexts")
    chosen = np.random.default_rng(seed).permutation(len(ids))[:min(count, len(ids))]
    return ids[chosen], int(len(ids))


def summarize(values, streams, bootstrap, environment_ids):
    values = np.asarray(values, np.float64)
    if values.ndim == 2:
        values = values.mean(axis=1)  # uniform goals within every context
    if values.ndim != 1 or len(values) != len(streams) or not np.isfinite(values).all():
        raise ValueError("invalid context statistics")
    codes = np.searchsorted(environment_ids, streams)
    counts = np.bincount(codes, minlength=len(environment_ids))
    sums = np.bincount(codes, weights=values, minlength=len(environment_ids))
    numerator, denominator = bootstrap @ sums, bootstrap @ counts
    draws = np.divide(numerator, denominator, out=np.full(len(bootstrap), np.nan), where=denominator > 0)
    valid = np.isfinite(draws)
    interval = np.quantile(draws[valid], [0.025, 0.975]).tolist() if (counts > 0).sum() >= 2 and valid.sum() >= 2 else None
    return {"mean": float(values.mean()), "ci95": interval, "contexts": len(values),
            "nonempty_streams": int((counts > 0).sum()), "valid_bootstrap_replicates": int(valid.sum()),
            "context_quantiles_05_50_95": np.quantile(values, [0.05, 0.5, 0.95]).tolist()}


def collect_states(spec, envs, steps, seed):
    """Original random-policy simulator, retaining texture/time without images.

    Both admitted conditions have deterministic rendering given stored state;
    selected histories are rendered later through the original renderer.
    """
    import jax
    import jax.numpy as jnp
    from studies.redcube import env as R
    reset, step = R.batched(spec)
    key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    _, state = reset(jax.random.split(reset_key, envs), jnp.zeros(envs, jnp.int32))

    @jax.jit
    def collect(state, keys):
        def body(state, key):
            action_key, transition_key = jax.random.split(key)
            action = jax.random.randint(action_key, (envs,), 0, R.N_ACTIONS)
            _, nxt, _, done, _ = step(state, action, transition_key)
            return nxt, (state, action, done)
        return jax.lax.scan(body, state, keys)

    final, (states, actions, dones) = collect(state, jax.random.split(key, steps))
    arrays = {name: np.asarray(jnp.concatenate([getattr(states, name), getattr(final, name)[None]]))
              for name in R.State._fields}
    arrays.update(actions=np.asarray(actions, np.int32), dones=np.asarray(dones, bool))
    if not np.array_equal(arrays["episode"][1:] != arrays["episode"][:-1], arrays["dones"]):
        raise ValueError("fresh collection reset metadata disagree")
    return arrays


def module_state_hash(model):
    import flax.nnx as nnx
    import jax
    digest = hashlib.sha256()
    for value in jax.tree.leaves(nnx.state(model, nnx.Param, nnx.BatchStat)):
        digest.update(np.ascontiguousarray(np.asarray(value)).view(np.uint8))
    return digest.hexdigest()


def runtime_helpers(model, spec, batch):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from studies.redcube import env as R

    @jax.jit
    def render(pos, texture):
        n = len(pos)
        states = R.State(pos, jnp.zeros((n, 2), jnp.int32), texture,
                         jnp.zeros(n, jnp.int32), jnp.zeros(n, jnp.int32))
        return jax.vmap(lambda state: R.render(spec, state, jax.random.PRNGKey(0)))(states)

    @nnx.jit
    def encode(model, frames):
        return model.encode(frames[:, None], train=False)[:, 0]

    def encode_states(pos, texture):
        if len(pos) != len(texture) or not len(pos):
            raise ValueError("empty/misaligned states")
        outputs = []
        for begin in range(0, len(pos), batch):
            p, b = pos[begin:begin + batch], texture[begin:begin + batch]
            actual = len(p)
            if actual < batch:
                p = np.pad(p, ((0, batch - actual), (0, 0)), mode="edge")
                b = np.pad(b, ((0, batch - actual), (0, 0)), mode="edge")
            frames = render(jnp.asarray(p, jnp.int32), jnp.asarray(b, jnp.uint32))
            outputs.append(np.asarray(encode(model, frames))[:actual])
        z = np.concatenate(outputs)
        if not np.isfinite(z).all():
            raise ValueError("nonfinite frozen features")
        return z

    def parity(pos, texture):
        # Match flattened frame batch for flat and eight-frame context calls.
        count = min(len(pos), batch)
        p = np.pad(pos[:count], ((0, batch - count), (0, 0)), mode="edge")
        b = np.pad(texture[:count], ((0, batch - count), (0, 0)), mode="edge")
        frames = render(jnp.asarray(p, jnp.int32), jnp.asarray(b, jnp.uint32))
        direct = np.asarray(encode(model, frames))
        @nnx.jit
        def encode_context(model, frames):
            return model.encode(frames, train=False)
        contextual = np.asarray(encode_context(model, frames.reshape((-1, 8) + frames.shape[1:]))).reshape(direct.shape)
        permutation = np.random.default_rng(9187).permutation(batch)
        permuted = np.asarray(encode(model, frames[permutation]))[np.argsort(permutation)]
        context_diff = float(np.max(np.abs(direct - contextual)))
        permutation_diff = float(np.max(np.abs(direct - permuted)))
        if not np.array_equal(direct, contextual) or not np.array_equal(direct, permuted):
            raise ValueError(f"same-shape encoder parity failed: context={context_diff}, permutation={permutation_diff}")
        return {"encoder_frame_batch": batch, "reference_context_batch": batch // 8,
                "context_max_abs_difference": context_diff, "permutation_max_abs_difference": permutation_diff}

    return encode_states, parity


def simulator_branch_check(spec, current, next_pos, next_wind, seed):
    import jax
    import jax.numpy as jnp
    from studies.redcube import env as R
    count = min(len(next_pos), 64)
    states = R.State(*(jnp.asarray(current[name][:count]) for name in R.State._fields))
    repeated = jax.tree.map(lambda value: jnp.repeat(value, 4, axis=0), states)
    actions = jnp.tile(jnp.arange(4), count)
    keys = jax.random.split(jax.random.PRNGKey(seed), count * 4)
    _, actual, rewards, dones = jax.jit(jax.vmap(R.make(spec)[1]))(repeated, actions, keys)
    actual_pos = np.asarray(actual.pos).reshape(count, 4, 2)
    actual_wind = np.asarray(actual.wind).reshape(count, 4, 2)
    matches = (next_wind[:count, None] == actual_wind[:, :, None]).all(-1)
    if not matches.any(-1).all():
        raise ValueError("simulator produced a wind outcome outside enumerated support")
    paired_pos = (next_pos[:count] == actual_pos[:, :, None]).all(-1)
    if not (paired_pos & matches).any(-1).all() or np.any(np.asarray(rewards) != 0) or np.asarray(dones).any():
        raise ValueError("exact branch positions/safe outcomes disagree with simulator")
    return {"sampled_simulator_steps": count * 4, "all_in_enumerated_support": True,
            "all_safe_rewards_zero": True}


def run(args):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import train as T
    import jax
    from studies.redcube import env as R
    from dataclasses import asdict

    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("refusing to overwrite nonempty output")
    if args.encoder_batch < 8 or args.encoder_batch % 8 or min(args.contexts, args.envs, args.steps, args.bootstrap_reps) < 1:
        raise ValueError("invalid dimensions; encoder batch must be a positive multiple of eight")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if (R.SAFE_LO, R.SAFE_HI, R.MAX_STEPS, R.WIND_MAX, R.WIND_STAY) != (2, 55, 128, 2, 0.8):
        raise ValueError("environment constants differ from the declared protocol")
    if not np.array_equal(R.MOVES, MOVES):
        raise ValueError("native action ordering changed")
    model_hash = sha256(args.model)
    if args.expected_model_sha256 and args.expected_model_sha256 != model_hash:
        raise ValueError("checkpoint SHA256 mismatch")
    bundle, model_meta = T.load_model(args.model)
    model = bundle.model
    if model.latent_dim != 512 or model.n_actions != 4 or model.history != 8:
        raise ValueError("requires the declared 512-dimensional/four-action/eight-frame checkpoint")
    before = module_state_hash(model)
    spec = R.CONFIGS[args.condition]
    manifest = {"status": "collecting", "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "spec": asdict(spec), "model_sha256": model_hash, "model_meta": model_meta,
        "frozen_state_sha256_before": before, "sources": {str(p): sha256(p) for p in
            (Path(__file__), Path(T.__file__), Path(R.__file__))},
        "interpretation": "oracle physical-state/texture conditional latent cost; no fitted predictor or reward/control claim",
        "goals": GOALS.tolist(), "innovations": INNOVATIONS.tolist(), "branch_weights": WIND_PROBABILITIES.tolist(),
        "compute_dtype": str(T.CDT), "cost_dtype": "float64", "jax_version": jax.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": [str(device) for device in jax.devices()]}
    write_json(args.out / "manifest.json", manifest)
    collection = collect_states(spec, args.envs, args.steps, args.seed)
    np.savez_compressed(args.out / "collection.npz", **collection)
    ids, eligible = select_contexts(collection["dones"], 8, args.contexts, args.selection_seed)
    times, env = ids.T
    ti = times[:, None] - 7 + np.arange(8)
    ei = np.broadcast_to(env[:, None], ti.shape)
    histories = {name: collection[name][ti, ei] for name in R.State._fields}
    history_actions = collection["actions"][ti[:, :-1], ei[:, :-1]]
    current = {name: histories[name][:, -1] for name in R.State._fields}
    inferred = infer_wind(histories["pos"], history_actions)
    if not np.array_equal(inferred, current["wind"]):
        raise ValueError("wind inferred from no-reset history differs from current state")
    next_wind, next_pos, probabilities = branch_law(current["wind"], current["pos"])
    safe = safe_all_branches(next_pos, current["step"])
    border = ((next_pos >= 2) & (next_pos <= 55)).all(axis=(1, 2, 3))
    cap = current["step"] + 1 < 128
    np.savez_compressed(args.out / "contexts.npz", ids=ids, safe=safe, history_actions=history_actions,
                        goals=GOALS, **{f"history_{k}": v for k, v in histories.items()})
    selection = {"eligible_past_contexts": eligible, "sampled_contexts": len(ids), "admitted_contexts": int(safe.sum()),
        "border_excluded": int((~border).sum()), "cap_excluded": int((~cap).sum()),
        "selection_fraction": float(safe.mean()), "ids_sha256": array_hash(ids),
        "admitted_ids_sha256": array_hash(ids[safe]), "collection_sha256": sha256(args.out / "collection.npz"),
        "contexts_sha256": sha256(args.out / "contexts.npz")}
    manifest.update(status="contexts_fixed_before_encoding", selection=selection)
    write_json(args.out / "manifest.json", manifest)
    print("[contexts] " + json.dumps(selection), flush=True)
    if not safe.any():
        manifest.update(status="empty_safe_cohort", elapsed_seconds=time.time() - started)
        write_json(args.out / "manifest.json", manifest)
        return
    current = {k: v[safe] for k, v in current.items()}
    next_wind, next_pos, probabilities = next_wind[safe], next_pos[safe], probabilities
    n = len(next_pos)
    selection["wind_states"] = [{"wind": list(w), "sampled": int((inferred == w).all(-1).sum()),
                                "admitted": int((current["wind"] == w).all(-1).sum())}
                               for w in itertools.product(range(-2, 3), repeat=2)]
    selection["admitted_position_min"] = current["pos"].min(axis=0).tolist()
    selection["admitted_position_max"] = current["pos"].max(axis=0).tolist()
    simulator_check = simulator_branch_check(spec, current, next_pos, next_wind, args.seed + 2)
    encode, check_parity = runtime_helpers(model, spec, args.encoder_batch)
    parity = check_parity(current["pos"], current["texture"])
    branch_pos = next_pos.reshape(-1, 2)
    branch_texture = np.repeat(current["texture"], 4 * 9, axis=0)
    branch_z = encode(branch_pos, branch_texture).reshape(n, 4, 9, 512)
    if args.condition == "off_wind":
        goal_z = encode(GOALS, np.repeat(current["texture"][:1], len(GOALS), axis=0))[None]
    else:
        goal_pos = np.broadcast_to(GOALS[None], (n, len(GOALS), 2)).reshape(-1, 2)
        goal_texture = np.repeat(current["texture"], len(GOALS), axis=0)
        goal_z = encode(goal_pos, goal_texture).reshape(n, len(GOALS), 512)
    print("[encoded] " + json.dumps({"contexts": n, "branch_frames": len(branch_pos),
        "goal_frames": int(np.prod(goal_z.shape[:2])), "parity": parity}), flush=True)
    latent = costs(branch_z, goal_z, probabilities, args.cost_block)
    decision = decision_metrics(latent["mean"], latent["exact"])
    if np.any(np.abs(latent["identity_residual"]) > decision["tolerance"][:, None]):
        raise ValueError("direct/decomposed expected-cost identity exceeds fixed tolerance")
    physical = costs(next_pos, GOALS[None], probabilities, args.cost_block)
    physical_decision = decision_metrics(physical["mean"], physical["exact"])
    if not np.array_equal(physical_decision["mean_optimal"], physical_decision["exact_optimal"]):
        raise ValueError("identity-position negative control disagrees")
    affine = np.asarray([[1, 2], [2, -1], [0.5, 0.25]], np.float64)
    bias = np.asarray([3, -4, 1], np.float64)
    affine_cost = costs(np.einsum("nakd,fd->nakf", next_pos, affine) + bias,
                        (GOALS @ affine.T + bias)[None], probabilities, args.cost_block)
    affine_decision = decision_metrics(affine_cost["mean"], affine_cost["exact"])
    if not np.array_equal(affine_decision["mean_optimal"], affine_decision["exact_optimal"]):
        raise ValueError("affine-position negative control disagrees")
    deterministic_q = np.eye(9, dtype=np.float64)[4]
    deterministic = costs(branch_z, goal_z, deterministic_q, args.cost_block)
    if np.any(deterministic["variance"] != 0) or decision_metrics(deterministic["mean"], deterministic["exact"])["robust_disagreement"].any():
        raise ValueError("deterministic-branch negative control failed")
    physical_exact = physical["exact"] * 2  # raw 2D pixel-squared units
    mean_actions, exact_actions = decision["mean_optimal"], decision["exact_optimal"]
    physical_mean = np.where(mean_actions, physical_exact, 0).sum(axis=1) / mean_actions.sum(axis=1)
    physical_full = np.where(exact_actions, physical_exact, 0).sum(axis=1) / exact_actions.sum(axis=1)
    derived = {"physical_cost_mean_choice": physical_mean, "physical_cost_exact_latent_choice": physical_full,
        "physical_regret_mean_choice": physical_mean - physical_exact.min(axis=1),
        "physical_regret_exact_latent_choice": physical_full - physical_exact.min(axis=1),
        "physical_difference_mean_minus_exact_latent": physical_mean - physical_full}
    all_streams = np.arange(args.envs)
    bootstrap = np.random.default_rng(args.bootstrap_seed).multinomial(args.envs, np.full(args.envs, 1 / args.envs), size=args.bootstrap_reps)
    streams = ids[safe, 1]
    summaries = {key: summarize(decision[key], streams, bootstrap, all_streams) for key in
                 ("robust_disagreement", "regret_min", "regret_max", "regret_uniform", "raw_argmin_disagreement", "mean_gap", "exact_gap")}
    summaries["action_variance_spread"] = summarize(np.ptp(latent["variance"], axis=1), streams, bootstrap, all_streams)
    summaries.update({key: summarize(value, streams, bootstrap, all_streams) for key, value in derived.items()})
    np.savez(args.out / "features.npz", branch_z=branch_z, goal_z=goal_z, branch_pos=next_pos,
             branch_wind=next_wind, weights=probabilities, goals=GOALS, admitted_ids=ids[safe])
    np.savez_compressed(args.out / "costs.npz", **{f"latent_{k}": v for k, v in latent.items()},
        **{f"decision_{k}": v for k, v in decision.items()}, **derived,
        physical_exact_cost=physical_exact, physical_variance=physical["variance"] * 2,
        bootstrap_environment_ids=all_streams, bootstrap_multiplicities=bootstrap)
    after = module_state_hash(model)
    if after != before:
        raise ValueError("frozen model parameters/statistics changed")
    result = {"condition": args.condition, "model_seed": model_meta.get("cfg", {}).get("seed"),
        "contexts": n, "goals_per_context": len(GOALS), "pairs": n * len(GOALS), "selection": selection,
        "summaries": summaries, "max_identity_residual": float(np.max(np.abs(latent["identity_residual"]))),
        "controls": {"identity_position_robust_disagreements": 0, "affine_position_robust_disagreements": 0,
                     "deterministic_branch_robust_disagreements": 0, "simulator": simulator_check},
        "parity": parity, "frozen_state_unchanged": True,
        "units": "latent coordinate-mean squared distance; physical secondary in raw pixel-squared distance",
        "uncertainty": "full collection-stream bootstrap; goals averaged within context; conditional on fixed encoder",
        "elapsed_seconds": round(time.time() - started, 3)}
    write_json(args.out / "results.json", result)
    manifest.update(status="complete", selection=selection, parity=parity,
        frozen_state_sha256_after=after, frozen_state_unchanged=True, elapsed_seconds=result["elapsed_seconds"],
        artifacts={name: sha256(args.out / name) for name in
                   ("collection.npz", "contexts.npz", "features.npz", "costs.npz", "results.json")})
    write_json(args.out / "manifest.json", manifest)
    print("[done] " + json.dumps({"condition": args.condition, "pairs": result["pairs"],
          "robust_disagreement": summaries["robust_disagreement"], "latent_regret": summaries["regret_uniform"],
          "seconds": result["elapsed_seconds"]}), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=CONDITIONS, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--expected-model-sha256")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--contexts", type=int, default=1024)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--selection-seed", type=int, default=20260924)
    p.add_argument("--encoder-batch", type=int, default=2048)
    p.add_argument("--cost-block", type=int, default=32)
    p.add_argument("--bootstrap-seed", type=int, default=20260925)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    return p


if __name__ == "__main__":
    options = parser().parse_args()
    manifest_preexisting = (options.out / "manifest.json").exists()
    try:
        run(options)
    except Exception as error:
        manifest_path = options.out / "manifest.json"
        if not manifest_preexisting and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") != "complete":
                manifest.update(status="failed", error=repr(error))
                write_json(manifest_path, manifest)
        raise
