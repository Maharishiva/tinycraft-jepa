"""Protocol-matched diagnostics for a frozen JEPA imagination pipeline.

All arms use the same policy, real dataset starts, context burn-in, horizon,
optimizer, update count and final evaluation keys. Only the source of the next
observation and reward/termination changes:

  learned       frozen JEPA dynamics; frozen learned reward/done heads
  oracle_heads  exact simulator dynamics, re-encoded; the same learned heads
  oracle_true   exact simulator dynamics, re-encoded; true reward/done

The simulator is a diagnostic transition source. Hidden simulator state is never
an input to the actor or the learned heads. Exact transitions become absorbing
after a true terminal event. The oracle_heads arm still uses its learned reward
and continuation on those absorbing states; this choice is reported explicitly.
No reward clipping, shaping, latent scaling, privileged action choice, or best
checkpoint selection is used.

Example:
 python -m studies.revision.matched_ppo --arm oracle_true --model MODEL.npz \
   --data train.npz --model-seed 0 --policy-seed 0 --out runs/revision/m0/p0/oracle_true
"""
import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import probe
import train as T
from studies.stoch import env as S
from studies.stoch import rl as P


ARMS = ("learned", "oracle_heads", "oracle_true")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def episode_steps(episodes, dones):
    """Recover within-episode time; archives begin at a genuine reset."""
    episodes, dones = np.asarray(episodes), np.asarray(dones, bool)
    if episodes.shape != (dones.shape[0] + 1, dones.shape[1]):
        raise ValueError("episode metadata must include the final frame")
    if not np.array_equal(episodes[1:] != episodes[:-1], dones):
        raise ValueError("episode boundaries disagree with recorded dones")
    if np.any(episodes[0] != 0):
        raise ValueError("state reconstruction requires an archive beginning at reset")
    row = np.arange(episodes.shape[0], dtype=np.int32)[:, None]
    reset = np.concatenate([np.ones_like(episodes[:1], bool), dones], axis=0)
    return row - np.maximum.accumulate(np.where(reset, row, 0), axis=0)


def sample_starts(rng, data, resets, uniform, steps, batch, p_reset):
    """Exactly the original mixture, with recorded state indices retained."""
    use_r = rng.random(batch) < p_reset
    ri = rng.integers(0, len(resets[0]), batch)
    ui = rng.integers(0, len(uniform[0]), batch)
    t = np.where(use_r, resets[0][ri], uniform[0][ui])
    e = np.where(use_r, resets[1][ri], uniform[1][ui])
    ti = np.where(use_r[:, None], t[:, None], t[:, None] + np.arange(P.CTX))
    ei = np.broadcast_to(e[:, None], ti.shape)
    frames = data["frames"][ti, ei]
    incoming = np.where(use_r[:, None], 0, data["actions"][np.maximum(ti - 1, 0), ei]).astype(np.int32)
    last = ti[:, -1]
    state = S.State(
        jnp.asarray(data["pos"][last, e], jnp.int32),
        jnp.asarray(data["wind"][last, e], jnp.int32),
        jnp.zeros((batch, 2), jnp.uint32),  # unused: only texture-free worlds accepted
        jnp.asarray(steps[last, e], jnp.int32),
        jnp.asarray(data["episode"][last, e], jnp.int32),
        jnp.asarray(data["wall_x"][last, e], jnp.int32),
        jnp.asarray(data["gap"][last, e], jnp.int32),
    )
    ids = np.stack([t, e, last, use_r.astype(np.int32)], axis=1).astype(np.int32)
    return jnp.asarray(frames), jnp.asarray(incoming), jnp.asarray(use_r), state, ids


def validate_reconstruction(spec, data, steps, seed=173, count=1024):
    """Require exact current pixels before any oracle training is allowed."""
    n, e = data["actions"].shape
    rng = np.random.default_rng(seed)
    t, env = rng.integers(0, n + 1, count), rng.integers(0, e, count)
    state = S.State(
        jnp.asarray(data["pos"][t, env], jnp.int32),
        jnp.asarray(data["wind"][t, env], jnp.int32),
        jnp.zeros((count, 2), jnp.uint32),
        jnp.asarray(steps[t, env], jnp.int32),
        jnp.asarray(data["episode"][t, env], jnp.int32),
        jnp.asarray(data["wall_x"][t, env], jnp.int32),
        jnp.asarray(data["gap"][t, env], jnp.int32),
    )
    rendered = np.asarray(jax.jit(jax.vmap(lambda s, k: S.render(spec, s, k)))(
        state, jax.random.split(jax.random.PRNGKey(seed), count)))
    exact = (rendered == data["frames"][t, env]).all(axis=(1, 2, 3))
    if not exact.all():
        raise ValueError(f"simulator state reconstruction failed: {exact.sum()}/{count} exact")
    return {"frames_checked": count, "exact_frames": int(exact.sum()), "seed": seed}


def make_oracle_collector(bundle, spec, args):
    model, heads = bundle.model, bundle.heads
    B, H = args.batch, args.hor
    simulator_step = jax.vmap(S.make(spec)[1])
    true_signals = args.arm == "oracle_true"

    @nnx.jit
    def collect(ac, ctx_f, ca, use_r, state, key):
        encoded = model.encode(ctx_f)

        def burn(g, inp):
            return ac.step(inp[0], inp[1], g)[2], None

        g, _ = jax.lax.scan(burn, jnp.zeros((B, P.PHID)),
                           (encoded[:, :-1].swapaxes(0, 1), ca[:, :-1].swapaxes(0, 1)))
        g = jnp.where(use_r[:, None], 0.0, g)

        def step(carry, k):
            state, current_frame, stopped, g, prev_a, obs = carry
            # Match the action key used by make_jepa_dream; transition noise has
            # its own key and never reaches the policy.
            ka, ks = jax.random.split(k)
            logits, vlog, g2 = ac.step(obs, prev_a, g)
            act = jax.random.categorical(ka, logits).astype(jnp.int32)
            frame, nxt, true_r, true_d = simulator_step(state, act, jax.random.split(ks, B))
            select = lambda old, new: jnp.where(stopped.reshape((B,) + (1,) * (new.ndim - 1)), old, new)
            nxt = jax.tree.map(select, state, nxt)
            frame = select(current_frame, frame)
            true_r, true_d = jnp.where(stopped, 0.0, true_r), stopped | true_d
            nxt_z = model.encode(frame[:, None])[:, 0]
            if true_signals:
                reward, done = true_r, true_d.astype(jnp.float32)
            else:
                predicted_r, predicted_d, _ = heads(obs, act)
                reward, done = predicted_r.mean(-1), jax.nn.sigmoid(predicted_d)
            output = (obs, prev_a, g, act, P.logprob(logits, act), P.hl_expect(vlog), reward, done, stopped)
            return (nxt, frame, true_d, g2, act, nxt_z), output

        init = (state, ctx_f[:, -1], jnp.zeros((B,), bool), g, ca[:, -1], encoded[:, -1])
        (_, _, finally_stopped, g, prev_a, obs), outputs = jax.lax.scan(step, init, jax.random.split(key, H))
        obs_seq, pas, gs, acts, lps, values, rewards, dones, stopped_before = outputs
        _, last_logits, _ = ac.step(obs, prev_a, g)
        adv, ret = P.gae(values, rewards, dones, P.hl_expect(last_logits), args.gamma, args.lam)
        alive = jnp.concatenate([jnp.ones((1, B)), jnp.cumprod(1.0 - dones[:-1], axis=0)], axis=0)
        diag = jnp.stack([jnp.sum(rewards * alive) / (alive.sum() + 1e-6), dones.mean(),
                          jnp.sum(rewards * alive) / B, alive.mean(), finally_stopped.mean(),
                          jnp.sum(alive * stopped_before) / (alive.sum() + 1e-6),
                          jnp.sum(rewards * alive * stopped_before) / B])
        batch = dict(obs=obs_seq, prev_a=pas, g0=gs, act=acts, logp=lps, adv=adv, ret=ret, alive=alive)
        return jax.tree.map(jax.lax.stop_gradient, batch), diag

    return collect


def make_update(args):
    """Original dream PPO updater, shared unchanged by every diagnostic arm."""
    nch, nb = args.hor // args.chunk, args.batch // args.mb

    @nnx.jit
    def update(ac, opt, batch, ci, bi):
        m = batch["alive"]
        mu = jnp.sum(batch["adv"] * m) / (jnp.sum(m) + 1e-6)
        var = jnp.sum((batch["adv"] - mu) ** 2 * m) / (jnp.sum(m) + 1e-6)
        normalized_adv = (batch["adv"] - mu) * jax.lax.rsqrt(var + 1e-5)

        def sl(x):
            x = x.reshape((nch, args.chunk) + x.shape[1:])[ci]
            return x.reshape((args.chunk, nb, args.mb) + x.shape[2:])[:, bi]

        obs, pa, act = sl(batch["obs"]), sl(batch["prev_a"]), sl(batch["act"])
        oldlp, adv, ret, weight = sl(batch["logp"]), sl(normalized_adv), sl(batch["ret"]), sl(m)
        initial_g = jax.lax.stop_gradient(batch["g0"].reshape((nch, args.chunk) + batch["g0"].shape[1:])[ci, 0]
                                          .reshape((nb, args.mb, P.PHID))[bi])

        def loss(ac):
            def step(g, inp):
                logits, value_logits, g2 = ac.step(inp[0], inp[1], g)
                return g2, (logits, value_logits)

            _, (logits, value_logits) = jax.lax.scan(step, initial_g, (obs, pa))
            denominator = weight.sum() + 1e-6
            ratio = jnp.exp(P.logprob(logits, act) - oldlp)
            pg = -jnp.sum(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - args.clip, 1 + args.clip) * adv)
                          * weight) / denominator
            vf = jnp.sum(P.hl_loss(value_logits, ret) * weight) / denominator
            entropy = jnp.sum(P.entropy(logits) * weight) / denominator
            return pg + args.vf * vf - args.ent * entropy, (pg, vf, entropy)

        (total, aux), grads = nnx.value_and_grad(loss, has_aux=True)(ac)
        opt.update(ac, grads)
        return jnp.stack([total, *aux])

    return update


def run(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output: {args.out}")
    if args.hor % args.chunk or args.batch % args.mb:
        raise ValueError("horizon and batch must split exactly into chunks and minibatches")
    args.out.mkdir(parents=True, exist_ok=True)
    spec = S.CONFIGS[args.config]
    if spec.noise != "off":
        raise ValueError("exact reconstruction is currently restricted to texture-free worlds")
    with np.load(args.data) as archive:
        data = {name: archive[name] for name in ("frames", "actions", "dones", "pos", "wind", "episode", "wall_x", "gap")}
        data_meta = json.loads(str(archive["meta"]))
    if data_meta.get("spec") != asdict(spec):
        raise ValueError("requested environment differs from the dataset specification")
    data["actions"] = data["actions"].astype(np.int32)
    steps = episode_steps(data["episode"], data["dones"])
    reconstruction = validate_reconstruction(spec, data, steps)
    bundle, model_meta = T.load_model(args.model)
    if bundle.heads is None:
        raise ValueError("the diagnostic requires the original fitted reward/done heads")
    if model_meta.get("cfg", {}).get("seed") != args.model_seed:
        raise ValueError("--model-seed disagrees with checkpoint metadata")
    source_files = [Path(__file__), Path(P.__file__), Path(S.__file__), Path(T.__file__),
                    Path(probe.__file__), Path(P.R.__file__)]
    save_policy = P.save_ac
    if args.torso == "native_grid":
        from studies.revision import policy as grid_policy
        cfg = model_meta["cfg"]
        if cfg.get("encoder") != "tile" or cfg.get("tokens") != 16 or cfg.get("token_dim") != 32:
            raise ValueError("native_grid requires the actual 4x4x32 tile encoder topology")
        ac = grid_policy.make_ac(nnx.Rngs(args.policy_seed + 2), int(bundle.model.latent_dim), torso="native_grid")
        save_policy = grid_policy.save_ac
        source_files.append(Path(grid_policy.__file__))
    else:
        ac = P.AC(nnx.Rngs(args.policy_seed + 2), "latent", int(bundle.model.latent_dim))
    opt = nnx.Optimizer(ac, optax.chain(optax.clip_by_global_norm(1.0),
                                      optax.adamw(args.lr, weight_decay=0.01)), wrt=nnx.Param)
    update = make_update(args)
    if args.arm == "learned":
        original = P.make_jepa_dream(bundle, args.batch, args.hor, args.gamma, args.lam)
        collect = lambda ac, fr, ca, ur, state, key: original(ac, fr, ca, ur, key)
    else:
        collect = make_oracle_collector(bundle, spec, args)
    observe = lambda frames: bundle.model.encode(frames[:, None])[:, 0]
    evaluate = P.make_eval(spec, args.eval_episodes, observe)[0]
    provenance = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_sha256": sha256(args.model), "data_sha256": sha256(args.data),
        "source_sha256": {str(path): sha256(path) for path in source_files},
        "reconstruction": reconstruction, "latent_dim": int(bundle.model.latent_dim),
        "policy_params": P.n_params(ac), "model_seed": args.model_seed, "policy_seed": args.policy_seed,
        "terminal_semantics": "exact simulator arms freeze the terminal state; learned-head arm retains learned signals",
        "evaluation": "final checkpoint only; independent fixed key; per-episode outputs retained",
        "model_cfg": model_meta.get("cfg"), "heads_cfg": model_meta.get("heads"),
        "jax_version": jax.__version__, "devices": [str(device) for device in jax.devices()],
    }
    (args.out / "config.json").write_text(json.dumps(provenance, indent=2))
    n, e = data["actions"].shape
    resets, uniform = P.start_tables(data["dones"], n, e, P.CTX)
    host = np.random.default_rng(args.policy_seed + 11)
    rng = jax.random.PRNGKey(args.policy_seed + 7)
    starts_log = []
    start = time.time()
    print("[init] " + json.dumps({"arm": args.arm, "policy_params": P.n_params(ac),
                                  "latent_dim": int(bundle.model.latent_dim), "reconstruction": reconstruction}), flush=True)
    for u in range(1, args.updates + 1):
        t0 = time.time()
        rng, rollout_key = jax.random.split(rng)
        frames, incoming, use_reset, state, ids = sample_starts(host, data, resets, uniform, steps, args.batch, args.p_reset)
        starts_log.append(ids)
        batch, diag = collect(ac, frames, incoming, use_reset, state, rollout_key)
        diag = np.asarray(diag)
        for _ in range(args.epochs):
            for ci in range(args.hor // args.chunk):
                for bi in range(args.batch // args.mb):
                    loss = np.asarray(update(ac, opt, batch, jnp.asarray(ci), jnp.asarray(bi)))
        row = {"update": u, "transitions": u * args.batch * args.hor, "seconds": round(time.time() - t0, 3),
               "reward": float(diag[0]), "done": float(diag[1]), "rollout_return": float(diag[2]),
               "alive": float(diag[3]), "loss": float(loss[0]), "pg": float(loss[1]),
               "vf": float(loss[2]), "entropy": float(loss[3])}
        if args.arm != "learned":
            row.update(true_terminal_fraction=float(diag[4]), postterminal_alive_mass_fraction=float(diag[5]),
                       postterminal_reward_per_rollout=float(diag[6]))
        with (args.out / "ledger.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print("[update] " + json.dumps(row), flush=True)
        if time.time() - start > args.max_minutes * 60:
            print("[stop] time limit reached; incomplete budget will be explicit", flush=True)
            break
    train_seconds = time.time() - start
    outcomes = jax.device_get(evaluate(ac, jax.random.PRNGKey(args.eval_seed)))
    stats = P.summarize(*outcomes)
    final = {"arm": args.arm, "torso": args.torso, "model_seed": args.model_seed, "policy_seed": args.policy_seed,
             "eval_seed": args.eval_seed, "updates_completed": u, "updates_requested": args.updates,
             "budget_complete": u == args.updates, "transitions": u * args.batch * args.hor,
             "train_seconds": round(train_seconds, 3), "total_seconds": round(time.time() - start, 3), **stats}
    save_policy(ac, args.out / "model.pkl", final)
    final["policy_sha256"] = sha256(args.out / "model.pkl")
    final["model_sha256"] = provenance["model_sha256"]
    np.savez_compressed(args.out / "episodes.npz", returns=outcomes[0], lengths=outcomes[1], walls=outcomes[2],
                        eval_seed=args.eval_seed, model_seed=args.model_seed, policy_seed=args.policy_seed)
    np.savez_compressed(args.out / "starts.npz", indices=np.stack(starts_log),
                        columns=np.array(["context_start_t", "environment", "last_frame_t", "is_reset"]))
    (args.out / "final.json").write_text(json.dumps(final, indent=2))
    print("[final] " + json.dumps(final), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--config", default="gate", choices=sorted(S.CONFIGS))
    p.add_argument("--model-seed", required=True, type=int)
    p.add_argument("--policy-seed", default=0, type=int)
    p.add_argument("--torso", choices=("mlp", "native_grid"), default="mlp")
    p.add_argument("--eval-seed", default=20260919, type=int)
    p.add_argument("--eval-episodes", default=256, type=int)
    p.add_argument("--updates", default=150, type=int)
    p.add_argument("--batch", default=512, type=int)
    p.add_argument("--hor", default=16, type=int)
    p.add_argument("--chunk", default=16, type=int)
    p.add_argument("--mb", default=128, type=int)
    p.add_argument("--epochs", default=4, type=int)
    p.add_argument("--p-reset", default=0.5, type=float)
    p.add_argument("--gamma", default=0.99, type=float)
    p.add_argument("--lam", default=0.9, type=float)
    p.add_argument("--lr", default=2.5e-4, type=float)
    p.add_argument("--clip", default=0.2, type=float)
    p.add_argument("--vf", default=0.5, type=float)
    p.add_argument("--ent", default=0.01, type=float)
    p.add_argument("--max-minutes", default=60.0, type=float)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
