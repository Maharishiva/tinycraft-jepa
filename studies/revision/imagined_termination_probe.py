"""What does a frozen model's termination head say along its own imagination?

For B uniformly sampled eight-frame real contexts, each frozen model is rolled
forward for H steps under the REAL recorded actions (the data policy is uniform
random, so this is also what PPO sees at its first update, minus the actor's
initialisation bias). Two curves per model and world, both read from the same
heads the imagination uses:

  rollout    head on the model's own predicted latent at step k
  posterior  head on the encoder latent of the real frame at step k (teacher forced)

together with the recorded termination label at step k and, for the JEPA, the
distance between predicted and posterior latents. Nothing is trained; the
adapters are the frozen ones of the shared-CNN campaign (decoded_models).

  JAX_PLATFORMS=cpu python -m studies.revision.imagined_termination_probe --family jepa \
      --config gray_frame_wind --model /dev/shm/redcube/gray_frame_wind/tile_factored/model_probe.npz \
      --heads runs/revision/shared_decoded/heads/gray_frame_wind/model_probe_fullheads.npz \
      --data /dev/shm/redcube/gray_frame_wind/train.npz --out runs/revision/shared_decoded/termination_probe
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from studies.revision import decoded_models as D
from studies.stoch import env as S
from studies.stoch import rl as P


def sample_block(rng, data, horizon, batch):
    T, E = data["actions"].shape
    F = data["frames"].shape[0]
    _, uniform = P.start_tables(data["dones"], T, E, P.CTX)
    table = np.stack(uniform, axis=1)
    table = table[table[:, 0] <= min(T - 7 - horizon, F - 8 - horizon)]
    ids = rng.integers(0, len(table), batch)
    t, e = table[ids, 0], table[ids, 1]
    ti = t[:, None] + np.arange(P.CTX + horizon)
    ei = np.broadcast_to(e[:, None], ti.shape)
    block = data["frames"][ti, ei]
    incoming = data["actions"][t[:, None] + np.arange(P.CTX) - 1, ei[:, :P.CTX]].astype(np.int32)
    ci = t[:, None] + P.CTX - 1 + np.arange(horizon)
    cont_actions = data["actions"][ci, ei[:, :horizon]].astype(np.int32)
    cont_dones = data["dones"][ci, ei[:, :horizon]].astype(np.float32)
    return block, incoming, cont_actions, cont_dones


def jepa_curves(adapter, block, incoming, cont_actions, batch):
    @nnx.jit
    def run(block, incoming, cont_actions):
        post = adapter.encode(block)
        emb = post[:, :P.CTX]
        act_hist = jnp.concatenate([incoming[:, 1:], jnp.zeros((batch, 1), jnp.int32)], 1)

        def step(carry, act):
            emb_hist, act_hist = carry
            acts = act_hist.at[:, -1].set(act)
            reward, done = adapter.read_current(emb_hist[:, -1], act)
            nxt = adapter.predict(emb_hist, acts)
            carry = (jnp.concatenate([emb_hist[:, 1:], nxt[:, None]], 1),
                     jnp.concatenate([acts[:, 1:], jnp.zeros((batch, 1), jnp.int32)], 1))
            return carry, (reward, done, nxt)

        _, (r_roll, d_roll, z) = jax.lax.scan(step, (emb, act_hist), cont_actions.T)
        r_post, d_post = jax.vmap(adapter.read_current, in_axes=(1, 1))(post[:, P.CTX - 1:-1], cont_actions)
        target = jnp.swapaxes(post[:, P.CTX:], 0, 1)
        err = jnp.sqrt(jnp.mean((z - target) ** 2, -1))
        scale = jnp.sqrt(jnp.mean(jnp.var(target, axis=1), -1))
        return dict(done_rollout=d_roll, done_posterior=d_post, reward_rollout=r_roll, reward_posterior=r_post,
                    latent_rmse=err, latent_scale=scale, latent_norm_rollout=jnp.linalg.norm(z, axis=-1),
                    latent_norm_posterior=jnp.linalg.norm(target, axis=-1))
    return run(jnp.asarray(block), jnp.asarray(incoming), jnp.asarray(cont_actions))


def flow_curves(adapter, block, incoming, cont_actions, batch, key):
    wm, win = adapter.wm, adapter.wm.win

    @nnx.jit
    def run(wstate, block, incoming, cont_actions, key):
        post = adapter.encode(block)
        ctx = post[:, :P.CTX]

        def window_loop(fill):
            seq = jnp.zeros((batch, win) + ctx.shape[-3:]).at[:, :P.CTX].set(ctx)
            aseq = jnp.zeros((batch, win), jnp.int32).at[:, :P.CTX].set(incoming)

            def step(carry, inp):
                seq, aseq, n, r = carry
                k, act, kf, kd = inp
                slide = lambda sa: tuple(jnp.concatenate([v[:, 1:], v[:, -1:]], 1) for v in sa)
                seq, aseq = jax.lax.cond(n == win, slide, lambda sa: sa, (seq, aseq))
                r = jnp.where(n == win, jnp.maximum(r - 1, 0), r)
                n = jnp.minimum(n, win - 1)
                aseq = aseq.at[:, n].set(act)
                z = fill(seq, aseq, n, r, k, kf)
                seq = seq.at[:, n].set(z)
                reward, done = wm.read(wstate, seq, aseq, kd)
                return (seq, aseq, n + 1, r), (jnp.take(reward, n, 1), jnp.take(done, n, 1), z)

            horizon = cont_actions.shape[1]
            keys = jax.random.split(key, 2 * horizon).reshape(horizon, 2, -1)
            init = (seq, aseq, jnp.asarray(P.CTX), jnp.asarray(P.CTX))
            _, out = jax.lax.scan(step, init, (jnp.arange(horizon), cont_actions.T, keys[:, 0], keys[:, 1]))
            return out

        r_roll, d_roll, z = window_loop(lambda seq, aseq, n, r, k, kf: wm.flow(wstate, seq, wm.taus(batch, n, r), aseq, n, kf))
        r_post, d_post, _ = window_loop(lambda seq, aseq, n, r, k, kf: post[:, P.CTX + k])
        target = jnp.swapaxes(post[:, P.CTX:], 0, 1)
        flat = lambda a: a.reshape(a.shape[:2] + (-1,))
        err = jnp.sqrt(jnp.mean((flat(z) - flat(target)) ** 2, -1))
        scale = jnp.sqrt(jnp.mean(jnp.var(flat(target), axis=1), -1))
        return dict(done_rollout=d_roll, done_posterior=d_post, reward_rollout=r_roll, reward_posterior=r_post,
                    latent_rmse=err, latent_scale=scale, latent_norm_rollout=jnp.linalg.norm(flat(z), axis=-1),
                    latent_norm_posterior=jnp.linalg.norm(flat(target), axis=-1))
    return run(wm.state, jnp.asarray(block), jnp.asarray(incoming), jnp.asarray(cont_actions), key)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=["jepa", "flow"], required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--heads", type=Path)
    p.add_argument("--flow-code", default="studies/stoch/flow_recipe")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")
    with np.load(a.data) as archive:
        data = {name: archive[name] for name in ("frames", "actions", "dones")}
        meta = json.loads(str(archive["meta"]))
    recorded_spec = dict(meta["spec"])
    recorded_spec.setdefault("gate", False)
    if recorded_spec != asdict(S.CONFIGS[a.config]) or meta["split"] != "train":
        raise ValueError("wrong world or split")
    block, incoming, cont_actions, cont_dones = sample_block(np.random.default_rng(a.seed), data, a.horizon, a.batch)
    adapter = (D.load_jepa(str(a.model), None if a.heads is None else str(a.heads)) if a.family == "jepa"
               else D.load_flow(str(a.model), a.flow_code, 4, nfe=4))
    curves = (jepa_curves(adapter, block, incoming, cont_actions, a.batch) if a.family == "jepa"
              else flow_curves(adapter, block, incoming, cont_actions, a.batch, jax.random.PRNGKey(a.seed)))
    curves = {k: np.asarray(v, np.float32) for k, v in jax.device_get(curves).items()}
    adapter.assert_frozen()
    labels = cont_dones.T
    same_episode = np.concatenate([np.ones((1, a.batch), np.float32), np.cumprod(1 - labels, 0)[:-1]], 0)
    target_valid = np.cumprod(1 - labels, 0)

    def masked_mean(x, m):
        return (x * m).sum(1) / np.maximum(m.sum(1), 1)

    out = dict(family=a.family, config=a.config, batch=a.batch, horizon=a.horizon, seed=a.seed,
               model_sha256=adapter.checkpoint_sha256, data_sha256=D.sha256(a.data),
               label_rate_per_step=masked_mean(labels, same_episode).tolist(),
               same_episode_fraction=same_episode.mean(1).tolist(),
               done_rollout_all=curves["done_rollout"].mean(1).tolist(),
               done_rollout_in_episode=masked_mean(curves["done_rollout"], same_episode).tolist(),
               done_posterior_in_episode=masked_mean(curves["done_posterior"], same_episode).tolist(),
               reward_rollout_in_episode=masked_mean(curves["reward_rollout"], same_episode).tolist(),
               reward_posterior_in_episode=masked_mean(curves["reward_posterior"], same_episode).tolist(),
               latent_rmse_valid=masked_mean(curves["latent_rmse"], target_valid).tolist(),
               latent_scale=curves["latent_scale"].tolist(),
               latent_norm_rollout=curves["latent_norm_rollout"].mean(1).tolist(),
               latent_norm_posterior=masked_mean(curves["latent_norm_posterior"], target_valid).tolist(),
               survival_of_imagination=np.cumprod(1 - curves["done_rollout"], 0).mean(1).tolist(),
               survival_of_data=target_valid.mean(1).tolist())
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.config}_{a.family}.json").write_text(json.dumps(out, indent=1))
    brief = {k: [round(v, 3) for v in out[k][:4]] + ["..."] + [round(out[k][-1], 3)]
             for k in ("label_rate_per_step", "done_rollout_in_episode", "done_posterior_in_episode",
                       "latent_rmse_valid", "latent_scale", "survival_of_imagination", "survival_of_data")}
    print(json.dumps(dict(config=a.config, family=a.family, **brief)), flush=True)


if __name__ == "__main__":
    main()
