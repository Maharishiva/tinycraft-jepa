"""Off-budget pixel probe: decode LeWM latents back to frames.

The JEPA objective never sees pixels.  This decoder is trained *after* the 10-minute budget on
frozen encoder embeddings (the encoder is not updated and no gradient flows into it), purely so
that latent rollouts can be looked at, and so the frozen benchmark in eval.py can score them.
"""
import argparse
import json
import time
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import train as T
from train import GRID, H, MP, PATCH, W, ResBlock


class Decoder(nnx.Module):
    """latent (D,) -> frame (63, 63, 3) in [0, 1]; a grid latent keeps its spatial layout
    (a 4x4 token grid is first brought to 9x9 by a stride-2 transposed convolution)."""

    def __init__(self, dim, hidden, depth, rngs, tokens=1):
        self.tokens = tokens
        self.side = int(round(tokens ** 0.5))
        width = dim // tokens if tokens > 1 else dim
        self.fc = nnx.Linear(width, hidden if tokens > 1 else GRID * GRID * hidden, rngs=rngs, **MP)
        self.up = (nnx.ConvTranspose(hidden, hidden, (3, 3), strides=(2, 2), padding="VALID", rngs=rngs, **MP)
                   if self.side == 4 else None)
        self.blocks = nnx.List([ResBlock(hidden, rngs) for _ in range(depth)])
        self.out = nnx.Linear(hidden, PATCH * PATCH * 3, rngs=rngs, **MP)

    def __call__(self, z):
        n = z.shape[0]
        if self.tokens > 1:
            h = self.fc(z.reshape(n, self.side, self.side, -1))
            if self.up is not None:
                h = nnx.gelu(self.up(h))
        else:
            h = self.fc(z).reshape(n, GRID, GRID, -1)
        for block in self.blocks:
            h = block(h)
        tiles = self.out(h).reshape(n, GRID, GRID, PATCH, PATCH, 3)
        frame = tiles.transpose(0, 1, 3, 2, 4, 5).reshape(n, H, W, 3)
        return nnx.sigmoid(frame.astype(jnp.float32))


class Heads(nnx.Module):
    """Reward, termination and behaviour-value read-outs on frozen latents (stop-gradient).

    The JEPA objective has no reward anywhere; these are what a planner needs on top of it, and
    they are fitted after the budget on the dataset's own rewards and returns.
    """

    def __init__(self, dim, hidden, rngs, members=1, actions=T.N_ACTIONS):
        self.members = members
        self.actions = actions
        self.reward = nnx.List([nnx.Linear(dim + actions, hidden, rngs=rngs, **MP),
                                nnx.Linear(hidden, hidden, rngs=rngs, **MP),
                                nnx.Linear(hidden, members, rngs=rngs, **MP)])
        self.done = nnx.List([nnx.Linear(dim + actions, hidden, rngs=rngs, **MP),
                              nnx.Linear(hidden, 1, rngs=rngs, **MP)])
        self.value = nnx.List([nnx.Linear(dim, hidden, rngs=rngs, **MP),
                               nnx.Linear(hidden, hidden, rngs=rngs, **MP),
                               nnx.Linear(hidden, members, rngs=rngs, **MP)])

    def _run(self, layers, x):
        for layer in layers[:-1]:
            x = nnx.gelu(layer(x))
        return layers[-1](x).astype(jnp.float32)

    def __call__(self, latent, action):
        """Returns reward (N, members), done logit (N,), value (N, members)."""
        joint = jnp.concatenate([latent, jax.nn.one_hot(action, self.actions, dtype=latent.dtype)], -1)
        return self._run(self.reward, joint), self._run(self.done, joint)[..., 0], self._run(self.value, latent)


def pessimism(x, lam):
    """Ensemble mean minus lam standard deviations; identity for a single member."""
    return x.mean(-1) - lam * x.std(-1)


def returns(rewards, dones, gamma):
    """Discounted Monte-Carlo return of the behaviour policy, cut at episode ends."""
    out = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1], rewards.dtype)
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[t] + gamma * (1.0 - dones[t]) * running
        out[t] = running
    return out


def train_heads(latents, actions, rewards, dones, targets, steps, batch, lr, hidden, seed, tail,
                members=1, n_actions=T.N_ACTIONS):
    heads = Heads(latents.shape[-1], hidden, T.rngs(seed + 11), members, n_actions)
    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, max(1, steps // 20), steps, end_value=0.0)
    opt = nnx.Optimizer(heads, optax.chain(optax.clip_by_global_norm(1.0),
                                           optax.adamw(schedule, weight_decay=1e-4)), wrt=nnx.Param)
    flat = lambda x: x.reshape(-1, *x.shape[2:])
    z, a = flat(latents), flat(actions)
    r, d, g = flat(rewards), flat(dones), flat(targets)
    # A truncated Monte-Carlo return is only unbiased far from the end of the recorded window.
    tail_rows = tail * latents.shape[1]

    @nnx.jit
    def run(heads, opt, keys):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def loop(carry, key):
            heads, opt = carry
            k1, k2, k3 = jax.random.split(key, 3)
            ids = jax.random.randint(k1, (batch,), 0, z.shape[0])
            vids = jax.random.randint(k2, (batch,), 0, tail_rows)
            # Each ensemble member sees its own bootstrap sample, so they disagree off-data.
            mask = jax.random.bernoulli(k3, 0.5, (batch, members)) * 2.0 if members > 1 else 1.0

            def loss_fn(heads):
                pred_r, logit_d, _ = heads(z[ids], a[ids])
                _, _, pred_v = heads(z[vids], a[vids])
                loss_r = jnp.mean(mask * jnp.square(pred_r - r[ids][:, None]))
                loss_d = jnp.mean(optax.sigmoid_binary_cross_entropy(logit_d, d[ids]))
                loss_v = jnp.mean(mask * jnp.square(pred_v - g[vids][:, None]))
                return loss_r + loss_d + 0.1 * loss_v, jnp.stack([loss_r, loss_d, loss_v])

            (_, parts), grads = nnx.value_and_grad(loss_fn, has_aux=True)(heads)
            opt.update(heads, grads)
            return (heads, opt), parts

        return loop((heads, opt), keys)[1]

    keys = jax.random.split(jax.random.PRNGKey(seed + 5), steps)
    history = []
    for begin in range(0, steps, 500):
        history.append(np.asarray(jax.device_get(run(heads, opt, keys[begin:begin + 500]))))
        print(f"[heads] {begin + history[-1].shape[0]}/{steps} reward {history[-1][-1, 0]:.4f} "
              f"done {history[-1][-1, 1]:.4f} value {history[-1][-1, 2]:.4f}", flush=True)
    return heads, np.concatenate(history)


def head_quality(heads, latents, actions, rewards, targets, tail_rows, count=100_000):
    """Correlations of the fitted heads, on the frames they were fitted from."""
    count = min(count, latents.shape[0] * latents.shape[1])
    flat = lambda x: np.asarray(x.reshape(-1, *x.shape[2:]))[:count]
    z, a, r, g = flat(latents), flat(actions), flat(rewards), flat(targets)
    pred_r, pred_v = [], []
    for i in range(0, count, 8192):
        out = heads(jnp.asarray(z[i:i + 8192]), jnp.asarray(a[i:i + 8192]))
        pred_r.append(np.asarray(out[0].mean(-1)))
        pred_v.append(np.asarray(out[2].mean(-1)))
    pred_r, pred_v = np.concatenate(pred_r), np.concatenate(pred_v)
    keep = slice(0, min(count, tail_rows))
    return {"reward_corr": float(np.corrcoef(pred_r, r)[0, 1]),
            "value_corr": float(np.corrcoef(pred_v[keep], g[keep])[0, 1]),
            "reward_rmse": float(np.sqrt(np.mean((pred_r - r) ** 2))),
            "reward_event_recall": float(np.mean(np.abs(pred_r[r != 0]) > 0.05)),
            "value_mean": float(pred_v[keep].mean()), "return_mean": float(g[keep].mean())}


@nnx.jit
def _embed_chunk(model, frames_u8):
    return model.encode(frames_u8[None])[0]


def embed_all(model, frames, chunk=2048):
    """Encoder embeddings for every frame, with the encoder frozen (inference mode)."""
    flat = frames.reshape((-1, H, W, 3))
    out = [np.asarray(jax.device_get(_embed_chunk(model, flat[i:i + chunk])))
           for i in range(0, flat.shape[0], chunk)]
    return jnp.asarray(np.concatenate(out))


def train_decoder(frames, latents, steps, batch, lr, hidden, depth, seed, tokens):
    decoder = Decoder(latents.shape[-1], hidden, depth, T.rngs(seed), tokens)
    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, max(1, steps // 20), steps, end_value=0.0)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
    opt = nnx.Optimizer(decoder, tx, wrt=nnx.Param)
    flat = frames.reshape((-1, H, W, 3))

    @nnx.jit
    def run(decoder, opt, flat, latents, keys):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def loop(carry, key):
            decoder, opt = carry
            ids = jax.random.randint(key, (batch,), 0, latents.shape[0])
            target = flat[ids].astype(jnp.float32) / 255.0
            loss, grads = nnx.value_and_grad(
                lambda d: jnp.mean(jnp.square(d(latents[ids]) - target)))(decoder)
            opt.update(decoder, grads)
            return (decoder, opt), loss

        return loop((decoder, opt), keys)[1]

    keys = jax.random.split(jax.random.PRNGKey(seed + 3), steps)
    losses = []
    for begin in range(0, steps, 500):
        losses.append(np.asarray(jax.device_get(run(decoder, opt, flat, latents, keys[begin:begin + 500]))))
        print(f"[decoder] {begin + losses[-1].shape[0]}/{steps} mse {losses[-1][-1]:.5f}", flush=True)
    return decoder, np.concatenate(losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/tinywmcraft_500k/train.npz"))
    parser.add_argument("--out", type=Path, default=None, help="default: <model>_decoder.npz")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--frames", type=int, default=250_000, help="frames used to fit the decoder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heads", action="store_true", help="also fit reward / done / value heads")
    parser.add_argument("--head-steps", type=int, default=6000)
    parser.add_argument("--head-batch", type=int, default=1024)
    parser.add_argument("--head-hidden", type=int, default=512)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--held-out", type=int, default=200, help="time rows kept out of the fit")
    parser.add_argument("--members", type=int, default=1, help="bootstrapped ensemble size")
    parser.add_argument("--no-decoder", action="store_true")
    args = parser.parse_args()

    bundle, meta = T.load_model(args.model)
    model = bundle.model
    tokens = model.n_tokens
    t0 = time.time()
    with np.load(args.data) as archive:
        steps_per_env, envs = archive["actions"].shape
        frames = jnp.asarray(archive["frames"].reshape((-1, H, W, 3))[:args.frames])
        rows = args.frames // envs
        extras = {k: np.asarray(archive[k][:rows], np.float32) for k in ("rewards", "dones")}
        extras["actions"] = np.asarray(archive["actions"][:rows], np.int32)
    latents = embed_all(model, frames)
    print(f"[probe] latents {latents.shape} in {time.time() - t0:.1f}s "
          f"mean {float(latents.mean()):.3f} std {float(latents.std()):.3f}", flush=True)

    decoder, heads = bundle.decoder, bundle.heads
    if not args.no_decoder:
        decoder, losses = train_decoder(frames, latents, args.steps, args.batch, args.lr,
                                        args.hidden, args.depth, args.seed, tokens)
        meta["decoder"] = {"hidden": args.hidden, "depth": args.depth, "steps": args.steps,
                           "tokens": tokens,
                           "batch": args.batch, "lr": args.lr, "frames": int(frames.shape[0]),
                           "decoder_s": round(time.time() - t0, 1), "final_mse": float(losses[-1])}
        print(f"[probe] decoder mse {losses[-1]:.5f} in {time.time() - t0:.1f}s", flush=True)

    if args.heads:
        t1 = time.time()
        grid_latents = latents.reshape(rows, envs, -1)
        targets = returns(extras["rewards"], extras["dones"], args.gamma)
        tail = max(rows // 4, rows - int(np.ceil(4.0 / (1.0 - args.gamma))))
        heads, history = train_heads(grid_latents, jnp.asarray(extras["actions"]),
                                     jnp.asarray(extras["rewards"]), jnp.asarray(extras["dones"]),
                                     jnp.asarray(targets), args.head_steps, args.head_batch,
                                     args.head_lr, args.head_hidden, args.seed, tail, args.members,
                                     model.n_actions)
        quality = head_quality(heads, grid_latents, jnp.asarray(extras["actions"]),
                               extras["rewards"], targets, tail * envs)
        with np.load(args.data) as archive:
            span = slice(rows, rows + args.held_out)
            held_frames = jnp.asarray(archive["frames"][span].reshape((-1, H, W, 3)))
            held = {k: np.asarray(archive[k][rows:rows + args.held_out + 400], np.float32)
                    for k in ("rewards", "dones")}
            held_actions = jnp.asarray(archive["actions"][span], jnp.int32)
        held_targets = returns(held["rewards"], held["dones"], args.gamma)[:args.held_out]
        held_latents = embed_all(model, held_frames).reshape(args.held_out, envs, -1)
        quality.update({f"heldout_{k}": v for k, v in head_quality(
            heads, held_latents, held_actions, held["rewards"][:args.held_out], held_targets,
            args.held_out * envs).items()})
        meta["heads"] = {"hidden": args.head_hidden, "members": args.members,
                         "steps": args.head_steps,
                         "batch": args.head_batch, "lr": args.head_lr, "gamma": args.gamma,
                         "transitions": int(rows * envs), "value_tail": int(tail),
                         "heads_s": round(time.time() - t1, 1),
                         "final": [float(x) for x in history[-1]], **quality}
        print(f"[probe] heads {json.dumps(quality)}", flush=True)

    out = args.out or args.model.with_name(args.model.stem + "_decoder.npz")
    T.save(out, model, meta, decoder=decoder, heads=heads)
    print(f"[probe] saved {time.time() - t0:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
