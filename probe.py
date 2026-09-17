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
from train import GRID, H, MP, NRM, PATCH, W


class ResBlock(nnx.Module):
    def __init__(self, ch, rngs):
        self.n1 = nnx.RMSNorm(ch, rngs=rngs, **NRM)
        self.c1 = nnx.Conv(ch, ch, (3, 3), padding="SAME", rngs=rngs, **MP)
        self.n2 = nnx.RMSNorm(ch, rngs=rngs, **NRM)
        self.c2 = nnx.Conv(ch, ch, (3, 3), padding="SAME", rngs=rngs, **MP)

    def __call__(self, h):
        return h + self.c2(nnx.gelu(self.n2(self.c1(nnx.gelu(self.n1(h))))))


class Decoder(nnx.Module):
    """latent (D,) -> frame (63, 63, 3) in [0, 1]; a grid latent keeps its spatial layout."""

    def __init__(self, dim, hidden, depth, rngs, tokens=1):
        self.tokens = tokens
        width = dim // tokens if tokens > 1 else dim
        self.fc = nnx.Linear(width, hidden if tokens > 1 else GRID * GRID * hidden, rngs=rngs, **MP)
        self.blocks = nnx.List([ResBlock(hidden, rngs) for _ in range(depth)])
        self.out = nnx.Linear(hidden, PATCH * PATCH * 3, rngs=rngs, **MP)

    def __call__(self, z):
        n = z.shape[0]
        shaped = z.reshape(n, GRID, GRID, -1) if self.tokens > 1 else z
        h = self.fc(shaped).reshape(n, GRID, GRID, -1)
        for block in self.blocks:
            h = block(h)
        tiles = self.out(h).reshape(n, GRID, GRID, PATCH, PATCH, 3)
        frame = tiles.transpose(0, 1, 3, 2, 4, 5).reshape(n, H, W, 3)
        return nnx.sigmoid(frame.astype(jnp.float32))


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
    args = parser.parse_args()

    (model, _), meta = T.load_model(args.model)
    tokens = GRID * GRID if model.grid else 1
    t0 = time.time()
    with np.load(args.data) as archive:
        frames = jnp.asarray(archive["frames"].reshape((-1, H, W, 3))[:args.frames])
    latents = embed_all(model, frames)
    print(f"[probe] latents {latents.shape} in {time.time() - t0:.1f}s "
          f"mean {float(latents.mean()):.3f} std {float(latents.std()):.3f}", flush=True)
    decoder, losses = train_decoder(frames, latents, args.steps, args.batch, args.lr,
                                    args.hidden, args.depth, args.seed, tokens)
    meta["decoder"] = {"hidden": args.hidden, "depth": args.depth, "steps": args.steps,
                       "tokens": tokens,
                       "batch": args.batch, "lr": args.lr, "frames": int(frames.shape[0]),
                       "decoder_s": round(time.time() - t0, 1), "final_mse": float(losses[-1])}
    out = args.out or args.model.with_name(args.model.stem + "_decoder.npz")
    T.save(out, model, meta, decoder=decoder)
    print(f"[probe] decoder mse {losses[-1]:.5f} in {time.time() - t0:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
