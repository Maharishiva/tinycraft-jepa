"""LeWorldModel (LeWM) for Craftax-Classic pixels, ported to JAX / Flax NNX.

Reference implementation (PyTorch): https://github.com/lucas-maes/le-wm
    "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"
    Lucas Maes*, Quentin Le Lidec*, Damien Scieur, Yann LeCun, Randall Balestriero,
    arXiv:2603.19312.  Architecture (ViT encoder -> projector -> AdaLN-zero autoregressive
    predictor) and objective (next-embedding MSE + SIGReg, no stop-gradient, no EMA teacher,
    no pixel decoder) follow `jepa.py` / `module.py` / `train.py` of that repository.

The objective is unchanged.  Adaptations for this task are structural only:
  * 63x63 Craftax frames with patch size 7 (9x9 = 81 patches) instead of 224/14 (16x16).
  * discrete Craftax actions are one-hot encoded before the reference `Embedder`.
  * budgeted schedule: one cosine cycle over the step count that fits 10 minutes on one A100.
"""
import argparse
import functools
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

H = W = 63
PATCH, GRID = 7, 9
N_ACTIONS = 17
CTX, HOR = 8, 16
CDT, PDT = jnp.bfloat16, jnp.float32
MP = dict(dtype=CDT, param_dtype=PDT)
NRM = dict(dtype=PDT, param_dtype=PDT)
IMAGENET_MEAN = jnp.asarray([0.485, 0.456, 0.406])
IMAGENET_STD = jnp.asarray([0.229, 0.224, 0.225])


@dataclass(frozen=True)
class Config:
    seed: int = 0
    steps: int = 11_000
    batch: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-3
    warmup: float = 0.1
    emb_dim: int = 192
    enc_depth: int = 12
    enc_heads: int = 3
    enc_mlp: int = 768
    pred_depth: int = 6
    pred_heads: int = 16
    pred_dim_head: int = 64
    pred_mlp: int = 2048
    mlp_hidden: int = 2048
    history: int = 3
    num_preds: int = 1
    dropout: float = 0.1
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_proj: int = 1024
    action_bottleneck: int = 10
    data_budget: int = 500_000
    chunk: int = 250


def rngs(seed=0):
    return nnx.Rngs(params=seed, dropout=seed + 1)


def normalize(frames_u8):
    return (frames_u8.astype(jnp.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def sigreg(proj, key, knots, num_proj):
    """Sketched isotropic Gaussian regularizer (LeJEPA / LeWM `module.SIGReg`), proj: (T, B, D)."""
    t = jnp.linspace(0.0, 3.0, knots)
    dt = 3.0 / (knots - 1)
    weights = jnp.full((knots,), 2.0 * dt).at[jnp.asarray([0, knots - 1])].set(dt)
    window = jnp.exp(-jnp.square(t) / 2.0)
    directions = jax.random.normal(key, (proj.shape[-1], num_proj))
    directions = directions / jnp.linalg.norm(directions, axis=0, keepdims=True)
    x_t = (proj @ directions)[..., None] * t
    err = jnp.square(jnp.cos(x_t).mean(-3) - window) + jnp.square(jnp.sin(x_t).mean(-3))
    return jnp.mean((err @ (weights * window)) * proj.shape[-2])


class FeedForward(nnx.Module):
    def __init__(self, dim, hidden, rngs, dropout=0.0):
        self.norm = nnx.LayerNorm(dim, rngs=rngs, **NRM)
        self.fc1 = nnx.Linear(dim, hidden, rngs=rngs, **MP)
        self.fc2 = nnx.Linear(hidden, dim, rngs=rngs, **MP)
        self.drop = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, train):
        h = self.drop(nnx.gelu(self.fc1(self.norm(x))), deterministic=not train)
        return self.drop(self.fc2(h), deterministic=not train)


class Attention(nnx.Module):
    def __init__(self, dim, heads, dim_head, rngs, dropout=0.0, causal=False, bias=False):
        self.heads, self.dim_head, self.causal = heads, dim_head, causal
        inner = heads * dim_head
        self.norm = nnx.LayerNorm(dim, rngs=rngs, **NRM)
        self.to_qkv = nnx.Linear(dim, inner * 3, use_bias=bias, rngs=rngs, **MP)
        self.to_out = nnx.Linear(inner, dim, rngs=rngs, **MP)
        self.drop = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, train):
        b, t = x.shape[0], x.shape[1]
        qkv = self.to_qkv(self.norm(x)).reshape(b, t, 3, self.heads, self.dim_head)
        out = jax.nn.dot_product_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], is_causal=self.causal)
        return self.drop(self.to_out(out.reshape(b, t, -1)), deterministic=not train)


class Block(nnx.Module):
    """Pre-norm transformer block (encoder)."""

    def __init__(self, dim, heads, dim_head, mlp, rngs, dropout=0.0):
        self.attn = Attention(dim, heads, dim_head, rngs, dropout, causal=False, bias=True)
        self.ff = FeedForward(dim, mlp, rngs, dropout)

    def __call__(self, x, train):
        x = x + self.attn(x, train)
        return x + self.ff(x, train)


class ConditionalBlock(nnx.Module):
    """AdaLN-zero block conditioned on the action embedding (LeWM `module.ConditionalBlock`)."""

    def __init__(self, dim, heads, dim_head, mlp, rngs, dropout):
        self.attn = Attention(dim, heads, dim_head, rngs, dropout, causal=True, bias=False)
        self.ff = FeedForward(dim, mlp, rngs, dropout)
        self.norm1 = nnx.LayerNorm(dim, use_scale=False, use_bias=False, epsilon=1e-6, rngs=rngs, **NRM)
        self.norm2 = nnx.LayerNorm(dim, use_scale=False, use_bias=False, epsilon=1e-6, rngs=rngs, **NRM)
        self.ada = nnx.Linear(dim, 6 * dim, kernel_init=nnx.initializers.zeros,
                              bias_init=nnx.initializers.zeros, rngs=rngs, **MP)

    def __call__(self, x, cond, train):
        shift1, scale1, gate1, shift2, scale2, gate2 = jnp.split(self.ada(nnx.silu(cond)), 6, -1)
        x = x + gate1 * self.attn(self.norm1(x) * (1 + scale1) + shift1, train)
        return x + gate2 * self.ff(self.norm2(x) * (1 + scale2) + shift2, train)


class ViT(nnx.Module):
    """ViT-tiny read out through its class token; 7x7 patches over a 63x63 frame."""

    def __init__(self, cfg, rngs):
        dim = cfg.emb_dim
        self.patch = nnx.Conv(3, dim, (PATCH, PATCH), strides=(PATCH, PATCH), padding="VALID", rngs=rngs, **MP)
        self.cls = nnx.Param(0.02 * jax.random.normal(rngs.params(), (1, 1, dim), PDT))
        self.pos = nnx.Param(0.02 * jax.random.normal(rngs.params(), (1, 1 + GRID * GRID, dim), PDT))
        self.blocks = nnx.List([Block(dim, cfg.enc_heads, dim // cfg.enc_heads, cfg.enc_mlp, rngs)
                                for _ in range(cfg.enc_depth)])
        self.norm = nnx.LayerNorm(dim, rngs=rngs, **NRM)

    def __call__(self, x, train=False):
        n = x.shape[0]
        h = self.patch(x).reshape(n, GRID * GRID, -1)
        cls = jnp.broadcast_to(self.cls[...].astype(h.dtype), (n, 1, h.shape[-1]))
        h = jnp.concatenate([cls, h], 1) + self.pos[...].astype(h.dtype)
        for block in self.blocks:
            h = block(h, train)
        return self.norm(h)


class MLP(nnx.Module):
    """Linear - BatchNorm - GELU - Linear (LeWM `module.MLP` with BatchNorm1d)."""

    def __init__(self, dim, hidden, out, rngs):
        self.fc1 = nnx.Linear(dim, hidden, rngs=rngs, **MP)
        self.bn = nnx.BatchNorm(hidden, rngs=rngs, **NRM)
        self.fc2 = nnx.Linear(hidden, out, rngs=rngs, **MP)

    def __call__(self, x, train):
        return self.fc2(nnx.gelu(self.bn(self.fc1(x), use_running_average=not train)))


class Embedder(nnx.Module):
    """Action encoder (LeWM `module.Embedder`; its Conv1d(k=1) is a per-step linear map)."""

    def __init__(self, in_dim, bottleneck, emb_dim, rngs, mlp_scale=4):
        self.patch = nnx.Linear(in_dim, bottleneck, rngs=rngs, **MP)
        self.fc1 = nnx.Linear(bottleneck, mlp_scale * emb_dim, rngs=rngs, **MP)
        self.fc2 = nnx.Linear(mlp_scale * emb_dim, emb_dim, rngs=rngs, **MP)

    def __call__(self, actions):
        return self.fc2(nnx.silu(self.fc1(self.patch(actions))))


class Predictor(nnx.Module):
    """Causal AdaLN-zero transformer over embeddings (LeWM `module.ARPredictor`)."""

    def __init__(self, cfg, rngs):
        dim = cfg.emb_dim
        self.pos = nnx.Param(jax.random.normal(rngs.params(), (1, cfg.history, dim), PDT))
        self.blocks = nnx.List([ConditionalBlock(dim, cfg.pred_heads, cfg.pred_dim_head,
                                                 cfg.pred_mlp, rngs, cfg.dropout)
                                for _ in range(cfg.pred_depth)])
        self.norm = nnx.LayerNorm(dim, rngs=rngs, **NRM)

    def __call__(self, x, cond, train=False):
        h = x + self.pos[...][:, :x.shape[1]].astype(x.dtype)
        for block in self.blocks:
            h = block(h, cond, train)
        return self.norm(h)


class LeWM(nnx.Module):
    def __init__(self, cfg, rngs):
        self.encoder = ViT(cfg, rngs)
        self.projector = MLP(cfg.emb_dim, cfg.mlp_hidden, cfg.emb_dim, rngs)
        self.pred_proj = MLP(cfg.emb_dim, cfg.mlp_hidden, cfg.emb_dim, rngs)
        self.action_encoder = Embedder(N_ACTIONS, cfg.action_bottleneck, cfg.emb_dim, rngs)
        self.predictor = Predictor(cfg, rngs)
        self.history = cfg.history

    def tokens(self, frames_u8, train=False):
        b, t = frames_u8.shape[:2]
        return self.encoder(normalize(frames_u8.reshape(b * t, H, W, 3)), train)

    def encode(self, frames_u8, train=False):
        b, t = frames_u8.shape[:2]
        cls = self.tokens(frames_u8, train)[:, 0]
        return self.projector(cls, train).reshape(b, t, -1).astype(PDT)

    def act_embed(self, actions):
        onehot = jax.nn.one_hot(actions, N_ACTIONS, dtype=CDT)
        return self.action_encoder(onehot).astype(PDT)

    def predict(self, emb, act_emb, train=False):
        h = self.predictor(emb, act_emb, train)
        flat = self.pred_proj(h.reshape(-1, h.shape[-1]), train)
        return flat.reshape(emb.shape).astype(PDT)


def valid_starts(dones, span):
    """Window starts whose `span` frames contain no episode reset (eval.py's rule)."""
    t, e = dones.shape
    cumulative = np.concatenate([np.zeros((1, e), np.int64), np.cumsum(dones, axis=0)])
    starts = np.arange(0, t - span + 2)
    ok = cumulative[starts + span - 1] == cumulative[starts]
    rows, cols = np.nonzero(ok)
    return np.stack([starts[rows], cols], 1).astype(np.int32)


def train(data_npz, out, cfg):
    t0 = time.time()
    with np.load(data_npz) as archive:
        frames_np = archive["frames"]
        actions_np = archive["actions"].astype(np.int32)
        dones_np = archive["dones"].astype(np.int64)
    if actions_np.size > cfg.data_budget:
        raise ValueError(f"the training data budget is {cfg.data_budget} transitions")
    span = cfg.history + cfg.num_preds
    table = jnp.asarray(valid_starts(dones_np, span))
    frames = jnp.asarray(frames_np)
    actions = jnp.asarray(actions_np)
    del frames_np, actions_np
    print(f"[data] frames {frames.shape} windows {table.shape[0]} in {time.time() - t0:.1f}s", flush=True)

    model = LeWM(cfg, rngs(cfg.seed))
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    warm = max(1, int(cfg.warmup * cfg.steps))
    schedule = optax.warmup_cosine_decay_schedule(0.0, cfg.lr, warm, cfg.steps, end_value=0.0)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=cfg.weight_decay))
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
    print(f"[model] {n_params / 1e6:.2f}M parameters", flush=True)

    @nnx.jit
    def run(model, opt, frames, actions, table, keys):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def loop(carry, key):
            model, opt = carry
            k_idx, k_sig = jax.random.split(key)
            ids = jax.random.randint(k_idx, (cfg.batch,), 0, table.shape[0])
            start, env = table[ids, 0], table[ids, 1]
            index = start[:, None] + jnp.arange(span)[None]
            window = frames[index, env[:, None]]
            window_actions = actions[index[:, :-1], env[:, None]]

            def loss_fn(model):
                emb = model.encode(window, True)
                act_emb = model.act_embed(window_actions)
                pred = model.predict(emb[:, :cfg.history], act_emb[:, :cfg.history], True)
                pred_loss = jnp.mean(jnp.square(pred - emb[:, cfg.num_preds:]))
                reg = sigreg(emb.transpose(1, 0, 2), k_sig, cfg.sigreg_knots, cfg.sigreg_proj)
                return pred_loss + cfg.sigreg_weight * reg, (pred_loss, reg)

            (_, (pred_loss, reg)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
            opt.update(model, grads)
            return (model, opt), jnp.stack([pred_loss, reg])

        return loop((model, opt), keys)[1]

    t1 = time.time()
    keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 7), cfg.steps)
    history = []
    for begin in range(0, cfg.steps, cfg.chunk):
        losses = run(model, opt, frames, actions, table, keys[begin:begin + cfg.chunk])
        history.append(np.asarray(jax.device_get(losses)))
        done = begin + history[-1].shape[0]
        print(f"[train] {done}/{cfg.steps} pred {history[-1][-1, 0]:.4f} sigreg {history[-1][-1, 1]:.4f}"
              f"  {time.time() - t1:.1f}s", flush=True)
    history = np.concatenate(history)
    train_s = time.time() - t0

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.parent / "losses.npz", history=history)
    with open(data_npz, "rb") as handle:
        data_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    meta = {"cfg": asdict(cfg), "params": n_params, "train_s": round(train_s, 1),
            "loop_s": round(time.time() - t1, 1), "data_sha256": data_sha,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "final_pred_loss": float(history[-1, 0]), "final_sigreg": float(history[-1, 1])}
    save(out, model, meta)
    print(f"[done] {train_s:.1f}s -> {out}", flush=True)
    return model, meta


def _weights(module):
    """Parameters and batch statistics, leaving the rng state out of the checkpoint."""
    _, params, stats, _ = nnx.split(module, nnx.Param, nnx.BatchStat, ...)
    return params, stats


def save(path, model, meta, decoder=None):
    arrays = {}
    for name, module in (("model", model), ("decoder", decoder)):
        if module is None:
            continue
        for kind, state in zip(("p", "b"), _weights(module)):
            arrays.update({f"{name}.{kind}.{i}": np.asarray(jax.device_get(leaf))
                           for i, leaf in enumerate(jax.tree.leaves(state))})
    np.savez(path, meta=json.dumps(meta), **arrays)


def _restore(module, archive, prefix):
    graph, params, stats, rest = nnx.split(module, nnx.Param, nnx.BatchStat, ...)
    restored = []
    for kind, state in zip(("p", "b"), (params, stats)):
        leaves = [jnp.asarray(archive[f"{prefix}.{kind}.{i}"]) for i in range(len(jax.tree.leaves(state)))]
        restored.append(jax.tree.unflatten(jax.tree.structure(state), leaves))
    return nnx.merge(graph, restored[0], restored[1], rest)


def load_model(path=None):
    """Return (model, checkpoint metadata), with measured training seconds in 'train_s'."""
    if path is None:
        raise ValueError("LeWM has no untrained baseline path; pass --model <checkpoint>")
    archive = np.load(path)
    meta = json.loads(str(archive["meta"]))
    cfg = Config(**{f.name: meta["cfg"][f.name] for f in fields(Config) if f.name in meta["cfg"]})
    model = _restore(LeWM(cfg, rngs()), archive, "model")
    decoder = None
    if "decoder.p.0" in archive.files:
        from probe import Decoder

        spec = meta["decoder"]
        decoder = _restore(Decoder(cfg.emb_dim, spec["hidden"], spec["depth"], rngs()), archive, "decoder")
    return (model, decoder), meta


@functools.partial(nnx.jit, static_argnames="horizon")
def _rollout_latents(model, context, out_actions, horizon):
    """Autoregressive latent rollout (LeWM `JEPA.rollout`): keep the last `history` states."""
    emb = model.encode(context)
    act_emb = model.act_embed(out_actions)
    hist = model.history
    carry = (emb[:, -hist:], act_emb[:, context.shape[1] - hist:context.shape[1]])

    def step(carry, j):
        emb_hist, act_hist = carry
        nxt = model.predict(emb_hist, act_hist)[:, -1:]
        nxt_act = jax.lax.dynamic_slice_in_dim(act_emb, context.shape[1] + j, 1, axis=1)
        carry = (jnp.concatenate([emb_hist[:, 1:], nxt], 1), jnp.concatenate([act_hist[:, 1:], nxt_act], 1))
        return carry, nxt[:, 0]

    _, preds = jax.lax.scan(step, carry, jnp.arange(horizon))
    return emb, preds.transpose(1, 0, 2)


def outgoing_actions(context_actions, actions):
    """Action k drives frame k -> k+1; eval.py hands us the *incoming* action of every frame."""
    tail = jnp.zeros((context_actions.shape[0], 1), jnp.int32)
    return jnp.concatenate([jnp.asarray(context_actions[:, 1:], jnp.int32), jnp.asarray(actions, jnp.int32), tail], 1)


def rollout(model, context_frames, context_actions, actions, seed):
    """Return uint8 frames [B, T, ...], where T = actions.shape[1]."""
    model, decoder = model
    if decoder is None:
        raise RuntimeError("this checkpoint has no pixel decoder; run probe.py --decoder first")
    horizon = actions.shape[1]
    _, preds = _rollout_latents(model, jnp.asarray(context_frames),
                                outgoing_actions(context_actions, actions), horizon=horizon)
    frames = decoder(preds.reshape(-1, preds.shape[-1]))
    frames = frames.reshape(preds.shape[0], horizon, H, W, 3)
    return np.asarray(jnp.clip(frames * 255.0 + 0.5, 0, 255).astype(jnp.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tinywmcraft_500k/train.npz"))
    parser.add_argument("--out", type=Path, default=Path("runs/lewm/model.npz"))
    for field in fields(Config):
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=field.type, default=field.default)
    args = parser.parse_args()
    train(args.data, args.out, Config(**{f.name: getattr(args, f.name) for f in fields(Config)}))


if __name__ == "__main__":
    main()
