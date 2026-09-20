"""The one policy every arm shares: the pop-flow-wm Impala-CNN + GRU actor-critic (train_mbrl.py shapes: channels
(64, 128, 128), embedding 512, GRU 256 over the embedding and the previous action, HL-Gauss critic with 128 bins),
the HL-Gauss helpers, GAE, and the pickle checkpoint format of the earlier PPO runs.

`cond_dim` extra scalars enter the GRU beside the previous action (0 = the original 6,818,193-parameter net);
`q_head` adds a 17-way Q head for offline RL; `recurrent=False` drops the GRU and sends the embedding, the
previous action and cond straight to the head, the memoryless control (the trainers keep passing a state array,
which that variant returns untouched). Every arm calls `step(obs_u8, prev_a, g, cond)` with cond of shape
(B, cond_dim), so a policy without conditioning receives a (B, 0) array."""
from __future__ import annotations

import pickle
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp

A_N = 17
PCH = (64, 128, 128)
PEMB = 512
PHID = 256
VBINS = 128
HLSIG = 0.18
CDT, PDT = jnp.bfloat16, jnp.float32
MP = dict(dtype=CDT, param_dtype=PDT)
FP = dict(dtype=PDT, param_dtype=PDT)


def to01(u8):
    return u8.astype(jnp.float32) / 255.0


def to_u8(obs):
    return jnp.asarray(jnp.clip(jnp.rint(obs * 255.0), 0, 255), jnp.uint8)


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def hl_support(n):
    return symexp(jnp.linspace(-15.0, 15.0, n)) / 100.0


def hl_expect(logits):
    return jnp.sum(jax.nn.softmax(logits) * hl_support(logits.shape[-1]), -1)


def hl_loss(logits, target, sigma=HLSIG):
    n = logits.shape[-1]
    c = jnp.linspace(-15.0, 15.0, n)
    y = symlog(target * 100.0)[..., None]
    w = jnp.exp(-0.5 * ((c - y) / sigma) ** 2)
    w = w / jnp.maximum(w.sum(-1, keepdims=True), 1e-6)
    return -jnp.sum(w * jax.nn.log_softmax(logits), -1)


def logprob(logits, a):
    return jnp.take_along_axis(jax.nn.log_softmax(logits), a[..., None], -1).squeeze(-1)


def entropy(logits):
    return -jnp.sum(jax.nn.softmax(logits) * jax.nn.log_softmax(logits), -1)


def gae(values, rewards, dones, last_v, gamma, lam):
    """Time-major (T, B). dones in [0, 1] act as one minus the continuation probability."""
    def step(carry, x):
        adv, nv = carry
        v, r, d = x
        delta = r + gamma * (1.0 - d) * nv - v
        adv = delta + gamma * lam * (1.0 - d) * adv
        return (adv, v), adv
    _, adv = jax.lax.scan(step, (jnp.zeros_like(last_v), last_v), (values[::-1], rewards[::-1], dones[::-1]))
    adv = adv[::-1]
    return adv, adv + values


class ImpalaBlock(nnx.Module):
    def __init__(self, cin, ch, rngs):
        self.conv = nnx.Conv(cin, ch, (3, 3), padding="SAME", rngs=rngs, **MP)
        self.r1a = nnx.Conv(ch, ch, (3, 3), padding="SAME", rngs=rngs, **MP)
        self.r1b = nnx.Conv(ch, ch, (3, 3), padding="SAME", rngs=rngs, **MP)
        self.r2a = nnx.Conv(ch, ch, (3, 3), padding="SAME", rngs=rngs, **MP)
        self.r2b = nnx.Conv(ch, ch, (3, 3), padding="SAME", rngs=rngs, **MP)

    def __call__(self, x):
        x = nnx.max_pool(self.conv(x), (3, 3), strides=(2, 2), padding="SAME")
        x = x + self.r1b(nnx.relu(self.r1a(nnx.relu(x))))
        x = x + self.r2b(nnx.relu(self.r2a(nnx.relu(x))))
        return x


class GRU(nnx.Module):
    def __init__(self, din, dh, rngs):
        self.wz = nnx.Linear(din + dh, dh, rngs=rngs, **FP)
        self.wr = nnx.Linear(din + dh, dh, rngs=rngs, **FP)
        self.wh = nnx.Linear(din + dh, dh, rngs=rngs, **FP)

    def __call__(self, x, g):
        xg = jnp.concatenate([x, g], -1)
        z = jax.nn.sigmoid(self.wz(xg))
        r = jax.nn.sigmoid(self.wr(xg))
        gh = jnp.tanh(self.wh(jnp.concatenate([x, r * g], -1)))
        return (1.0 - z) * g + z * gh


class AC(nnx.Module):
    def __init__(self, rngs, cond_dim=0, q_head=False, recurrent=True):
        self.cond_dim = cond_dim
        self.recurrent = recurrent
        self.b1 = ImpalaBlock(3, PCH[0], rngs)
        self.b2 = ImpalaBlock(PCH[0], PCH[1], rngs)
        self.b3 = ImpalaBlock(PCH[1], PCH[2], rngs)
        self.fc = nnx.Linear(8 * 8 * PCH[2], PEMB, rngs=rngs, **MP)
        self.gru = GRU(PEMB + A_N + cond_dim, PHID, rngs) if recurrent else None
        self.l1 = nnx.Linear(PEMB + (PHID if recurrent else A_N + cond_dim), 512, rngs=rngs, **FP)
        self.actor = nnx.Linear(512, A_N, kernel_init=nnx.initializers.normal(0.01), rngs=rngs, **FP)
        self.critic = nnx.Linear(512, VBINS, rngs=rngs, **FP)
        self.q = nnx.Linear(512, A_N, rngs=rngs, **FP) if q_head else None

    def emb(self, obs_u8):
        x = self.b3(self.b2(self.b1(to01(obs_u8).astype(CDT))))
        e = self.fc(nnx.relu(x).reshape((x.shape[0], -1)))
        return nnx.relu(e).astype(jnp.float32)

    def hidden(self, obs_u8, prev_a, g, cond):
        e = self.emb(obs_u8)
        u = jnp.concatenate([e, jax.nn.one_hot(prev_a, A_N), cond], -1)
        if self.gru is None:
            return nnx.relu(self.l1(u)), g
        g2 = self.gru(u, g)
        return nnx.relu(self.l1(jnp.concatenate([e, g2], -1))), g2

    def step(self, obs_u8, prev_a, g, cond):
        """(B, 63, 63, 3) u8, (B,) int, (B, PHID), (B, cond_dim) -> logits, value logits, next GRU state."""
        h, g2 = self.hidden(obs_u8, prev_a, g, cond)
        return self.actor(h), self.critic(h), g2

    def step_q(self, obs_u8, prev_a, g, cond):
        h, g2 = self.hidden(obs_u8, prev_a, g, cond)
        return self.actor(h), self.q(h), g2


def n_params(module):
    return int(sum(x.size for x in jax.tree.leaves(nnx.state(module, nnx.Param))))


def save_ac(ac, path, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as f:
        pickle.dump({"state": {"ac": jax.device_get(nnx.split(ac)[1])},
                     "arch": {"cond_dim": ac.cond_dim, "q_head": ac.q is not None, "recurrent": ac.recurrent},
                     "meta": meta}, f)


def load_ac(path, q_head=None):
    """Checkpoints of this package and of the earlier pop-flow-wm runs (which carry no arch record)."""
    with Path(path).open("rb") as f:
        d = pickle.load(f)
    arch = dict(d["arch"]) if "arch" in d else {"cond_dim": 0, "q_head": False}
    if q_head is not None:
        arch["q_head"] = q_head
    ac = AC(nnx.Rngs(0), **arch)
    nnx.update(ac, jax.tree.map(jnp.asarray, d["state"]["ac"]))
    return ac, d["meta"]
