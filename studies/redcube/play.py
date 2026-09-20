"""Play the red cube side by side: the real environment, the flow-matching world model, and the LeWM arms.

Every model is primed on the same real frames, then free-runs on your keystrokes and drifts from the real
environment on its own.  LeWM frames are its predicted latents through the off-budget pixel decoder.

    python -m studies.redcube.play --config color_frame_wind --flow-code <recipe dir> \
        --flow /dev/shm/redcube/color_frame_wind/flow/model.npz \
        --jepa /dev/shm/redcube/color_frame_wind/jepa2/model_probe.npz /dev/shm/redcube/color_frame_wind/grid/model_probe.npz
    python -m studies.redcube.play ... --selftest      # headless, writes play_selftest.png

keys: arrows or w a s d = move, backspace = new episode, esc = quit
"""
import argparse
import os
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import probe
import train as T
from studies.redcube import env as R
from studies.redcube import flow as F

CTX = 8


class FlowWM:
    """Sliding latent window; real frames sit in the context at signal 1.0, imagined ones at CTX_SIGNAL."""

    def __init__(self, train_mod, model, steps):
        self.m, self.model, self.win = train_mod, model, model.win
        self.gdef, self.state = nnx.split(model.df)
        self.sample = jax.jit(lambda state, seq, taus, aseq, slot, key: train_mod._flow_one(
            nnx.merge(self.gdef, state), seq, taus, aseq, slot, key, steps))
        self.key = jax.random.PRNGKey(0)

    def reset(self, obs_u8):
        z0 = self.m.encode(self.model, jnp.asarray(obs_u8)[None])[0]
        self.seq = jnp.zeros((1, self.win) + z0.shape).at[0, 0].set(z0)
        self.aseq = jnp.zeros((1, self.win), jnp.int32)
        self.taus = jnp.zeros((1, self.win)).at[0, 0].set(1.0)
        self.n = 1
        return np.asarray(self.m.decode(self.model, z0[None]))[0]

    def step(self, action, prime_obs=None):
        if self.n >= self.win:
            self.seq = self.seq.at[0, :-1].set(self.seq[0, 1:])
            self.aseq = self.aseq.at[0, :-1].set(self.aseq[0, 1:])
            self.taus = self.taus.at[0, :-1].set(self.taus[0, 1:]).at[0, -1].set(0.0)
            self.n = self.win - 1
        s = self.n
        self.aseq = self.aseq.at[0, s].set(action)
        if prime_obs is not None:
            z, level = self.m.encode(self.model, jnp.asarray(prime_obs)[None])[0], 1.0
        else:
            self.key, kr = jax.random.split(self.key)
            z, level = self.sample(self.state, self.seq, self.taus, self.aseq, jnp.asarray(s, jnp.int32), kr)[0], self.m.CTX_SIGNAL
        self.seq, self.taus, self.n = self.seq.at[0, s].set(z), self.taus.at[0, s].set(level), self.n + 1
        return np.asarray(self.m.decode(self.model, z[None]))[0]


class JepaWM:
    """LeWM history of `history` latents and their outgoing actions; the next latent is the predictor's output."""

    def __init__(self, bundle):
        self.model, self.decoder = bundle.model, bundle.decoder
        self.hist = self.model.history
        self.encode = nnx.jit(lambda m, x: m.encode(x))
        self.predict = nnx.jit(lambda m, e, a: m.predict(e, m.act_embed(a))[:, -1])
        self.decode = nnx.jit(lambda d, z: jnp.clip(d(z) * 255.0 + 0.5, 0, 255).astype(jnp.uint8))

    def reset(self, obs_u8):
        z0 = self.encode(self.model, jnp.asarray(obs_u8)[None, None])[:, 0]
        self.emb = jnp.repeat(z0[:, None], self.hist, 1)
        self.act = jnp.zeros((1, self.hist), jnp.int32)
        return np.asarray(self.decode(self.decoder, z0))[0]

    def step(self, action, prime_obs=None):
        self.act = self.act.at[0, -1].set(action)
        if prime_obs is not None:
            z = self.encode(self.model, jnp.asarray(prime_obs)[None, None])[:, 0]
        else:
            z = self.predict(self.model, self.emb, self.act)
        self.emb = jnp.concatenate([self.emb[:, 1:], z[:, None]], 1)
        self.act = jnp.concatenate([self.act[:, 1:], jnp.zeros((1, 1), jnp.int32)], 1)
        return np.asarray(self.decode(self.decoder, z))[0]


def load_all(args):
    for mod in (T, probe):
        mod.CDT = jnp.float32
        mod.MP = dict(dtype=jnp.float32, param_dtype=jnp.float32)
    wms, labels = [], []
    if args.flow is not None:
        train_mod, _, model = F.load(args.flow_code, args.flow, R.N_ACTIONS)
        train_mod.CDT = jnp.float32
        train_mod.MP = dict(dtype=jnp.float32, param_dtype=jnp.float32)
        wms.append(FlowWM(train_mod, model, args.steps))
        labels.append("flow-matching WM")
    for path in args.jepa:
        bundle, _ = T.load_model(path)
        wms.append(JepaWM(bundle))
        labels.append(f"LeWM {bundle.model.latent_dim}-d")
    return wms, labels


def episode_start(spec, wms, key, prime):
    reset, step = R.make(spec)
    reset, step = jax.jit(reset), jax.jit(step)
    key, k0 = jax.random.split(key)
    obs, state = reset(k0)
    real = np.asarray(obs)
    imgs = [wm.reset(real) for wm in wms]
    for _ in range(prime):
        key, ka, ks = jax.random.split(key, 3)
        a = int(jax.random.randint(ka, (), 0, R.N_ACTIONS))
        obs, state, _, done = step(state, a, ks)
        real = np.asarray(obs)
        imgs = [wm.step(a, real) for wm in wms]
    return step, key, state, real, imgs, bool(done)


def run(args):
    import pygame
    spec = R.CONFIGS[args.config]
    wms, labels = load_all(args)
    key = jax.random.PRNGKey(args.seed)
    step, key, state, real, imgs, dead = episode_start(spec, wms, key, args.prime)
    t = 0
    pygame.init()
    font = pygame.font.SysFont("menlo,consolas,monospace", 15)
    big = pygame.font.SysFont("menlo,consolas,monospace", 17, bold=True)
    n_panes = 1 + len(wms)
    scale = args.scale or {2: 7, 3: 6, 4: 5}.get(n_panes, 4)
    pane, sidebar, gap, head = R.H * scale, 230, 16, 28
    screen = pygame.display.set_mode((gap + n_panes * (pane + gap) + sidebar, head + pane + gap))
    pygame.display.set_caption(f"red cube {spec.name}: REAL vs world models")
    KEY = {pygame.K_LEFT: 0, pygame.K_UP: 1, pygame.K_RIGHT: 2, pygame.K_DOWN: 3,
           pygame.K_a: 0, pygame.K_w: 1, pygame.K_d: 2, pygame.K_s: 3}
    hints = [("move", "arrows / wasd"), ("reset", "backspace"), ("quit", "esc"), ("", ""),
             ("red box", "= imagined")]

    def draw():
        screen.fill((18, 18, 20))
        for i, (img, label) in enumerate([(real, "REAL")] + list(zip(imgs, labels))):
            x = gap + i * (pane + gap)
            surf = pygame.surfarray.make_surface(np.asarray(img).transpose(1, 0, 2))
            screen.blit(pygame.transform.scale(surf, (pane, pane)), (x, head))
            screen.blit(big.render(label, True, (235, 235, 235)), (x, 6))
            if i > 0:
                pygame.draw.rect(screen, (196, 40, 40), (x - 3, head - 3, pane + 6, pane + 6), 3)
        sx, y = gap + n_panes * (pane + gap), 10
        for k, v in hints:
            screen.blit(font.render(f"{k:<8}{v}", True, (160, 170, 180)), (sx, y))
            y += 21
        screen.blit(font.render(f"step {t}  wind {np.asarray(state.wind).tolist()}", True, (120, 200, 130)), (sx, y + 10))
        if dead:
            screen.blit(big.render("DEAD -- backspace", True, (240, 90, 90)), (gap, head + pane - 24))
        pygame.display.flip()

    draw()
    running = True
    while running:
        ev = pygame.event.wait()
        if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
            running = False
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_BACKSPACE:
            step, key, state, real, imgs, dead = episode_start(spec, wms, key, args.prime)
            t = 0
            draw()
        elif ev.type == pygame.KEYDOWN and ev.key in KEY and not dead:
            a = KEY[ev.key]
            key, ks = jax.random.split(key)
            obs, state, _, done = step(state, a, ks)
            real, dead = np.asarray(obs), bool(done)
            imgs = [wm.step(a) for wm in wms]
            t += 1
            draw()
    pygame.quit()


def selftest(args):
    import imageio.v2 as imageio
    spec = R.CONFIGS[args.config]
    wms, labels = load_all(args)
    step, key, state, real, imgs, dead = episode_start(spec, wms, jax.random.PRNGKey(args.seed), min(args.prime, CTX - 1))
    up = lambda im: np.repeat(np.repeat(np.asarray(im), 3, 0), 3, 1)
    rows = []
    for a in [2] * 4 + [1] * 3 + [0] * 3:
        key, ks = jax.random.split(key)
        obs, state, _, _ = step(state, a, ks)
        real = np.asarray(obs)
        sep = np.full((up(real).shape[0], 3, 3), 255, np.uint8)
        cols = [up(real), sep]
        for wm in wms:
            cols += [up(wm.step(a)), sep]
        rows.append(np.concatenate(cols[:-1], 1))
    gap = np.full((4, rows[0].shape[1], 3), 30, np.uint8)
    out = rows[0]
    for r in rows[1:]:
        out = np.concatenate([out, gap, r], 0)
    imageio.imwrite(args.out, out)
    print(f"selftest ok -> {args.out} (REAL | {' | '.join(labels)}), one row per step", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, choices=sorted(R.CONFIGS))
    p.add_argument("--flow", type=Path, default=None)
    p.add_argument("--flow-code", type=Path, default=None)
    p.add_argument("--jepa", type=Path, nargs="*", default=[])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prime", type=int, default=CTX - 1, help="real-context steps before you take over")
    p.add_argument("--steps", type=int, default=8, help="Euler steps per frame of the flow model")
    p.add_argument("--scale", type=int, default=0)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--out", type=Path, default=Path("play_selftest.png"))
    a = p.parse_args()
    if a.selftest:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        selftest(a)
    else:
        run(a)
