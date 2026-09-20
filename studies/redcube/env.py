"""Red cube: a 63x63 toy environment for the "JEPA and stochastic worlds" argument.

A red 6x6 cube moves 3 pixels per action (left, up, right, down).  A hidden wind, an integer
random walk in [-2, 2] per axis starting from zero, is added to every move.  A red 1-pixel border sits one pixel
inside the frame; touching it is death (reward -1, episode over).  Otherwise reward is 0 and the
episode is capped at 128 steps.  The background is one of: black, gray noise, colour noise; the
noise is either redrawn every frame or drawn once per episode and frozen.

    python -m studies.redcube.env --config color_frame_wind --out /dev/shm/redcube/color_frame_wind
    python -m studies.redcube.env --stats            # episode-length statistics of the random policy
    python -m studies.redcube.env --preview x.png    # one frame per configuration
"""
import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

H = W = 63
CUBE, STEP, BORDER = 6, 3, 1
SAFE_LO, SAFE_HI = BORDER + 1, W - 1 - BORDER - CUBE
START_LO, START_HI = 16, 42
MAX_STEPS = 128
N_ACTIONS = 4
ACTION_NAMES = ("left", "up", "right", "down")
MOVES = jnp.asarray([[-STEP, 0], [0, -STEP], [STEP, 0], [0, STEP]], jnp.int32)
WIND_MAX, WIND_STAY = 2, 0.8
RED = jnp.asarray([255, 0, 0], jnp.uint8)
CENTRE = (W - CUBE) / 2.0


@dataclass(frozen=True)
class Spec:
    noise: str = "color"
    per_episode: bool = False
    wind: bool = True

    @property
    def name(self):
        if self.noise == "off":
            return f"off_{'wind' if self.wind else 'still'}"
        return f"{self.noise}_{'episode' if self.per_episode else 'frame'}_{'wind' if self.wind else 'still'}"


CONFIGS = {s.name: s for s in [Spec(noise, per_episode, wind)
                               for noise in ("color", "gray", "off")
                               for per_episode in ((False, True) if noise != "off" else (False,))
                               for wind in (True, False)]}
ORDER = ["color_frame_wind", "color_episode_wind", "off_wind", "off_still", "color_frame_still",
         "color_episode_still", "gray_frame_wind", "gray_episode_wind", "gray_frame_still", "gray_episode_still"]


class State(NamedTuple):
    pos: jax.Array
    wind: jax.Array
    texture: jax.Array
    step: jax.Array
    episode: jax.Array


def background(spec, key):
    if spec.noise == "off":
        return jnp.zeros((H, W, 3), jnp.uint8)
    if spec.noise == "gray":
        return jnp.broadcast_to(jax.random.randint(key, (H, W, 1), 0, 256).astype(jnp.uint8), (H, W, 3))
    return jax.random.randint(key, (H, W, 3), 0, 256).astype(jnp.uint8)


def render(spec, state, key):
    img = background(spec, state.texture if spec.per_episode else key)
    yy, xx = jnp.mgrid[0:H, 0:W]
    inside = (xx >= BORDER) & (xx <= W - 1 - BORDER) & (yy >= BORDER) & (yy <= H - 1 - BORDER)
    ring = inside & ((xx == BORDER) | (xx == W - 1 - BORDER) | (yy == BORDER) | (yy == H - 1 - BORDER))
    x0, y0 = jnp.clip(state.pos[0], 0, W - CUBE), jnp.clip(state.pos[1], 0, H - CUBE)
    cube = (xx >= x0) & (xx < x0 + CUBE) & (yy >= y0) & (yy < y0 + CUBE)
    return jnp.where((ring | cube)[..., None], RED, img)


def wind_walk(wind, key):
    roll = jax.random.uniform(key, (2,))
    delta = jnp.where(roll < (1 - WIND_STAY) / 2, -1, jnp.where(roll < 1 - WIND_STAY, 1, 0))
    return jnp.clip(wind + delta, -WIND_MAX, WIND_MAX)


def make(spec):
    """reset(key) -> (obs, state); step(state, action, key) -> (obs, state, reward, done)."""

    def reset(key, episode=0):
        k_pos, k_tex, k_frame = jax.random.split(key, 3)
        pos = jax.random.randint(k_pos, (2,), START_LO, START_HI + 1)
        state = State(pos, jnp.zeros(2, jnp.int32), k_tex, jnp.asarray(0, jnp.int32), jnp.asarray(episode, jnp.int32))
        return render(spec, state, k_frame), state

    def step(state, action, key):
        k_wind, k_frame = jax.random.split(key)
        wind = wind_walk(state.wind, k_wind) if spec.wind else state.wind
        pos = state.pos + MOVES[action] + wind
        dead = jnp.any(pos < SAFE_LO) | jnp.any(pos > SAFE_HI)
        t = state.step + 1
        state = State(pos, wind, state.texture, t, state.episode)
        done = dead | (t >= MAX_STEPS)
        return render(spec, state, k_frame), state, jnp.where(dead, -1.0, 0.0), done

    return reset, step


def batched(spec):
    """Vectorised reset/step with auto-reset folded in: after a done the returned obs/state are fresh."""
    reset, step = make(spec)
    vreset = jax.vmap(reset)

    def vstep(state, action, key):
        ks, kr = jax.random.split(key)
        n = action.shape[0]
        obs, nstate, reward, done = jax.vmap(step)(state, action, jax.random.split(ks, n))
        robs, rstate = jax.vmap(reset)(jax.random.split(kr, n), nstate.episode + 1)
        pick = lambda r, x: jnp.where(done.reshape((n,) + (1,) * (x.ndim - 1)), r, x)
        return obs, jax.tree.map(pick, rstate, nstate), reward, done, jnp.where(done[:, None, None, None], robs, obs)

    return jax.jit(vreset), jax.jit(vstep)


def centre_policy(state):
    """Reference controller on the true state: step against the larger centre offset, wind included."""
    drift = state.pos.astype(jnp.float32) + state.wind - CENTRE
    axis = jnp.argmax(jnp.abs(drift), -1)
    sign = jnp.take_along_axis(drift, axis[:, None], 1)[:, 0] > 0
    return jnp.where(axis == 0, jnp.where(sign, 0, 2), jnp.where(sign, 1, 3)).astype(jnp.int32)


def collect(spec, envs, steps, seed, policy=None):
    """Random-policy (or `policy(state, key)`) rollout with resets: frames (T+1, E, H, W, 3), actions (T, E) ..."""
    vreset, vstep = batched(spec)
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    obs, state = vreset(jax.random.split(k0, envs), jnp.zeros(envs, jnp.int32))

    @jax.jit
    def body(carry, k):
        state, obs = carry
        ka, ks = jax.random.split(k)
        action = jax.random.randint(ka, (envs,), 0, N_ACTIONS) if policy is None else policy(state, ka)
        shown, nstate, reward, done, nobs = vstep(state, action, ks)
        return (nstate, nobs), (obs, action, reward, done, state.pos, state.wind, state.episode, shown)

    (state, obs), (frames, actions, rewards, dones, pos, wind, episode, shown) = jax.lax.scan(
        body, (state, obs), jax.random.split(key, steps))
    frames = jnp.concatenate([frames, obs[None]])
    pos = jnp.concatenate([pos, state.pos[None]])
    wind = jnp.concatenate([wind, state.wind[None]])
    episode = jnp.concatenate([episode, state.episode[None]])
    return {"frames": np.asarray(frames), "actions": np.asarray(actions, np.int16),
            "rewards": np.asarray(rewards, np.float32), "dones": np.asarray(dones, bool),
            "pos": np.asarray(pos, np.int16), "wind": np.asarray(wind, np.int8),
            "episode": np.asarray(episode, np.int32), "death_frames": np.asarray(shown)}


def episode_stats(dones, rewards):
    t, e = dones.shape
    lengths, deaths = [], 0
    for env in range(e):
        ends = np.nonzero(dones[:, env])[0]
        start = 0
        for end in ends:
            lengths.append(end - start + 1)
            deaths += rewards[end, env] < 0
            start = end + 1
    lengths = np.asarray(lengths)
    return {"episodes": int(len(lengths)), "mean_length": float(lengths.mean()), "median_length": float(np.median(lengths)),
            "min_length": int(lengths.min()), "p10_length": float(np.percentile(lengths, 10)),
            "deaths": int(deaths), "timeouts": int(len(lengths) - deaths), "frac_len_ge_25": float((lengths >= 25).mean())}


def save(path, data, spec, split, seed, envs, steps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"env": "redcube", "spec": asdict(spec), "config": spec.name, "num_actions": N_ACTIONS, "envs": envs,
            "steps": steps, "split": split, "seed": seed, "policy": "uniform random", "max_steps": MAX_STEPS,
            "reward": "-1 on touching the border, else 0"}
    death_frames = data.pop("death_frames")
    np.savez(path, meta=json.dumps(meta), **data)
    return death_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="color_frame_wind", choices=sorted(CONFIGS))
    parser.add_argument("--out", type=Path, default=None, help="directory for train.npz / eval.npz")
    parser.add_argument("--envs", type=int, default=100)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()

    if args.stats:
        for name in ORDER:
            spec = CONFIGS[name]
            data = collect(spec, 64, 512, args.seed)
            print(name, json.dumps(episode_stats(data["dones"], data["rewards"])), flush=True)
        return
    if args.preview is not None:
        from PIL import Image

        tiles = []
        for name in ORDER:
            spec = CONFIGS[name]
            obs, state = make(spec)[0](jax.random.PRNGKey(args.seed))
            tiles.append(np.asarray(obs))
        row = np.concatenate([np.repeat(np.repeat(t, 3, 0), 3, 1) for t in tiles], 1)
        Image.fromarray(row).save(args.preview)
        print(f"preview -> {args.preview}", flush=True)
        return

    spec = CONFIGS[args.config]
    out = args.out or Path("data/redcube") / spec.name
    for split, steps, seed in (("train", args.train_steps, args.seed), ("eval", args.eval_steps, args.seed + 1000)):
        t0 = time.time()
        data = collect(spec, args.envs, steps, seed)
        stats = episode_stats(data["dones"], data["rewards"])
        save(out / f"{split}.npz", data, spec, split, seed, args.envs, steps)
        print(f"[{spec.name}] {split}: {steps * args.envs} transitions in {time.time() - t0:.1f}s "
              f"{json.dumps(stats)} -> {out / f'{split}.npz'}", flush=True)


if __name__ == "__main__":
    main()
