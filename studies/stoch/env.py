"""Stochastic-control worlds for the JEPA vs flow-matching study (superset of studies.redcube.env).

The red-cube world (63x63, red 6x6 cube, four 3-px moves, hidden wind, death on the border, 128-step cap,
background off / gray noise per frame or per episode) plus the **gate** world:

  A white wall, 3 px wide and full height, enters at the right edge every 22 steps and scrolls left 3 px per
  step.  It has a 16-px gap at the top or at the bottom (probability 1/2 each), but the gap is drawn only once
  the wall has reached x <= 40: before that the wall is solid.  Touching the wall is death (reward -1), like
  the border.  A cube at the centre sees the gap 3 moves before contact but needs 5 moves to reach either gap,
  so it can only pre-commit to one side; a cube that waits on the left third sees the gap 6+ moves early and
  always makes it.  The mean of the two futures is a half-open wall on both sides, which is not a state of
  the world.

    python -m studies.stoch.env --config gate --out /dev/shm/stoch/gate --train-steps 5000
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

from studies.redcube import env as R

H, W, CUBE, STEP, BORDER = R.H, R.W, R.CUBE, R.STEP, R.BORDER
SAFE_LO, SAFE_HI, START_LO, START_HI, MAX_STEPS = R.SAFE_LO, R.SAFE_HI, R.START_LO, R.START_HI, R.MAX_STEPS
N_ACTIONS, ACTION_NAMES, MOVES, CENTRE, RED = R.N_ACTIONS, R.ACTION_NAMES, R.MOVES, R.CENTRE, R.RED
WALL_W, WALL_SPEED, WALL_PERIOD, WALL_ENTER, REVEAL_X = 3, 3, 22, 60, 40
GAP = 16
GAP_ROWS = ((BORDER + 3, BORDER + 3 + GAP), (H - 1 - BORDER - 3 - GAP, H - 1 - BORDER - 3))
WALL_RGB = jnp.asarray([255, 255, 255], jnp.uint8)
NO_WALL = -99


@dataclass(frozen=True)
class Spec:
    noise: str = "off"
    per_episode: bool = False
    wind: bool = False
    gate: bool = False

    @property
    def name(self):
        base = R.Spec(self.noise, self.per_episode, self.wind).name
        return f"gate_{base}" if self.gate else base


CONFIGS = {s.name: s for s in [Spec(**asdict(spec)) for spec in R.CONFIGS.values()]}
CONFIGS.update({s.name: s for s in [Spec("off", False, False, True), Spec("off", False, True, True),
                                    Spec("gray", False, False, True), Spec("gray", True, False, True)]})
CONFIGS["gate"] = CONFIGS["gate_off_still"]


class State(NamedTuple):
    pos: jax.Array
    wind: jax.Array
    texture: jax.Array
    step: jax.Array
    episode: jax.Array
    wall_x: jax.Array
    gap: jax.Array


def wall_mask(wall_x, gap):
    """(H, W) bool: wall pixels for a wall at `wall_x` with gap `gap` (0 top, 1 bottom), solid before REVEAL_X."""
    yy, xx = jnp.mgrid[0:H, 0:W]
    inside = (xx > BORDER) & (xx < W - 1 - BORDER) & (yy > BORDER) & (yy < H - 1 - BORDER)
    column = inside & (xx >= wall_x) & (xx < wall_x + WALL_W)
    lo = jnp.where(gap == 0, GAP_ROWS[0][0], GAP_ROWS[1][0])
    hi = jnp.where(gap == 0, GAP_ROWS[0][1], GAP_ROWS[1][1])
    opening = (yy >= lo) & (yy < hi) & (wall_x <= REVEAL_X)
    return column & ~opening & (wall_x != NO_WALL)


def render(spec, state, key):
    rspec = R.Spec(spec.noise, spec.per_episode, spec.wind)
    img = R.render(rspec, R.State(state.pos, state.wind, state.texture, state.step, state.episode), key)
    if not spec.gate:
        return img
    return jnp.where(wall_mask(state.wall_x, state.gap)[..., None], WALL_RGB, img)


def cube_hits_wall(pos, wall_x, gap):
    x0, y0 = pos[0], pos[1]
    lo = jnp.where(gap == 0, GAP_ROWS[0][0], GAP_ROWS[1][0])
    hi = jnp.where(gap == 0, GAP_ROWS[0][1], GAP_ROWS[1][1])
    overlap_x = (wall_x < x0 + CUBE) & (wall_x + WALL_W > x0) & (wall_x != NO_WALL)
    in_gap = (y0 >= lo) & (y0 + CUBE <= hi) & (wall_x <= REVEAL_X)
    return overlap_x & ~in_gap


EDGE_LO, EDGE_HI = 3, 8


def edge_pos(key):
    """A start 3 to 8 px from one of the four borders (the other coordinate uniform as usual): survival then
    needs the position, a state-blind policy dies on the wrong side half the time."""
    k_side, k_dist, k_other = jax.random.split(key, 3)
    side = jax.random.randint(k_side, (), 0, 4)
    dist = jax.random.randint(k_dist, (), EDGE_LO, EDGE_HI + 1)
    other = jax.random.randint(k_other, (), START_LO, START_HI + 1)
    near = jnp.where(side % 2 == 0, SAFE_LO + dist, SAFE_HI - dist)
    return jnp.where(side < 2, jnp.stack([near, other]), jnp.stack([other, near])).astype(jnp.int32)


def make(spec, start="uniform"):
    """reset(key) -> (obs, state); step(state, action, key) -> (obs, state, reward, done).
    `start` is the reset distribution: "uniform" (the data's [16, 42]^2) or "edge" (edge_pos)."""

    def reset(key, episode=0):
        k_pos, k_tex, k_frame = jax.random.split(key, 3)
        pos = edge_pos(k_pos) if start == "edge" else jax.random.randint(k_pos, (2,), START_LO, START_HI + 1)
        state = State(pos, jnp.zeros(2, jnp.int32), k_tex, jnp.asarray(0, jnp.int32), jnp.asarray(episode, jnp.int32),
                      jnp.asarray(NO_WALL, jnp.int32), jnp.asarray(0, jnp.int32))
        return render(spec, state, k_frame), state

    def step(state, action, key):
        k_wind, k_frame, k_gap = jax.random.split(key, 3)
        wind = R.wind_walk(state.wind, k_wind) if spec.wind else state.wind
        pos = state.pos + MOVES[action] + wind
        t = state.step + 1
        if spec.gate:
            moved = jnp.where(state.wall_x == NO_WALL, NO_WALL, state.wall_x - WALL_SPEED)
            moved = jnp.where(moved < BORDER + 1 - WALL_W + 1, NO_WALL, moved)
            spawn = (moved == NO_WALL) & (t % WALL_PERIOD == 0)
            wall_x = jnp.where(spawn, WALL_ENTER, moved)
            gap = jnp.where(spawn, jax.random.bernoulli(k_gap, 0.5).astype(jnp.int32), state.gap)
        else:
            wall_x, gap = state.wall_x, state.gap
        dead = jnp.any(pos < SAFE_LO) | jnp.any(pos > SAFE_HI)
        if spec.gate:
            dead = dead | cube_hits_wall(pos, wall_x, gap)
        state = State(pos, wind, state.texture, t, state.episode, wall_x, gap)
        done = dead | (t >= MAX_STEPS)
        return render(spec, state, k_frame), state, jnp.where(dead, -1.0, 0.0), done

    return reset, step


def batched(spec, start="uniform"):
    reset, step = make(spec, start)
    vreset = jax.vmap(reset)

    def vstep(state, action, key):
        ks, kr = jax.random.split(key)
        n = action.shape[0]
        obs, nstate, reward, done = jax.vmap(step)(state, action, jax.random.split(ks, n))
        robs, rstate = jax.vmap(reset)(jax.random.split(kr, n), nstate.episode + 1)
        pick = lambda r, x: jnp.where(done.reshape((n,) + (1,) * (x.ndim - 1)), r, x)
        return obs, jax.tree.map(pick, rstate, nstate), reward, done, jnp.where(done[:, None, None, None], robs, obs)

    return jax.jit(vreset), jax.jit(vstep)


def collect(spec, envs, steps, seed):
    vreset, vstep = batched(spec)
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    obs, state = vreset(jax.random.split(k0, envs), jnp.zeros(envs, jnp.int32))

    @jax.jit
    def body(carry, k):
        state, obs = carry
        ka, ks = jax.random.split(k)
        action = jax.random.randint(ka, (envs,), 0, N_ACTIONS)
        _, nstate, reward, done, nobs = vstep(state, action, ks)
        return (nstate, nobs), (obs, action, reward, done, state.pos, state.wind, state.episode, state.wall_x, state.gap)

    (state, obs), (frames, actions, rewards, dones, pos, wind, episode, wall_x, gap) = jax.lax.scan(
        body, (state, obs), jax.random.split(key, steps))
    tail = lambda a, last: np.asarray(jnp.concatenate([a, last[None]]))
    return {"frames": tail(frames, obs), "actions": np.asarray(actions, np.int16),
            "rewards": np.asarray(rewards, np.float32), "dones": np.asarray(dones, bool),
            "pos": tail(pos, state.pos).astype(np.int16), "wind": tail(wind, state.wind).astype(np.int8),
            "episode": tail(episode, state.episode).astype(np.int32),
            "wall_x": tail(wall_x, state.wall_x).astype(np.int16), "gap": tail(gap, state.gap).astype(np.int8)}


def save(path, data, spec, split, seed, envs, steps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"env": "stoch", "spec": asdict(spec), "config": spec.name, "num_actions": N_ACTIONS, "envs": envs,
            "steps": steps, "split": split, "seed": seed, "policy": "uniform random", "max_steps": MAX_STEPS,
            "reward": "-1 on touching the border or the wall, else 0"}
    np.savez(path, meta=json.dumps(meta), **data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gate", choices=sorted(CONFIGS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=100)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()
    spec = CONFIGS[args.config]
    if args.preview is not None:
        from PIL import Image

        reset, step = make(spec)
        key = jax.random.PRNGKey(args.seed)
        obs, state = reset(key)
        tiles = [np.asarray(obs)]
        for t in range(40):
            key, ka, ks = jax.random.split(key, 3)
            obs, state, _, done = step(state, jnp.asarray(1 if t % 2 else 3), ks)
            tiles.append(np.asarray(obs))
        row = np.concatenate([np.repeat(np.repeat(t, 2, 0), 2, 1) for t in tiles[::4]], 1)
        Image.fromarray(row).save(args.preview)
        print(f"preview -> {args.preview}", flush=True)
        return
    for split, steps, seed in (("train", args.train_steps, args.seed), ("eval", args.eval_steps, args.seed + 1000)):
        t0 = time.time()
        data = collect(spec, args.envs, steps, seed)
        stats = R.episode_stats(data["dones"], data["rewards"])
        save(args.out / f"{split}.npz", data, spec, split, seed, args.envs, steps)
        print(f"[{spec.name}] {split}: {steps * args.envs} transitions in {time.time() - t0:.1f}s "
              f"{json.dumps(stats)} -> {args.out / f'{split}.npz'}", flush=True)


if __name__ == "__main__":
    main()
