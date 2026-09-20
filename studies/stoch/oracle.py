"""Oracle planner for the gate world: the exhaustive horizon-h search of studies.stoch.control, replanned every
step, but rolled out through the TRUE simulator state instead of a learned model.

It measures the ceiling of this planner class - a perfect one-step model and the true reward - so the only thing
left is the world's own stochasticity: the gap of a wall that has not yet reached REVEAL_X is a coin that has not
been flipped, and the planner averages the cost of every plan over K belief copies that each draw their own gap.
With --leak the copies keep the true gap, the cheating reference that reads the coin before it is flipped.

Protocol, key splitting and result rows are those of studies.stoch.control.run (same start states, same per-step
environment keys), so the rows are paired with the model controllers and the `random` row reproduces control.py's.

    python -m studies.stoch.oracle --horizon 5 --samples 4 --out runs/stoch/gate/oracle_h5_k4.json
"""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from studies.stoch import control as C
from studies.stoch import env as S

MAX_ROLLOUTS = 2 ** 24


def plans_table(horizon):
    """(4^h, h) int32: every action sequence in itertools.product order, the first action slowest."""
    grid = jnp.meshgrid(*[jnp.arange(S.N_ACTIONS, dtype=jnp.int32)] * horizon, indexing="ij")
    return jnp.stack([g.reshape(-1) for g in grid], 1)


def state_step(pos, wall_x, gap, t, action, coin):
    """studies.stoch.env.make(gate).step on the state alone: -> pos, wall_x, gap, t, dead."""
    pos = pos + S.MOVES[action]
    t = t + 1
    moved = jnp.where(wall_x == S.NO_WALL, S.NO_WALL, wall_x - S.WALL_SPEED)
    moved = jnp.where(moved < S.BORDER + 1 - S.WALL_W + 1, S.NO_WALL, moved)
    spawn = (moved == S.NO_WALL) & (t % S.WALL_PERIOD == 0)
    wall_x = jnp.where(spawn, S.WALL_ENTER, moved)
    gap = jnp.where(spawn, coin, gap)
    dead = (jnp.any(pos < S.SAFE_LO, -1) | jnp.any(pos > S.SAFE_HI, -1)
            | jax.vmap(S.cube_hits_wall)(pos, wall_x, gap))
    return pos, wall_x, gap, t, dead


@jax.jit
def search(pos, wall_x, gap, t, plans, gamma, key):
    """(n, 4^h) cost: minus the discounted return of every plan, averaged over the K belief copies."""
    n, k = wall_x.shape
    p, h = plans.shape
    b = n * p * k
    grow = lambda a: jnp.broadcast_to(a[:, None], (n, p) + a.shape[1:]).reshape((b,) + a.shape[2:])
    pos, wall_x, gap, t = grow(pos), grow(wall_x), grow(gap), grow(t)
    total, alive = jnp.zeros(b), jnp.ones(b)
    for j in range(h):
        key, kc = jax.random.split(key)
        action = jnp.broadcast_to(plans[None, :, j, None], (n, p, k)).reshape(b)
        coin = jax.random.bernoulli(kc, 0.5, (b,)).astype(jnp.int32)
        pos, wall_x, gap, t, dead = state_step(pos, wall_x, gap, t, action, coin)
        total = total + (gamma ** j) * alive * jnp.where(dead, -1.0, 0.0)
        alive = alive * (1.0 - dead.astype(jnp.float32))
    return -total.reshape(n, p, k).mean(-1)


def belief(state, samples, key, leak):
    """K copies of the true state; the gap of a wall that has not reached REVEAL_X is redrawn in every copy."""
    envs = state.pos.shape[0]
    pos = jnp.broadcast_to(state.pos[:, None], (envs, samples, 2))
    wall_x = jnp.broadcast_to(state.wall_x[:, None], (envs, samples))
    t = jnp.broadcast_to(state.step[:, None], (envs, samples))
    true_gap = jnp.broadcast_to(state.gap[:, None], (envs, samples))
    if leak:
        return pos, wall_x, true_gap, t
    coin = jax.random.bernoulli(key, 0.5, (envs, samples)).astype(jnp.int32)
    hidden = (wall_x != S.NO_WALL) & (wall_x > S.REVEAL_X)
    return pos, wall_x, jnp.where(hidden, coin, true_gap), t


def plan_cost(state, plans, gamma, samples, leak, key):
    """The search over all plans, chunked over envs so a chunk holds at most MAX_ROLLOUTS rollouts."""
    envs = state.pos.shape[0]
    rows = max(1, min(envs, MAX_ROLLOUTS // (plans.shape[0] * samples)))
    kb, kr = jax.random.split(key)
    pos, wall_x, gap, t = belief(state, samples, kb, leak)
    out = [search(pos[i:i + rows], wall_x[i:i + rows], gap[i:i + rows], t[i:i + rows], plans, gamma,
                  jax.random.fold_in(kr, i))
           for i in range(0, envs, rows)]
    return jnp.concatenate(out)


def pick(cost, key):
    """Index of a minimum-cost plan, uniformly random among all plans within 1e-9 of the minimum."""
    tie = cost <= jnp.min(cost, 1, keepdims=True) + 1e-9
    return jnp.argmax(jnp.where(tie, jax.random.uniform(key, cost.shape), -1.0), 1)


def run(spec, envs, steps, horizon, seed, gamma, samples, leak):
    key = jax.random.PRNGKey(seed)
    key, kw, kg, ks = jax.random.split(key, 4)
    frames, acts, state, vstep = C.warm_up(spec, envs, kw, C.CTX, "uniform")
    step_keys = jax.random.split(ks, steps * envs).reshape(steps, envs, -1)
    plans = plans_table(horizon)
    first = C.plan_index_first_action(horizon)
    results = {}
    for name in ("oracle", "random"):
        t0 = time.time()
        run_state = state
        alive = jnp.ones(envs)
        running = jnp.ones(envs)
        length = jnp.full((envs,), C.CTX - 1, jnp.int32)
        walls = jnp.zeros(envs, jnp.int32)
        pkey = jax.random.PRNGKey(seed + 17)
        for t in range(steps):
            pkey, sub, ksearch = jax.random.split(pkey, 3)
            if name == "random":
                action = jax.random.randint(sub, (envs,), 0, S.N_ACTIONS)
            else:
                kb, ktie = jax.random.split(ksearch)
                action = first[pick(plan_cost(run_state, plans, gamma, samples, leak, kb), ktie)]
            prev_wall = run_state.wall_x
            obs, run_state, reward, done = vstep(run_state, action, step_keys[t])
            length = length + running.astype(jnp.int32)
            walls = walls + (running * ((prev_wall != S.NO_WALL) & (run_state.wall_x == S.NO_WALL))).astype(jnp.int32)
            alive = alive * (1.0 - (reward < 0).astype(jnp.float32))
            running = running * (1.0 - done.astype(jnp.float32))
        length, walls, alive = np.asarray(length), np.asarray(walls), np.asarray(alive)
        results[name] = {"survival": float(alive.mean()), "survival_sem": float(alive.std() / np.sqrt(envs)),
                         "length": float(length.mean()),
                         "deaths_per_100": float(100.0 * (1.0 - alive).sum() / length.sum()),
                         "walls_passed": float(walls.mean()), "seconds": round(time.time() - t0, 1),
                         "lengths": length.tolist()}
        print(f"[oracle] {name}: " + json.dumps({k: v for k, v in results[name].items() if k != "lengths"}), flush=True)
    results.update({"horizon": horizon, "samples": samples, "envs": envs, "steps": steps, "gamma": gamma,
                    "leak": leak,
                    "protocol": "lengths from reset, cap is not a death, unrevealed gap resampled per copy"})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--samples", type=int, default=4, help="belief copies K per env")
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=S.MAX_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--leak", action="store_true", help="keep the true gap in every copy (cheating reference)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = run(S.CONFIGS["gate"], args.envs, args.steps, args.horizon, args.seed, args.gamma, args.samples,
                 args.leak)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
