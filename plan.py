"""Can we plan inside the JEPA?  Latent-space CEM, offline and closed-loop in Craftax.

The cost is the one LeWM uses for control (`JEPA.criterion`): squared distance between the
predicted latent and the latent of a goal image.  No decoder and no pixels are involved.
"""
import argparse
import functools
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import eval as E
import train as T
import viz

CTX = E.CTX


@functools.partial(nnx.jit, static_argnames="horizon")
def plan_cost(model, emb_hist, act_hist, plans, goal, horizon):
    """Latent cost of each action plan: emb_hist (N,h,D), act_hist (N,h), plans (N,horizon)."""
    padded = jnp.concatenate([plans, plans[:, -1:]], 1)

    def step(carry, j):
        emb, acts = carry
        nxt = model.predict(emb, model.act_embed(acts))[:, -1:]
        follow = jax.lax.dynamic_slice_in_dim(padded, j + 1, 1, axis=1)
        return (jnp.concatenate([emb[:, 1:], nxt], 1), jnp.concatenate([acts[:, 1:], follow], 1)), nxt[:, 0]

    act_hist = jnp.concatenate([act_hist[:, :-1], plans[:, :1]], 1)
    (_, _), states = jax.lax.scan(step, (emb_hist, act_hist), jnp.arange(horizon))
    final = states[-1]
    return jnp.mean(jnp.square(final - goal), -1), final


def batched_cost(model, emb_hist, act_hist, plans, goal, horizon, rows=8192):
    """Evaluate plan costs in row chunks; a grid latent makes the full population too large."""
    out = []
    for i in range(0, plans.shape[0], rows):
        sl = slice(i, i + rows)
        out.append(plan_cost(model, emb_hist[sl], act_hist[sl], plans[sl], goal[sl], horizon=horizon)[0])
    return jnp.concatenate(out)


def cem(model, emb_hist, act_hist, goal, horizon, key, pop=512, elite=32, iters=6, temperature=1.0):
    """Cross-entropy method over discrete action sequences in latent space."""
    n = emb_hist.shape[0]
    logits = jnp.zeros((n, horizon, T.N_ACTIONS))
    best_plan, best_cost = None, None
    for _ in range(iters):
        key, sub = jax.random.split(key)
        plans = jax.random.categorical(sub, logits[:, None] / temperature, axis=-1, shape=(n, pop, horizon))
        flat = plans.reshape(n * pop, horizon)
        cost = batched_cost(model, jnp.repeat(emb_hist, pop, 0), jnp.repeat(act_hist, pop, 0),
                            flat, jnp.repeat(goal, pop, 0), horizon).reshape(n, pop)
        order = jnp.argsort(cost, axis=1)
        top = jnp.take_along_axis(plans, order[:, :elite, None], axis=1)
        counts = jax.nn.one_hot(top, T.N_ACTIONS).sum(1)
        logits = jnp.log(counts + 0.5)
        winner = order[:, 0]
        cost_now = jnp.take_along_axis(cost, winner[:, None], 1)[:, 0]
        plan_now = jnp.take_along_axis(plans, winner[:, None, None], 1)[:, 0]
        if best_cost is None:
            best_plan, best_cost = plan_now, cost_now
        else:
            better = cost_now < best_cost
            best_plan = jnp.where(better[:, None], plan_now, best_plan)
            best_cost = jnp.where(better, cost_now, best_cost)
    return best_plan, best_cost


def context_state(model, frames, actions):
    """Latent history and its outgoing actions at the end of an 8-frame context."""
    hist = model.history
    emb = viz.encode(model, jnp.asarray(frames[:, :CTX]))
    act_hist = jnp.asarray(actions[:, CTX - hist + 1:CTX + 1], jnp.int32)
    return emb[:, -hist:], act_hist


def offline(model, data_path, n, horizon, seed, pop, iters):
    """Recover the true action sequence by planning towards the real future frame."""
    data = E.load_data(data_path)
    frames, actions = E.window_batch(data, E.sample_indices(data, n))
    emb_hist, act_hist = context_state(model, frames, actions)
    goal = viz.encode(model, jnp.asarray(frames[:, CTX + horizon - 1][:, None]))[:, 0]
    truth = jnp.asarray(actions[:, CTX:CTX + horizon], jnp.int32)
    key = jax.random.PRNGKey(seed)
    plan, cost = cem(model, emb_hist, act_hist, goal, horizon, key, pop=pop, iters=iters)
    true_cost, _ = plan_cost(model, emb_hist, act_hist, truth, goal, horizon=horizon)
    random_plans = jax.random.randint(jax.random.PRNGKey(seed + 1), (len(frames) * pop, horizon),
                                      0, T.N_ACTIONS)
    random_cost = batched_cost(model, jnp.repeat(emb_hist, pop, 0), jnp.repeat(act_hist, pop, 0),
                               random_plans, jnp.repeat(goal, pop, 0), horizon).reshape(len(frames), pop)
    percentile = float(jnp.mean(random_cost < true_cost[:, None]))
    return {"windows": int(len(frames)), "horizon": horizon,
            "first_action_accuracy": float(jnp.mean(plan[:, 0] == truth[:, 0])),
            "sequence_accuracy": float(jnp.mean(plan == truth)),
            "chance": 1.0 / T.N_ACTIONS,
            "true_sequence_percentile_among_random": percentile,
            "cost_plan": float(jnp.mean(cost)), "cost_true": float(jnp.mean(true_cost)),
            "cost_random": float(jnp.mean(random_cost))}


def closed_loop(model, envs, horizon, seed, pop, iters, warmup=CTX):
    """Plan in latent space, execute in the real Craftax environment, land on the goal."""
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
    params = env.default_params
    reset = jax.jit(jax.vmap(lambda k: env.reset(k, params)))
    step = jax.jit(jax.vmap(lambda k, s, a: env.step(k, s, a, params)))
    to_u8 = lambda x: np.asarray(jnp.clip(x * 255.0, 0, 255).astype(jnp.uint8))

    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)
    obs, state = reset(jax.random.split(sub, envs))
    frames = [to_u8(obs)]
    actions = [np.zeros(envs, np.int32)]
    for _ in range(warmup - 1):
        key, act_key, step_key = jax.random.split(key, 3)
        action = jax.random.randint(act_key, (envs,), 0, T.N_ACTIONS)
        obs, state, _, _, _ = step(jax.random.split(step_key, envs), state, action)
        frames.append(to_u8(obs))
        actions.append(np.asarray(action))
    frames = np.stack(frames, 1)
    actions = np.stack(actions + [np.zeros(envs, np.int32)], 1)

    key, goal_key = jax.random.split(key)
    goal_plan = jax.random.randint(goal_key, (envs, horizon), 0, T.N_ACTIONS)
    step_keys = [jax.random.split(k, envs) for k in jax.random.split(key, horizon)]
    goal_state, goal_obs = state, obs
    for j in range(horizon):
        goal_obs, goal_state, _, _, _ = step(step_keys[j], goal_state, goal_plan[:, j])
    goal_frame = to_u8(goal_obs)
    goal_position = np.asarray(goal_state.player_position)

    emb_hist, act_hist = context_state(model, frames, actions)
    goal_emb = viz.encode(model, jnp.asarray(goal_frame[:, None]))[:, 0]
    plan, _ = cem(model, emb_hist, act_hist, goal_emb, horizon, jax.random.PRNGKey(seed + 5),
                  pop=pop, iters=iters)
    random_plan = jax.random.randint(jax.random.PRNGKey(seed + 9), (envs, horizon), 0, T.N_ACTIONS)

    results = {}
    for name, sequence in (("planned", plan), ("random", random_plan), ("oracle", goal_plan)):
        run_state, run_obs = state, obs
        for j in range(horizon):
            run_obs, run_state, _, _, _ = step(step_keys[j], run_state, sequence[:, j])
        position = np.asarray(run_state.player_position)
        distance = np.abs(position - goal_position).sum(-1)
        pixels = np.mean((to_u8(run_obs).astype(np.float32) - goal_frame.astype(np.float32)) ** 2)
        results[name] = {"position_l1": float(distance.mean()),
                         "reached": float(np.mean(distance == 0)),
                         "pixel_mse": float(pixels)}
    results["envs"] = envs
    results["horizon"] = horizon
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/tinywmcraft_500k/eval.npz"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--pop", type=int, default=512)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--closed-loop", action="store_true")
    args = parser.parse_args()

    (model, _), meta = T.load_model(args.model)
    tag = args.tag or args.model.parent.name
    report = {"tag": tag, "offline": offline(model, args.data, args.windows, args.horizon,
                                             args.seed, args.pop, args.iters)}
    if args.closed_loop:
        report["craftax"] = closed_loop(model, args.envs, args.horizon, args.seed, args.pop, args.iters)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{tag}_plan.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
