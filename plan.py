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
import probe
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


@functools.partial(nnx.jit, static_argnames=("horizon", "gamma"))
def reward_parts(model, heads, emb_hist, act_hist, plans, horizon, gamma, lam=0.0):
    """Imagined return of each plan, split into collected reward and the bootstrap gamma^H V(z_H).

    The JEPA itself knows nothing about reward: r, done and V are the off-budget heads read out
    of the same frozen latents, along a latent rollout the JEPA produces.
    """
    padded = jnp.concatenate([plans, plans[:, -1:]], 1)

    def step(carry, j):
        emb, acts, alive, total = carry
        reward, logit_done, _ = heads(emb[:, -1], acts[:, -1])
        total = total + (gamma ** j) * alive * probe.pessimism(reward, lam)
        alive = alive * (1.0 - jax.nn.sigmoid(logit_done))
        nxt = model.predict(emb, model.act_embed(acts))[:, -1:]
        follow = jax.lax.dynamic_slice_in_dim(padded, j + 1, 1, axis=1)
        return (jnp.concatenate([emb[:, 1:], nxt], 1),
                jnp.concatenate([acts[:, 1:], follow], 1), alive, total), None

    act_hist = jnp.concatenate([act_hist[:, :-1], plans[:, :1]], 1)
    carry = (emb_hist, act_hist, jnp.ones(emb_hist.shape[0]), jnp.zeros(emb_hist.shape[0]))
    (emb, acts, alive, total), _ = jax.lax.scan(step, carry, jnp.arange(horizon))
    _, _, value = heads(emb[:, -1], acts[:, -1])
    return total, (gamma ** horizon) * alive * probe.pessimism(value, lam)


def batched_reward_cost(model, heads, emb_hist, act_hist, plans, horizon, gamma, lam=0.0,
                        rows=8192):
    out = []
    for i in range(0, plans.shape[0], rows):
        sl = slice(i, i + rows)
        total, boot = reward_parts(model, heads, emb_hist[sl], act_hist[sl], plans[sl],
                                   horizon=horizon, gamma=gamma, lam=lam)
        out.append(-(total + boot))
    return jnp.concatenate(out)


def cem(model, emb_hist, act_hist, goal, horizon, key, pop=512, elite=32, iters=6, temperature=1.0,
        heads=None, gamma=0.99, lam=0.0):
    """Cross-entropy method over discrete action sequences in latent space.

    With `heads` the objective is the imagined return; otherwise it is LeWM's goal-latent cost.
    """
    n = emb_hist.shape[0]
    logits = jnp.zeros((n, horizon, model.n_actions))
    best_plan, best_cost = None, None
    for _ in range(iters):
        key, sub = jax.random.split(key)
        plans = jax.random.categorical(sub, logits[:, None] / temperature, axis=-1, shape=(n, pop, horizon))
        flat = plans.reshape(n * pop, horizon)
        if heads is None:
            cost = batched_cost(model, jnp.repeat(emb_hist, pop, 0), jnp.repeat(act_hist, pop, 0),
                                flat, jnp.repeat(goal, pop, 0), horizon).reshape(n, pop)
        else:
            cost = batched_reward_cost(model, heads, jnp.repeat(emb_hist, pop, 0),
                                       jnp.repeat(act_hist, pop, 0), flat, horizon, gamma,
                                       lam).reshape(n, pop)
        order = jnp.argsort(cost, axis=1)
        top = jnp.take_along_axis(plans, order[:, :elite, None], axis=1)
        counts = jax.nn.one_hot(top, model.n_actions).sum(1)
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
                                      0, model.n_actions)
    random_cost = batched_cost(model, jnp.repeat(emb_hist, pop, 0), jnp.repeat(act_hist, pop, 0),
                               random_plans, jnp.repeat(goal, pop, 0), horizon).reshape(len(frames), pop)
    percentile = float(jnp.mean(random_cost < true_cost[:, None]))
    return {"windows": int(len(frames)), "horizon": horizon,
            "first_action_accuracy": float(jnp.mean(plan[:, 0] == truth[:, 0])),
            "sequence_accuracy": float(jnp.mean(plan == truth)),
            "chance": 1.0 / model.n_actions,
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
        action = jax.random.randint(act_key, (envs,), 0, model.n_actions)
        obs, state, _, _, _ = step(jax.random.split(step_key, envs), state, action)
        frames.append(to_u8(obs))
        actions.append(np.asarray(action))
    frames = np.stack(frames, 1)
    actions = np.stack(actions + [np.zeros(envs, np.int32)], 1)

    key, goal_key = jax.random.split(key)
    goal_plan = jax.random.randint(goal_key, (envs, horizon), 0, model.n_actions)
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
    random_plan = jax.random.randint(jax.random.PRNGKey(seed + 9), (envs, horizon), 0, model.n_actions)

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


def craftax():
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
    params = env.default_params
    return (jax.jit(jax.vmap(lambda k: env.reset(k, params))),
            jax.jit(jax.vmap(lambda k, s, a: env.step(k, s, a, params))))


def reward_loop(model, heads, envs, steps, horizon, seed, pop, iters, gamma, lam=0.0, warmup=CTX):
    """Model-predictive control for *reward* in the real game, against matched baselines.

    Each policy starts from the same post-warmup states and sees the same environment noise;
    the only difference is how the next action is chosen.
    """
    reset, env_step = craftax()
    to_u8 = lambda x: jnp.clip(x * 255.0, 0, 255).astype(jnp.uint8)
    hist = model.history

    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)
    obs, state = reset(jax.random.split(sub, envs))
    frames, acts = [to_u8(obs)], [jnp.zeros(envs, jnp.int32)]
    for _ in range(warmup - 1):
        key, act_key, step_key = jax.random.split(key, 3)
        action = jax.random.randint(act_key, (envs,), 0, model.n_actions)
        obs, state, _, _, _ = env_step(jax.random.split(step_key, envs), state, action)
        frames.append(to_u8(obs))
        acts.append(action)
    start_emb = viz.encode(model, jnp.stack(frames[-hist:], 1))
    start_act = jnp.concatenate([jnp.stack(acts[-hist + 1:], 1), jnp.zeros((envs, 1), jnp.int32)], 1)
    step_keys = jax.random.split(key, steps * envs).reshape(steps, envs, -1)
    single = jnp.tile(jnp.arange(model.n_actions, dtype=jnp.int32)[None, :, None], (envs, 1, 1))

    results, traces = {}, {}
    for name in ("random", "myopic", "greedy", "plan"):
        emb_hist, act_hist, run_state = start_emb, start_act, state
        alive = jnp.ones(envs)
        reward_log, alive_log, imagined = [], [], []
        pkey = jax.random.PRNGKey(seed + 17)
        for t in range(steps):
            pkey, sub = jax.random.split(pkey)
            if name == "random":
                action = jax.random.randint(sub, (envs,), 0, model.n_actions)
                plan = action[:, None]
            elif name in ("myopic", "greedy"):
                # myopic sees only the reward head, greedy adds one imagined step and the value.
                cost = batched_reward_cost(model, heads, jnp.repeat(emb_hist, model.n_actions, 0),
                                           jnp.repeat(act_hist, model.n_actions, 0),
                                           single.reshape(-1, 1), 1,
                                           gamma if name == "greedy" else 0.0,
                                           lam).reshape(envs, -1)
                action = jnp.argmin(cost, 1).astype(jnp.int32)
                plan = action[:, None]
            else:
                plan, _ = cem(model, emb_hist, act_hist, None, horizon, sub, pop=pop,
                              iters=iters, heads=heads, gamma=gamma, lam=lam)
                action = plan[:, 0]
            imagined.append(reward_parts(model, heads, emb_hist, act_hist, plan,
                                         horizon=plan.shape[1], gamma=gamma, lam=lam)[0])
            obs, run_state, reward, done, _ = env_step(step_keys[t], run_state, action)
            reward_log.append(reward * alive)
            alive_log.append(alive)
            alive = alive * (1.0 - done.astype(jnp.float32))
            emb_hist = jnp.concatenate([emb_hist[:, 1:], viz.encode(model, to_u8(obs)[:, None])], 1)
            act_hist = jnp.concatenate([act_hist[:, 1:-1], action[:, None],
                                        jnp.zeros((envs, 1), jnp.int32)], 1)
        rewards = np.asarray(jnp.stack(reward_log, 1))
        traces[name] = rewards
        results[name] = {"return": float(rewards.sum(1).mean()),
                         "return_std": float(rewards.sum(1).std()),
                         "reward_per_step": float(rewards.mean()),
                         "alive_at_end": float(np.asarray(alive).mean()),
                         "imagined_reward_mean": float(jnp.stack(imagined, 1).mean())}
        reach = horizon if name == "plan" else 1
        if name != "random" and steps > reach:
            discount = gamma ** np.arange(reach)
            realized = np.stack([(rewards[:, t:t + reach] * discount).sum(1)
                                 for t in range(steps - reach)], 1)
            planned = np.asarray(jnp.stack(imagined, 1))[:, :steps - reach]
            if realized.std() > 0 and planned.std() > 0:
                results[name]["imagined_vs_realized_corr"] = float(
                    np.corrcoef(planned.ravel(), realized.ravel())[0, 1])
        print(f"[reward] {name}: {json.dumps(results[name])}", flush=True)
    results.update({"envs": envs, "steps": steps, "horizon": horizon, "gamma": gamma, "lam": lam})
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
    parser.add_argument("--reward", action="store_true", help="MPC for reward instead of a goal image")
    parser.add_argument("--no-goal", action="store_true", help="skip the goal-image experiments")
    parser.add_argument("--steps", type=int, default=100, help="real environment steps per episode")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.0, help="ensemble disagreement penalty")
    parser.add_argument("--train-data", type=Path, default=Path("data/tinywmcraft_500k/train.npz"))
    args = parser.parse_args()

    bundle, meta = T.load_model(args.model)
    model = bundle.model
    tag = args.tag or args.model.parent.name
    report = {"tag": tag}
    if not args.no_goal:
        report["offline"] = offline(model, args.data, args.windows, args.horizon,
                                    args.seed, args.pop, args.iters)
        if args.closed_loop:
            report["craftax"] = closed_loop(model, args.envs, args.horizon, args.seed,
                                            args.pop, args.iters)
    if args.reward:
        if bundle.heads is None:
            raise RuntimeError("this checkpoint has no reward heads; run probe.py --heads on it")
        report["reward_mpc"] = reward_loop(model, bundle.heads, args.envs, args.steps, args.horizon,
                                           args.seed, args.pop, args.iters, args.gamma, args.lam)
        with np.load(args.train_data) as archive:
            report["reward_mpc"]["behaviour_policy"] = float(archive["rewards"].mean()) * args.steps
        report["reward_mpc"]["heads"] = meta["heads"]
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "_reward" if args.reward and args.no_goal else "_plan"
    (args.out / f"{tag}{suffix}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
