"""Audit of the off-budget heads: what they read on real latents, split by the cause of death, and what they read
on the model's own rollouts (deterministic predictor, residual head as the mean of 8 samples, one sample), where
the planner's imagined return comes from (the true return is never positive: the only reward is -1 at death).

    XLA_PYTHON_CLIENT_MEM_FRACTION=.2 python -m studies.stoch.heads_audit --config gate --data .../eval.npz \
        --model .../tile_factored/model_probe.npz --sampler .../tile_factored/sampler.npz \
        --out runs/stoch/gate/heads_audit_tile_factored.json
"""
import argparse
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import probe
import train as T
from studies.stoch import env as S
from studies.stoch import sampler as SM
from studies.stoch.wallmass import sampled_rollout

CTX, HOR, GAMMA = 8, 8, 0.99


def auc(score, positive):
    """Probability that a positive scores above a negative (rank statistic, ties counted half)."""
    score, positive = np.asarray(score, np.float64), np.asarray(positive, bool)
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(score))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@nnx.jit
def read(heads, z, a):
    r, d, v = heads(z, a)
    return r.mean(-1), jax.nn.sigmoid(d), v.mean(-1)


def chunked(heads, z, a, size=8192):
    outs = [read(heads, jnp.asarray(z[i:i + size]), jnp.asarray(a[i:i + size])) for i in range(0, len(z), size)]
    return tuple(np.concatenate([np.asarray(o[k]) for o in outs]) for k in range(3))


def real_latents(bundle, d):
    frames, actions, rewards = d["frames"], d["actions"].astype(np.int32), d["rewards"]
    pos, wall_x, gap = d["pos"].astype(np.int32), d["wall_x"].astype(np.int32), d["gap"].astype(np.int32)
    Tn, E = actions.shape
    Z = np.asarray(probe.embed_all(bundle.model, frames)).reshape(Tn + 1, E, -1)
    z, a = Z[:Tn].reshape(Tn * E, -1), actions.reshape(-1)
    r_pred, p_done, value = chunked(bundle.heads, z, a)
    death = rewards.reshape(-1) < 0
    nxt = pos[:Tn].reshape(-1, 2) + np.asarray(S.MOVES)[a]
    wx = wall_x[:Tn].reshape(-1)
    moved = np.where(wx == S.NO_WALL, S.NO_WALL, wx - S.WALL_SPEED)
    hits = np.asarray(jax.vmap(S.cube_hits_wall)(jnp.asarray(nxt), jnp.asarray(moved), jnp.asarray(gap[:Tn].reshape(-1))))
    wall, border, alive = death & hits, death & ~hits, ~death
    return {"transitions": int(Tn * E), "deaths": int(death.sum()), "wall_deaths": int(wall.sum()),
            "border_deaths": int(border.sum()),
            "reward_corr": float(np.corrcoef(r_pred, -death.astype(np.float32))[0, 1]),
            "reward_pred_mean": float(r_pred.mean()), "reward_pred_positive_fraction": float((r_pred > 0.01).mean()),
            "reward_pred_on_wall_deaths": float(r_pred[wall].mean()), "reward_pred_on_border_deaths": float(r_pred[border].mean()),
            "reward_pred_on_alive": float(r_pred[alive].mean()),
            "done_auc_all": auc(p_done, death), "done_auc_wall_vs_alive": auc(p_done[~border], wall[~border]),
            "done_auc_border_vs_alive": auc(p_done[~wall], border[~wall]),
            "p_done_wall_deaths": float(p_done[wall].mean()), "p_done_border_deaths": float(p_done[border].mean()),
            "p_done_alive": float(p_done[alive].mean()),
            "value_mean": float(value.mean()), "value_positive_fraction": float((value > 0.01).mean())}


def imagined(bundle, sampler, d, n, seed):
    frames, actions, dones = d["frames"], d["actions"].astype(np.int32), d["dones"]
    table = T.valid_starts(dones.astype(np.int64), CTX + HOR)
    pick = table[np.random.default_rng(seed).choice(len(table), n, replace=False)]
    ctx = jnp.asarray(np.stack([frames[s:s + CTX, e] for s, e in pick]))
    acts = jnp.asarray(np.stack([actions[s:s + CTX + HOR, e] for s, e in pick]))
    rollouts = {"deterministic": np.asarray(T._rollout_latents(bundle.model, ctx, acts, horizon=HOR)[1])}
    if sampler is not None:
        rollouts["sample"] = np.asarray(sampled_rollout(sampler, ctx, acts, HOR, jax.random.PRNGKey(seed)))
        sampler.mean_of = 8
        rollouts["mean_of_8"] = np.asarray(sampled_rollout(sampler, ctx, acts, HOR, jax.random.PRNGKey(seed + 1)))
        sampler.mean_of = 0
    future_actions = np.asarray(acts[:, CTX - 1:CTX - 1 + HOR]).reshape(-1)
    disc = GAMMA ** np.arange(HOR)
    out = {"windows": int(n)}
    for name, Z in rollouts.items():
        r, p, v = chunked(bundle.heads, Z.reshape(-1, Z.shape[-1]), future_actions)
        r, p, v = (x.reshape(n, HOR) for x in (r, p, v))
        alive = np.concatenate([np.ones((n, 1)), np.cumprod(1.0 - p[:, :-1], 1)], 1)
        reward_part = (disc * alive * r).sum(1)
        value_part = GAMMA ** HOR * alive[:, -1] * (1.0 - p[:, -1]) * v[:, -1]
        out[name] = {"reward_pred_mean_per_step": r.mean(0).tolist(),
                     "reward_pred_positive_fraction": float((r > 0.01).mean()),
                     "p_done_mean_per_step": p.mean(0).tolist(), "value_mean_per_step": v.mean(0).tolist(),
                     "imagined_return_mean": float((reward_part + value_part).mean()),
                     "imagined_return_positive_fraction": float(((reward_part + value_part) > 0).mean()),
                     "reward_part_mean": float(reward_part.mean()), "value_part_mean": float(value_part.mean())}
    return out


def planner_view(bundle, d, n, seed, horizon=3, chunk=16):
    """What the heads-MPC planner scores: the imagined return of every one of the 4^horizon plans from n real
    contexts through the deterministic predictor (control.py's cost, negated), the mean over plans against the
    max the planner picks (control.py reports -min(cost), the best of the plans)."""
    frames, actions, dones = d["frames"], d["actions"].astype(np.int32), d["dones"]
    table = T.valid_starts(dones.astype(np.int64), CTX + horizon)
    pick = table[np.random.default_rng(seed + 2).choice(len(table), n, replace=False)]
    P = S.N_ACTIONS ** horizon
    plans = np.stack([(np.arange(P) // S.N_ACTIONS ** (horizon - 1 - j)) % S.N_ACTIONS for j in range(horizon)], 1)
    disc = GAMMA ** np.arange(horizon)
    means, maxes, best_first = [], [], []
    for i in range(0, n, chunk):
        sub = pick[i:i + chunk]
        m = len(sub)
        ctx = np.repeat(np.stack([frames[s:s + CTX, e] for s, e in sub]), P, 0)
        acts = np.zeros((m * P, CTX + horizon), np.int32)
        acts[:, :CTX - 1] = np.repeat(np.stack([actions[s:s + CTX - 1, e] for s, e in sub]), P, 0)
        acts[:, CTX - 1:CTX - 1 + horizon] = np.tile(plans, (m, 1))
        Z = np.asarray(T._rollout_latents(bundle.model, jnp.asarray(ctx), jnp.asarray(acts), horizon=horizon)[1])
        r, p, v = chunked(bundle.heads, Z.reshape(-1, Z.shape[-1]), acts[:, CTX - 1:CTX - 1 + horizon].reshape(-1))
        r, p, v = (x.reshape(m, P, horizon) for x in (r, p, v))
        alive = np.concatenate([np.ones((m, P, 1)), np.cumprod(1.0 - p[:, :, :-1], 2)], 2)
        score = (disc * alive * r).sum(2) + GAMMA ** horizon * alive[:, :, -1] * (1.0 - p[:, :, -1]) * v[:, :, -1]
        means.append(score.mean(1))
        maxes.append(score.max(1))
        best_first.append(plans[score.argmax(1), 0])
    means, maxes, best_first = (np.concatenate(x) for x in (means, maxes, best_first))
    return {"horizon": horizon, "plans": int(P), "contexts": int(n),
            "mean_over_plans": float(means.mean()), "max_over_plans": float(maxes.mean()),
            "max_over_plans_positive_fraction": float((maxes > 0).mean()),
            "mean_over_plans_positive_fraction": float((means > 0).mean()),
            "chosen_first_action_histogram": np.bincount(best_first, minlength=S.N_ACTIONS).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="gate", choices=sorted(S.CONFIGS))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--sampler", type=Path)
    p.add_argument("--windows", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    bundle, meta = T.load_model(a.model)
    sampler = SM.load(bundle, a.sampler, 8)[0] if a.sampler else None
    d = np.load(a.data)
    result = {"model": str(a.model), "sampler": str(a.sampler) if a.sampler else None, "heads": meta.get("heads"),
              "real": real_latents(bundle, d), "imagined": imagined(bundle, sampler, d, a.windows, a.seed),
              "planner": planner_view(bundle, d, a.windows, a.seed)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "heads"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
