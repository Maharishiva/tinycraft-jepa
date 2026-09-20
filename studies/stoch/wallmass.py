"""How much of the wall survives each model's rollout, and does the flow model split the gap 50/50?

From N pre-reveal contexts of the gate evaluation set (last context frame has the wall solid at x in {42, 45}),
roll every model 8 steps under the recorded actions and measure, per step, the white-pixel mass of the decoded
frame relative to the real frame (the wall is the only white thing), plus for the flow model which gap each
sample shows (white mass in the top band vs the bottom band) against the recorded one.

    XLA_PYTHON_CLIENT_MEM_FRACTION=.12 python -m studies.stoch.wallmass --data /dev/shm/stoch/gate/eval.npz \
        --lewm /dev/shm/stoch/gate/lewm6/model_probe.npz --grid /dev/shm/stoch/gate/tile_factored/model_probe.npz \
        --flow /dev/shm/stoch/gate/flow_model.npz --flow-code studies/stoch/flow_recipe --out runs/stoch/gate/wallmass.json
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import train as T
import viz
from studies.redcube import flow as F
from studies.stoch import env as S
from studies.stoch import sampler as SM

CTX, HOR = 8, 8


def sampled_rollout(sampler, context, out_actions, horizon, key):
    """The JEPA rollout of train._rollout_latents with the sampled next latent in place of the mean."""
    emb = sampler.encode(context)
    hist = sampler.history
    n = context.shape[1]
    emb_hist, act_hist = emb[:, -hist:], out_actions[:, n - hist:n]
    preds = []
    for j in range(horizon):
        key, k = jax.random.split(key)
        nxt = sampler.sample(emb_hist, act_hist, k)
        preds.append(nxt)
        emb_hist = jnp.concatenate([emb_hist[:, 1:], nxt[:, None]], 1)
        act_hist = jnp.concatenate([act_hist[:, 1:], out_actions[:, n + j:n + j + 1]], 1)
    return jnp.stack(preds, 1)


def white(frames):
    return (frames.astype(np.int16).min(-1) > 200).astype(np.float32)


def gap_side(frames):
    """+1 top gap, -1 bottom gap, 0 undecided, from the white mass in the two gap bands of the wall column."""
    w = white(frames)
    (t0, t1), (b0, b1) = S.GAP_ROWS
    top, bottom = w[..., t0:t1, :].sum((-2, -1)), w[..., b0:b1, :].sum((-2, -1))
    return np.sign(bottom - top)


def contexts(data, n, seed):
    """(t, e) pairs whose frame t has the wall solid at x in {42, 45}, with no reset in the 8 + 8 window."""
    wall_x, dones = data["wall_x"], data["dones"]
    T_, E = dones.shape
    ok = np.zeros((T_, E), bool)
    ok[CTX - 1:T_ - HOR - 1] = np.isin(wall_x[CTX - 1:T_ - HOR - 1], (42, 45))
    idx = np.argwhere(ok)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    picked = []
    for t, e in idx:
        if dones[t - CTX + 1:t + HOR, e].any():
            continue
        picked.append((t, e))
        if len(picked) == n:
            break
    return np.asarray(picked)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--lewm", type=Path, required=True)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--flow", type=Path, required=True)
    p.add_argument("--flow-code", type=Path, required=True)
    p.add_argument("--grid-sampler", type=Path, help="flow head on the grid JEPA's residual: adds the sampled grid rollout")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    d = np.load(a.data)
    frames, actions, gap = d["frames"], d["actions"].astype(np.int32), d["gap"]
    picked = contexts(d, a.n, 0)
    ts, es = picked[:, 0], picked[:, 1]
    ctx = np.stack([frames[t - CTX + 1:t + 1, e] for t, e in picked])
    future = np.stack([frames[t + 1:t + 1 + HOR, e] for t, e in picked])
    taken = np.stack([actions[t - CTX + 1:t + 1 + HOR, e] for t, e in picked])
    out_actions = taken
    incoming_ctx = np.stack([actions[t - CTX:t, e] for t, e in picked])
    incoming_future = np.stack([actions[t:t + HOR, e] for t, e in picked])
    true_side = gap_side(future[:, -1])
    real_mass = white(future).sum((-2, -1))
    result = {"n": int(len(picked)), "real_white_mass_per_step": real_mass.mean(0).tolist(),
              "true_gap_top_fraction": float((true_side > 0).mean())}
    for name, path in (("lewm", a.lewm), ("grid", a.grid)):
        bundle, _ = T.load_model(path)
        _, preds = T._rollout_latents(bundle.model, jnp.asarray(ctx), jnp.asarray(out_actions), horizon=HOR)
        dec = np.asarray(viz.decode(bundle.decoder, preds, HOR))
        enc = np.asarray(viz.decode(bundle.decoder, viz.encode(bundle.model, jnp.asarray(future)), HOR))
        result[name] = {"rollout_white_mass_ratio": (white(dec).sum((-2, -1)) / np.maximum(real_mass, 1)).mean(0).tolist(),
                        "encoded_white_mass_ratio": (white(enc).sum((-2, -1)) / np.maximum(real_mass, 1)).mean(0).tolist(),
                        "rollout_gap_side_top": float((gap_side(dec[:, -1]) > 0).mean()),
                        "rollout_gap_side_undecided": float((gap_side(dec[:, -1]) == 0).mean())}
    def sampled_stats(videos):
        masses, sides = [], []
        for video in videos:
            masses.append(white(video).sum((-2, -1)) / np.maximum(real_mass, 1))
            sides.append(gap_side(video[:, -1]))
        sides = np.stack(sides)
        return {"rollout_white_mass_ratio": np.stack(masses).mean((0, 1)).tolist(),
                "sample_gap_top_fraction": float((sides > 0).mean()),
                "sample_gap_undecided_fraction": float((sides == 0).mean()),
                "sample_matches_true_gap": float((sides == true_side).mean()),
                "contexts_with_both_gaps_across_samples": float(((sides > 0).any(0) & (sides < 0).any(0)).mean())}

    if a.grid_sampler:
        bundle, _ = T.load_model(a.grid)
        sampler = SM.load(bundle, a.grid_sampler, 8)[0]
        videos = [np.asarray(viz.decode(bundle.decoder, sampled_rollout(sampler, jnp.asarray(ctx), jnp.asarray(out_actions), HOR, jax.random.PRNGKey(k)), HOR))
                  for k in range(a.samples)]
        result["grid_sampled"] = sampled_stats(videos)
    train_mod, _, model = F.load(a.flow_code, a.flow, S.N_ACTIONS)
    wm = F.WM(train_mod, model, 4)
    videos = [wm.imagine(ctx, incoming_ctx, incoming_future, jax.random.PRNGKey(k))[0][:, :HOR] for k in range(a.samples)]
    result["flow"] = sampled_stats(videos)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
