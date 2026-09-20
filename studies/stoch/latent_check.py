"""Where the mean lands: the gate's pre-reveal contexts in latent space and in pixels.

For the 64 pre-reveal contexts of wallmass.py, re-simulate from the reconstructed true state the recorded future
(gap as recorded), the counterfactual future (gap on the other side) and a no-wall future, encode all three, and
measure (1) the wall's amplitude in the latent, ||z_true - z_nowall||^2, against the predictor's teacher-forced
one-step residual, (2) where each rollout lands, by squared distance to the recorded future, the counterfactual,
their midpoint and the no-wall future, and by its coordinate on the wall axis (1 = wall, 0 = no wall) and on the
gap axis (+1 = recorded side, -1 = other side, 0 = midpoint), for the deterministic predictor, the residual head as
the mean of 8 samples and as one sample, and the flow model as the mean of 8 samples and as one sample, (3) the
decoded pictures: soft intensity over the wall pixels shared by both futures and in each gap band.

    XLA_PYTHON_CLIENT_MEM_FRACTION=.2 python -m studies.stoch.latent_check --data .../eval.npz \
        --grid .../tile_factored/model_probe.npz --grid-sampler .../tile_factored/sampler.npz \
        --flow .../flow_model.npz --flow-code studies/stoch/flow_recipe --out runs/stoch/gate/latent_check.json
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
from studies.stoch.wallmass import contexts, sampled_rollout, white

CTX, HOR = 8, 8


def states_at(d, picked):
    """The true env state at frame t of env e for every picked (t, e)."""
    episode = d["episode"]
    rows = []
    for t, e in picked:
        start = int(np.argmax(episode[:, e] == episode[t, e]))
        rows.append((d["pos"][t, e], t - start, episode[t, e], d["wall_x"][t, e], d["gap"][t, e]))
    n = len(rows)
    return S.State(jnp.asarray(np.stack([r[0] for r in rows]), jnp.int32), jnp.zeros((n, 2), jnp.int32),
                   jax.random.split(jax.random.PRNGKey(0), n), jnp.asarray([r[1] for r in rows], jnp.int32),
                   jnp.asarray([r[2] for r in rows], jnp.int32), jnp.asarray([r[3] for r in rows], jnp.int32),
                   jnp.asarray([r[4] for r in rows], jnp.int32))


def simulate(spec, state, actions, key, with_alive=False):
    """HOR frames from `state` under `actions` (N, HOR); optionally also whether every step stayed alive."""
    vstep = jax.jit(jax.vmap(S.make(spec)[1]))
    frames, alive = [], np.ones(actions.shape[0], bool)
    for j in range(HOR):
        key, k = jax.random.split(key)
        obs, state, _, done = vstep(state, actions[:, j], jax.random.split(k, actions.shape[0]))
        frames.append(np.asarray(obs))
        alive &= ~np.asarray(done, bool)
    return (np.stack(frames, 1), alive) if with_alive else np.stack(frames, 1)


def flow_latents(wm, ctx_f, inc_ctx, inc_fut, key, mean_of):
    """The flow model's latent rollout under given actions, one sample per step or the mean of `mean_of` samples."""
    B = ctx_f.shape[0]

    @jax.jit
    def roll(wstate, ctx_f, inc_ctx, inc_fut, key):
        ctx = wm.encode(ctx_f)
        seq = jnp.zeros((B, wm.win) + ctx.shape[-3:], ctx.dtype).at[:, :CTX].set(ctx)
        aseq = jnp.zeros((B, wm.win), jnp.int32).at[:, :CTX].set(inc_ctx)
        zs = []
        for j in range(HOR):
            key, kf = jax.random.split(key)
            n = CTX + j
            aseq = aseq.at[:, n].set(inc_fut[:, j])
            taus = wm.taus(B, n, CTX)
            if mean_of:
                z = jnp.mean(jnp.stack([wm.flow(wstate, seq, taus, aseq, n, k) for k in jax.random.split(kf, mean_of)]), 0)
            else:
                z = wm.flow(wstate, seq, taus, aseq, n, kf)
            seq = seq.at[:, n].set(z)
            zs.append(z)
        return jnp.stack(zs, 1)

    return roll(wm.state, ctx_f, inc_ctx, inc_fut, key)


def sq(a, b):
    return np.square(np.asarray(a, np.float64) - np.asarray(b, np.float64)).sum(-1)


def coord(x, origin, axis):
    """Coordinate of x along `axis` from `origin` (1 at origin + axis), nan where the axis vanishes."""
    x, origin, axis = (np.asarray(v, np.float64) for v in (x, origin, axis))
    norm = np.square(axis).sum(-1)
    return np.where(norm > 1e-9, ((x - origin) * axis).sum(-1) / np.maximum(norm, 1e-9), np.nan)


def landing(rollouts, z_true, z_cf, z_nw):
    """Per-step means over contexts of the distances and coordinates of each rollout (N, HOR, D)."""
    mid = (z_true + z_cf) / 2
    out = {"wall_amplitude_sq": sq(z_true, z_nw).mean(0).tolist(), "gap_amplitude_sq": sq(z_true, z_cf).mean(0).tolist(),
           "latent_dim": int(z_true.shape[-1])}
    for name, z in rollouts.items():
        out[name] = {"to_recorded_sq": sq(z, z_true).mean(0).tolist(), "to_counterfactual_sq": sq(z, z_cf).mean(0).tolist(),
                     "to_midpoint_sq": sq(z, mid).mean(0).tolist(), "to_nowall_sq": sq(z, z_nw).mean(0).tolist(),
                     "wall_coordinate": np.nanmean(coord(z, z_nw, z_true - z_nw), 0).tolist(),
                     "gap_coordinate": np.nanmean(coord(z, mid, (z_true - z_cf) / 2), 0).tolist(),
                     "gap_coordinate_abs": np.nanmean(np.abs(coord(z, mid, (z_true - z_cf) / 2)), 0).tolist()}
    return out


def intensity(frames):
    return np.asarray(frames).astype(np.float32).min(-1) / 255.0


def masked_mean(values, mask):
    """Per-step mean over contexts of the mean intensity inside `mask`, nan where the mask is empty."""
    m = mask.astype(np.float32)
    per = np.where(m.sum((-2, -1)) > 0, (values * m).sum((-2, -1)) / np.maximum(m.sum((-2, -1)), 1), np.nan)
    return np.nanmean(per, 0).tolist()


def pictures(decoded, fut_true, fut_cf):
    """Soft intensity of each decoded rollout on the shared wall pixels and in the two gap bands."""
    wt, wc = white(fut_true) > 0.5, white(fut_cf) > 0.5
    shared, gap_true, gap_cf = wt & wc, wc & ~wt, wt & ~wc
    out = {}
    for name, video in decoded.items():
        v = intensity(video)
        out[name] = {"shared_wall": masked_mean(v, shared), "recorded_gap_band": masked_mean(v, gap_true),
                     "other_gap_band": masked_mean(v, gap_cf)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--grid-sampler", type=Path)
    p.add_argument("--flow", type=Path)
    p.add_argument("--flow-code", type=Path)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--keep", type=int, default=16, help="contexts whose decoded frames are saved for the figures")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    spec = S.CONFIGS["gate"]
    d = np.load(a.data)
    frames, actions = d["frames"], d["actions"].astype(np.int32)
    picked = contexts(d, a.n, 0)
    ctx = np.stack([frames[t - CTX + 1:t + 1, e] for t, e in picked])
    recorded = np.stack([frames[t + 1:t + 1 + HOR, e] for t, e in picked])
    taken = np.stack([actions[t - CTX + 1:t + 1 + HOR, e] for t, e in picked])
    incoming_ctx = np.stack([actions[t - CTX:t, e] for t, e in picked])
    future_actions = jnp.asarray(taken[:, CTX - 1:CTX - 1 + HOR])
    state = states_at(d, picked)
    key = jax.random.PRNGKey(1)
    fut_true = simulate(spec, state, future_actions, key)
    fut_cf, cf_alive = simulate(spec, state._replace(gap=1 - state.gap), future_actions, key, with_alive=True)
    fut_nw = simulate(spec, state._replace(wall_x=jnp.full_like(state.wall_x, S.NO_WALL)), future_actions, key)
    exact = float((fut_true == recorded).all((1, 2, 3, 4)).mean())
    wall_x = np.asarray(state.wall_x)
    result = {"n": int(len(picked)), "reconstruction_exact_fraction": exact, "wall_x_at_context": wall_x.tolist(),
              "counterfactual_survives": int(cf_alive.sum()), "unrevealed_at_step_1": int((wall_x == 45).sum())}

    bundle, _ = T.load_model(a.grid)
    model = bundle.model
    enc = lambda f: np.asarray(viz.encode(model, jnp.asarray(f)))
    z_true, z_cf, z_nw = enc(fut_true), enc(fut_cf), enc(fut_nw)
    rollouts = {"deterministic": np.asarray(T._rollout_latents(model, jnp.asarray(ctx), jnp.asarray(taken), horizon=HOR)[1])}
    shifted_ctx = np.stack([np.stack([frames[t - CTX + 1 + j:t + 1 + j, e] for j in range(HOR)]) for t, e in picked])
    shifted_act = np.stack([np.stack([actions[t - CTX + 1 + j:t + 2 + j, e] for j in range(HOR)]) for t, e in picked])
    one_step = np.asarray(T._rollout_latents(model, jnp.asarray(shifted_ctx.reshape((-1, CTX) + ctx.shape[2:])),
                                             jnp.asarray(shifted_act.reshape(-1, CTX + 1)), horizon=1)[1])[:, 0]
    one_step = one_step.reshape(len(picked), HOR, -1)
    wall_dir = z_true - z_nw
    unit = wall_dir / np.maximum(np.sqrt(np.square(wall_dir).sum(-1, keepdims=True)), 1e-9)
    along = ((one_step - z_true) * unit).sum(-1)
    result["grid"] = {"one_step_residual_sq": sq(one_step, z_true).mean(0).tolist(),
                      "one_step_residual_along_wall_sq": np.square(along).mean(0).tolist(),
                      "one_step_residual_orthogonal_sq": (sq(one_step, z_true) - np.square(along)).mean(0).tolist(),
                      "one_step_wall_coordinate": np.nanmean(coord(one_step, z_nw, wall_dir), 0).tolist(),
                      "one_step_wall_coordinate_unrevealed": np.nanmean(coord(one_step, z_nw, wall_dir)[wall_x == 45], 0).tolist(),
                      "wall_amplitude_sq_unrevealed": sq(z_true, z_nw)[wall_x == 45].mean(0).tolist(),
                      "one_step_residual_sq_unrevealed": sq(one_step, z_true)[wall_x == 45].mean(0).tolist()}
    if a.grid_sampler:
        sampler = SM.load(bundle, a.grid_sampler, 8)[0]
        rollouts["sample"] = np.asarray(sampled_rollout(sampler, jnp.asarray(ctx), jnp.asarray(taken), HOR, jax.random.PRNGKey(2)))
        sampler.mean_of = 8
        rollouts["mean_of_8"] = np.asarray(sampled_rollout(sampler, jnp.asarray(ctx), jnp.asarray(taken), HOR, jax.random.PRNGKey(3)))
        sampler.mean_of = 0
    result["grid"].update(landing(rollouts, z_true, z_cf, z_nw))
    if cf_alive.any():
        result["grid"]["counterfactual_survives"] = landing({k: v[cf_alive] for k, v in rollouts.items()},
                                                            z_true[cf_alive], z_cf[cf_alive], z_nw[cf_alive])
    dec = lambda z: np.asarray(viz.decode(bundle.decoder, jnp.asarray(z), HOR))
    decoded = {name: dec(z) for name, z in rollouts.items()}
    decoded.update({"midpoint_of_futures": dec((z_true + z_cf) / 2), "recorded_future_encoded": dec(z_true),
                    "nowall_future_encoded": dec(z_nw)})
    result["grid"]["pictures"] = pictures(decoded, fut_true, fut_cf)
    saved = {f"grid_{k}": v[:a.keep] for k, v in decoded.items()}

    if a.flow:
        train_mod, _, fmodel = F.load(a.flow_code, a.flow, S.N_ACTIONS)
        wm = F.WM(train_mod, fmodel, 4)
        fenc = lambda f: np.asarray(wm.encode(jnp.asarray(f))).reshape(len(picked), HOR, -1)
        fz_true, fz_cf, fz_nw = fenc(fut_true), fenc(fut_cf), fenc(fut_nw)
        inc_ctx, inc_fut = jnp.asarray(incoming_ctx), future_actions
        flows = {"sample": flow_latents(wm, jnp.asarray(ctx), inc_ctx, inc_fut, jax.random.PRNGKey(4), 0),
                 "mean_of_8": flow_latents(wm, jnp.asarray(ctx), inc_ctx, inc_fut, jax.random.PRNGKey(5), 8)}
        fl = {k: np.asarray(v).reshape(len(picked), HOR, -1) for k, v in flows.items()}
        result["flow"] = landing(fl, fz_true, fz_cf, fz_nw)
        if cf_alive.any():
            result["flow"]["counterfactual_survives"] = landing({k: v[cf_alive] for k, v in fl.items()},
                                                                fz_true[cf_alive], fz_cf[cf_alive], fz_nw[cf_alive])
        fdecoded = {k: np.asarray(wm.decode(v)) for k, v in flows.items()}
        fdecoded["midpoint_of_futures"] = np.asarray(wm.decode(jnp.asarray((fz_true + fz_cf) / 2).reshape(flows["sample"].shape)))
        result["flow"]["pictures"] = pictures(fdecoded, fut_true, fut_cf)
        saved.update({f"flow_{k}": v[:a.keep] for k, v in fdecoded.items()})

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    np.savez_compressed(a.out.with_suffix(".npz"), context=ctx[:a.keep], recorded=fut_true[:a.keep], counterfactual=fut_cf[:a.keep],
                        nowall=fut_nw[:a.keep], wall_x=np.asarray(state.wall_x)[:a.keep], gap=np.asarray(state.gap)[:a.keep], **saved)
    print(json.dumps({k: v for k, v in result.items() if k not in ("wall_x_at_context",)}, indent=1), flush=True)


if __name__ == "__main__":
    main()
