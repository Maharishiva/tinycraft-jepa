"""One real-gate episode per trained policy from the same reset and the same per-step keys, saved as frames for the
blog gifs.  Every policy sees the same wall spawns and gap draws; only its actions differ.

    python -m studies.stoch.episodes --config gate --out runs/stoch/gate/episodes.npz \
        --policy "PPO inside the grid JEPA (original predictor)" runs/stoch/gate/ppo_jepa_tf_m0_det_p1/model_best.pkl latent ~/personal/stoch/gate/tile_factored/model_probe.npz \
        ...
Each --policy takes: label, checkpoint, observation mode (pixels | latent | decoded | flow), world-model npz (or - for pixels).
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import train as T
from studies.redcube import flow as F
from studies.stoch import env as S
from studies.stoch.rl import A_N, PHID, load_ac


def observer(mode, wm, flow_code=None, nfe=4):
    if mode == "pixels":
        return lambda f: f
    if mode == "flow":
        train_mod, _, model = F.load(flow_code, wm, A_N)
        wm = F.WM(train_mod, model, nfe)
        return lambda f: wm.encode(f[:, None])[:, 0]
    bundle, _ = T.load_model(wm)
    enc = lambda f: bundle.model.encode(f[:, None])[:, 0]
    if mode == "latent":
        return enc
    return lambda f: jnp.clip(bundle.decoder(enc(f)) * 255.0 + 0.5, 0, 255).astype(jnp.uint8).reshape(f.shape)


def episode(spec, ac, observe, seed, max_steps=S.MAX_STEPS):
    reset, step = S.make(spec)
    reset, step = jax.jit(reset), jax.jit(step)
    act = jax.jit(lambda o, pa, g, k: (lambda logits, _, g2: (jax.random.categorical(k, logits).astype(jnp.int32), g2))(*ac.step(o, pa, g)))
    key = jax.random.PRNGKey(seed)
    k_reset, key = jax.random.split(key)
    obs, state = reset(k_reset)
    frames, seen, walls, gaps, pas = [np.asarray(obs)], [np.asarray(observe(obs[None])[0]) if observe is not None else np.asarray(obs)], [int(state.wall_x)], [int(state.gap)], []
    g, pa = jnp.zeros((1, PHID)), jnp.zeros((1,), jnp.int32)
    done, t = False, 0
    while not done and t < max_steps:
        key, ka, ks = jax.random.split(key, 3)
        o = observe(obs[None]) if observe is not None else obs[None]
        a, g = act(o, pa, g, ka)
        obs, state, rew, d = step(state, a[0], ks)
        pa, done, t = a, bool(d), t + 1
        frames.append(np.asarray(obs)); walls.append(int(state.wall_x)); gaps.append(int(state.gap)); pas.append(int(a[0]))
        seen.append(np.asarray(observe(obs[None])[0]) if observe is not None else np.asarray(obs))
    return np.stack(frames), np.stack(seen), np.asarray(walls), np.asarray(gaps), np.asarray(pas), bool(done) and t < max_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="gate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--policy", nargs=4, action="append", metavar=("LABEL", "CKPT", "OBS", "WM"), required=True)
    p.add_argument("--flow-code", type=Path, default=Path("studies/stoch/flow_recipe"))
    a = p.parse_args()
    spec = S.CONFIGS[a.config]
    out = {}
    for i, (label, ckpt, mode, wm) in enumerate(a.policy):
        ac, meta = load_ac(ckpt)
        observe = None if mode == "pixels" else observer(mode, wm, a.flow_code)
        frames, seen, walls, gaps, acts, died = episode(spec, ac, observe, a.seed)
        print(f"[episode] {label}: {len(frames) - 1} steps, died={died}, walls passed={int(((walls[:-1] != S.NO_WALL) & (walls[1:] == S.NO_WALL)).sum())}", flush=True)
        out[f"frames_{i}"], out[f"seen_{i}"], out[f"wall_{i}"], out[f"gap_{i}"], out[f"act_{i}"] = frames, seen, walls, gaps, acts
        out[f"label_{i}"], out[f"died_{i}"] = np.asarray(label), np.asarray(died)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, n=np.asarray(len(a.policy)), **out)
    print(f"[episode] saved {a.out}")


if __name__ == "__main__":
    main()
