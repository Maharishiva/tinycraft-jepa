"""Wall subspace of a frozen JEPA latent from wall-difference vectors, for the amplitude intervention.

    python -m studies.stoch.wallspace2 --wm ~/personal/stoch/gate/tile_factored/model_probe.npz \
        --data ~/personal/stoch/gate/train.npz --out runs/stoch/gate/dwallspace_m0.npz

For random data frames that contain a wall, the same frame with the wall pixels erased is encoded too; the difference
d = z(frame) - z(frame without wall) is the wall's latent signal (the paper's wall amplitude, per frame).  The
principal directions of the d vectors that carry 90 % of their energy form the wall subspace; a random subspace of the
same rank is written next to it (drandspace_*.npz) as the sham control.  Both hold `basis` (k x dim).
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import train as T
from studies.stoch import env as S


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--energy", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    z = np.load(a.data)
    frames, wall_x, gap = z["frames"], z["wall_x"], z["gap"]
    rng = np.random.default_rng(a.seed)
    has_wall = np.argwhere(wall_x != S.NO_WALL)
    pick = has_wall[rng.choice(len(has_wall), a.n, replace=False)]
    ei, ti = pick[:, 0], pick[:, 1]
    f, wx, gp = frames[ei, ti], wall_x[ei, ti].astype(np.int32), gap[ei, ti].astype(np.int32)
    mask = np.asarray(jax.vmap(S.wall_mask)(jnp.asarray(wx), jnp.asarray(gp)))
    f_nowall = np.where(mask[..., None], 0, f).astype(np.uint8)
    bundle, _ = T.load_model(a.wm)
    enc = jax.jit(lambda x: bundle.model.encode(x[:, None])[:, 0])
    Z = np.concatenate([np.asarray(enc(jnp.asarray(f[i:i + 512]))) for i in range(0, a.n, 512)]).astype(np.float64)
    Z0 = np.concatenate([np.asarray(enc(jnp.asarray(f_nowall[i:i + 512]))) for i in range(0, a.n, 512)]).astype(np.float64)
    d = Z - Z0
    _, s, Vt = np.linalg.svd(d, full_matrices=False)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    k = int(np.searchsorted(energy, a.energy) + 1)
    basis = Vt[:k]
    rand = np.linalg.qr(rng.standard_normal((Z.shape[1], k)))[0].T
    Zc = Z - Z.mean(0)
    var = float((Zc ** 2).sum(1).mean())
    amp = float((d ** 2).sum(1).mean())
    revealed = wx <= S.REVEAL_X
    info = {"dim": int(Z.shape[1]), "rank": k, "n": a.n, "wall_amplitude_sq": amp,
            "wall_amplitude_sq_revealed": float((d[revealed] ** 2).sum(1).mean()),
            "wall_amplitude_sq_unrevealed": float((d[~revealed] ** 2).sum(1).mean()),
            "amplitude_in_subspace": float(((d @ basis.T) ** 2).sum(1).mean()) / amp,
            "amplitude_in_random": float(((d @ rand.T) ** 2).sum(1).mean()) / amp,
            "variance_fraction_wall": float(((Zc @ basis.T) ** 2).sum(1).mean()) / var,
            "variance_fraction_random": float(((Zc @ rand.T) ** 2).sum(1).mean()) / var,
            "latent_variance": var, "energy_top1": float(energy[0]), "energy_top8": float(energy[min(7, len(energy) - 1)])}
    print(json.dumps(info), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, basis=basis.astype(np.float32), info=json.dumps(info))
    np.savez(a.out.with_name(a.out.name.replace("dwallspace", "drandspace")), basis=rand.astype(np.float32), info=json.dumps(info))
    a.out.with_suffix(".json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
