"""Wall subspace of a frozen JEPA latent, for the amplitude intervention.

    python -m studies.stoch.wallspace --wm ~/personal/stoch/gate/tile_factored/model_probe.npz \
        --data ~/personal/stoch/gate/train.npz --out runs/stoch/gate/wallspace_m0.npz

A ridge probe maps the latent to one-hot wall position (every wall_x value in the data, plus "no wall") and gap state
(undrawn, top, bottom); the orthonormal span of its weight columns is the wall subspace.  A random subspace of the same
rank is written next to it (randspace_*.npz) as the sham control.  Both files hold `basis` (k x dim).  PPO in the real
gate then observes z + (gain - 1) P z (studies.stoch.rl --obs latent --scale-basis ... --gain ...).
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
    p.add_argument("--n", type=int, default=40000)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    z = np.load(a.data)
    frames, wall_x, gap = z["frames"], z["wall_x"], z["gap"]
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(frames.shape[0] * frames.shape[1], a.n, replace=False)
    ei, ti = np.divmod(idx, frames.shape[1])
    f, wx, gp = frames[ei, ti], wall_x[ei, ti].astype(np.int32), gap[ei, ti].astype(np.int32)
    bundle, _ = T.load_model(a.wm)
    enc = jax.jit(lambda x: bundle.model.encode(x[:, None])[:, 0])
    Z = np.concatenate([np.asarray(enc(jnp.asarray(f[i:i + 512]))) for i in range(0, a.n, 512)]).astype(np.float64)
    vals = np.unique(wx)
    wall_cls = np.searchsorted(vals, wx)
    revealed = (wx != S.NO_WALL) & (wx <= S.REVEAL_X)
    gap_cls = np.where(revealed, gp + 1, 0)
    Y = np.concatenate([np.eye(len(vals))[wall_cls], np.eye(3)[gap_cls]], 1)
    ntr = int(0.8 * a.n)
    mu, ymu = Z[:ntr].mean(0), Y[:ntr].mean(0)
    Zc, Yc = Z - mu, Y - ymu
    W = np.linalg.solve(Zc[:ntr].T @ Zc[:ntr] + a.ridge * np.eye(Z.shape[1]), Zc[:ntr].T @ Yc[:ntr])
    pred = Zc[ntr:] @ W + ymu
    nw = len(vals)
    acc_wall = float((pred[:, :nw].argmax(1) == wall_cls[ntr:]).mean())
    acc_gap = float((pred[:, nw:].argmax(1) == gap_cls[ntr:]).mean())
    acc_gap_rev = float((pred[:, nw:].argmax(1) == gap_cls[ntr:])[revealed[ntr:]].mean())
    U, s, _ = np.linalg.svd(W, full_matrices=False)
    k = int((s > 1e-6 * s[0]).sum())
    basis = U[:, :k].T
    rand = np.linalg.qr(rng.standard_normal((Z.shape[1], k)))[0].T
    var = float((Zc ** 2).sum(1).mean())
    frac_wall = float(((Zc @ basis.T) ** 2).sum(1).mean()) / var
    frac_rand = float(((Zc @ rand.T) ** 2).sum(1).mean()) / var
    info = {"dim": int(Z.shape[1]), "rank": k, "n_wall_classes": nw, "acc_wall": acc_wall, "acc_gap": acc_gap,
            "acc_gap_revealed": acc_gap_rev, "chance_wall": float(np.bincount(wall_cls).max() / a.n),
            "chance_gap": float(np.bincount(gap_cls).max() / a.n), "variance_fraction_wall": frac_wall,
            "variance_fraction_random": frac_rand, "latent_variance": var}
    print(json.dumps(info), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, basis=basis.astype(np.float32), info=json.dumps(info))
    np.savez(a.out.with_name(a.out.name.replace("wallspace", "randspace")), basis=rand.astype(np.float32), info=json.dumps(info))
    a.out.with_suffix(".json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
