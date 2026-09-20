"""What the LeWM grid latent and its predictor carry about the cube (CPU is enough).

  JAX_PLATFORMS=cpu python -m studies.redcube.diag_jepa --root /dev/shm/redcube/color_frame_wind
"""
import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import probe as P
import train as T
from studies.redcube import env as R


def ridge(xf, yf, xt, yt, lam):
    mu, sd = xf.mean(0), xf.std(0) + 1e-6
    a, b = (xf - mu) / sd, (xt - mu) / sd
    ym = yf.mean(0)
    w = np.linalg.solve(a.T @ a + lam * len(a) * np.eye(a.shape[1]), a.T @ (yf - ym))
    p = b @ w + ym
    return 1.0 - ((yt - p) ** 2).sum(0) / ((yt - yt.mean(0)) ** 2).sum(0)


SIDE = T.GRID


def cube_token(pos):
    cx, cy = pos[..., 0] + R.CUBE // 2, pos[..., 1] + R.CUBE // 2
    return (cy * SIDE // T.H) * SIDE + cx * SIDE // T.W


def token_locate(z_tok, w, b):
    score = (z_tok @ w + b)[..., 0]
    return score.argmax(-1)


def token_dist(a, b):
    ay, ax = np.divmod(a, SIDE)
    by, bx = np.divmod(b, SIDE)
    return np.abs(ay - by) + np.abs(ax - bx)


def cube_from_pixels(frames01, pos):
    f = frames01
    rho = np.clip(f[..., 0] - np.maximum(f[..., 1], f[..., 2]), 0, None)
    rho[..., :2, :] = rho[..., -2:, :] = rho[..., :, :2] = rho[..., :, -2:] = 0
    s = np.pad(rho, [(0, 0)] * (rho.ndim - 2) + [(1, 0), (1, 0)]).cumsum(-2).cumsum(-1)
    k = R.CUBE
    box = s[..., k:, k:] - s[..., :-k, k:] - s[..., k:, :-k] + s[..., :-k, :-k]
    flat = box.reshape(box.shape[0], -1)
    py, px = np.divmod(flat.argmax(-1), box.shape[-1])
    x = np.clip(pos[:, 0], 0, box.shape[-1] - 1)
    y = np.clip(pos[:, 1], 0, box.shape[-2] - 1)
    at_true = flat[np.arange(len(flat)), y * box.shape[-1] + x] / (k * k)
    return float((np.abs(px - x) + np.abs(py - y)).mean()), float(at_true.mean()), float(flat.max(-1).mean() / (k * k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=40)
    ap.add_argument("--envs", type=int, default=40)
    ap.add_argument("--arm", default="grid")
    args = ap.parse_args()
    d = np.load(args.root / "train.npz")
    n, e = args.rows, args.envs
    F = d["frames"][:n + 1, :e]
    pos = d["pos"][:n + 1, :e].astype(np.int64)
    act = d["actions"][:n, :e].astype(np.int32)
    dones = d["dones"][:n, :e]
    bundle, meta = T.load_model(args.root / args.arm / "model_probe.npz")
    m, dec = bundle.model, bundle.decoder
    global SIDE
    SIDE = int(round(meta["cfg"]["tokens"] ** 0.5))
    cell = T.W / SIDE
    print("[cfg]", {k: meta["cfg"][k] for k in ("tokens", "token_dim", "pred_width", "steps", "lr", "sigreg_weight")},
          "final pred", meta.get("final_pred_loss"), "sigreg", meta.get("final_sigreg"))

    z = np.asarray(P.embed_all(m, jnp.asarray(F))).reshape(n + 1, e, SIDE * SIDE, -1)
    tok = cube_token(pos)
    print(f"[latent] shape {z.shape}  mean {z.mean():.3f}  std {z.std():.3f}  per-dim std range "
          f"{z.reshape(-1, z.shape[-1]).std(0).min():.2f}..{z.reshape(-1, z.shape[-1]).std(0).max():.2f}")

    norm = np.linalg.norm(z, axis=-1)
    at = np.take_along_axis(norm, tok[..., None], -1)[..., 0]
    print(f"[encoder] token norm at cube {at.mean():.3f} vs all tokens {norm.mean():.3f} (std over tokens {norm.std():.3f})")

    same = ~dones
    rows = np.arange(n)
    zt, zn = z[:-1][same], z[1:][same]
    flat_t, flat_n = zt.reshape(len(zt), -1), zn.reshape(len(zn), -1)
    corr = ((flat_t - flat_t.mean(0)) * (flat_n - flat_n.mean(0))).mean(0) / (flat_t.std(0) * flat_n.std(0) + 1e-9)
    rnd = np.random.default_rng(0).permutation(len(flat_n))
    corr_r = ((flat_t - flat_t.mean(0)) * (flat_n[rnd] - flat_n.mean(0))).mean(0) / (flat_t.std(0) * flat_n.std(0) + 1e-9)
    print(f"[encoder] consecutive-frame per-dim correlation: mean {corr.mean():.3f}, dims > 0.5: {(corr > 0.5).mean() * 100:.1f}%, "
          f"> 0.2: {(corr > 0.2).mean() * 100:.1f}%  (shuffled pairs {corr_r.mean():.3f})")

    fit_rows = int(0.7 * (n + 1))
    zf, zh = z[:fit_rows].reshape(-1, SIDE * SIDE, z.shape[-1]), z[fit_rows:].reshape(-1, SIDE * SIDE, z.shape[-1])
    tf, th = tok[:fit_rows].reshape(-1), tok[fit_rows:].reshape(-1)
    yf = (np.arange(SIDE * SIDE)[None] == tf[:, None]).astype(np.float32) * 2 - 1
    xf = zf.reshape(-1, z.shape[-1])
    mu, sd = xf.mean(0), xf.std(0) + 1e-6
    xs = (xf - mu) / sd
    w = np.linalg.solve(xs.T @ xs + 1e-3 * len(xs) * np.eye(xs.shape[1]), xs.T @ (yf.reshape(-1) - yf.mean()))
    w_raw, b_raw = (w / sd)[:, None], np.array([yf.mean() - (mu / sd) @ w])
    loc = token_locate(zh, w_raw, b_raw)
    dist = token_dist(loc, th)
    print(f"[encoder] shared per-token linear 'cube here' detector, held-out rows: exact token {(dist == 0).mean() * 100:.1f}%, "
          f"within 1 token {(dist <= 1).mean() * 100:.1f}%  (chance {100 / (SIDE * SIDE):.1f}%)")

    pf, ph = pos[:fit_rows].reshape(-1, 2).astype(np.float32), pos[fit_rows:].reshape(-1, 2).astype(np.float32)
    for lam in (1e-2, 1e-1, 1.0, 10.0):
        r2 = ridge(zf.reshape(len(zf), -1), pf, zh.reshape(len(zh), -1), ph, lam)
        print(f"[encoder] ridge 2592-d -> (x, y), lam {lam:g}: R2 {r2[0]:.3f} {r2[1]:.3f}")
    grid_x = np.tile(np.arange(SIDE)[None, :], (SIDE, 1)).reshape(-1).astype(np.float32)
    grid_y = np.repeat(np.arange(SIDE), SIDE).astype(np.float32)
    soft = jax.nn.softmax(jnp.asarray((zh @ w_raw + b_raw)[..., 0]) * 4.0, axis=-1)
    ex, ey = (np.asarray(soft @ grid_x) + 0.5) * cell, (np.asarray(soft @ grid_y) + 0.5) * cell
    err = np.abs(ex - (ph[:, 0] + R.CUBE // 2)) + np.abs(ey - (ph[:, 1] + R.CUBE // 2))
    print(f"[encoder] position from the detector's soft-argmax: L1 error {err.mean():.1f} px (chance ~35, centre guess ~18)")

    ok = ~(dones[:-2] | dones[1:-1] | dones[2:])
    tt, ee = np.nonzero(ok)
    hist = np.stack([z[tt + k, ee].reshape(len(tt), -1) for k in range(3)], 1)
    acts = np.stack([act[tt + k, ee] for k in range(3)], 1)
    target = z[tt + 3, ee].reshape(len(tt), -1)
    last = z[tt + 2, ee].reshape(len(tt), -1)

    @jax.jit
    def predict(hist, acts):
        return m.predict(jnp.asarray(hist), m.act_embed(jnp.asarray(acts)))[:, -1]

    zhat = np.concatenate([np.asarray(predict(hist[i:i + 512], acts[i:i + 512])) for i in range(0, len(tt), 512)])
    mean_lat = z[:fit_rows].reshape(-1, z.shape[-1] * SIDE * SIDE).mean(0)
    mse = lambda a, b: float(((a - b) ** 2).mean())
    print(f"[predictor] one-step MSE: predictor {mse(zhat, target):.4f}  copy-last {mse(last, target):.4f}  "
          f"mean-latent {mse(mean_lat[None], target):.4f}  target variance {float(target.var(0).mean()):.4f}")
    print(f"[predictor] zhat std {zhat.std():.3f} (encoder latent std {target.std():.3f}); "
          f"per-dim std of zhat / target: {float((zhat.std(0) / (target.std(0) + 1e-9)).mean()):.3f}")
    alt = np.stack([np.concatenate([np.asarray(predict(hist[i:i + 512], np.concatenate([acts[i:i + 512, :2], np.full((min(512, len(tt) - i), 1), a, np.int32)], 1)))
                                    for i in range(0, len(tt), 512)]) for a in range(4)])
    diff = np.mean([np.linalg.norm(alt[a] - alt[b], axis=-1).mean() for a in range(4) for b in range(a + 1, 4)])
    print(f"[predictor] action sensitivity: mean ||zhat(a) - zhat(a')|| {diff:.3f} vs ||z_next - z_last|| "
          f"{np.linalg.norm(target - last, axis=-1).mean():.3f} vs ||zhat - z_next|| {np.linalg.norm(zhat - target, axis=-1).mean():.3f}")
    tok_next, tok_last = tok[tt + 3, ee], tok[tt + 2, ee]
    loc_hat = token_locate(zhat.reshape(len(tt), SIDE * SIDE, -1), w_raw, b_raw)
    loc_true = token_locate(target.reshape(len(tt), SIDE * SIDE, -1), w_raw, b_raw)
    print(f"[predictor] cube-here detector on zhat: exact next token {(loc_hat == tok_next).mean() * 100:.1f}%, within 1 "
          f"{(token_dist(loc_hat, tok_next) <= 1).mean() * 100:.1f}%, equals last token {(loc_hat == tok_last).mean() * 100:.1f}% "
          f"(same detector on the true next latent: {(loc_true == tok_next).mean() * 100:.1f}%)")
    hold = tt >= fit_rows
    if hold.sum() > 50:
        r2h = ridge(zhat[~hold], pos[tt + 3, ee][~hold].astype(np.float32), zhat[hold], pos[tt + 3, ee][hold].astype(np.float32), 1.0)
        r2t = ridge(target[~hold], pos[tt + 3, ee][~hold].astype(np.float32), target[hold], pos[tt + 3, ee][hold].astype(np.float32), 1.0)
        print(f"[predictor] ridge -> next (x, y): from zhat R2 {r2h[0]:.3f} {r2h[1]:.3f}; from the true next latent {r2t[0]:.3f} {r2t[1]:.3f}")

    if dec is not None:
        sel = np.arange(0, len(tt), max(1, len(tt) // 400))
        rec = np.asarray(dec(jnp.asarray(target[sel])))
        pred_px = np.asarray(dec(jnp.asarray(zhat[sel])))
        p_next = pos[tt + 3, ee][sel]
        for name, fr in (("decode(true next latent)", rec), ("decode(predicted latent)", pred_px)):
            err, at_true, peak = cube_from_pixels(fr, p_next)
            print(f"[decoder] {name}: cube position error {err:.1f} px, red mass at true {at_true:.3f}, peak mass {peak:.3f}")
        noise = np.random.default_rng(0).normal(size=zhat[sel].shape).astype(np.float32)
        renoised = np.asarray(dec(jnp.asarray(zhat[sel] + noise * (target.std(0)[None] * np.sqrt(np.clip(1 - (zhat.std(0) / (target.std(0) + 1e-9)) ** 2, 0, 1))[None]))))
        err, at_true, peak = cube_from_pixels(renoised, p_next)
        print(f"[decoder] decode(predicted latent + gaussian fill to the encoder's per-dim std): position error {err:.1f} px, "
              f"mass at true {at_true:.3f}, peak {peak:.3f}")


if __name__ == "__main__":
    main()
