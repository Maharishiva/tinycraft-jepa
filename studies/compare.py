"""One scene, three columns: ground truth | LeWM (JEPA) | our flow-matching world model.

Both models get the same 8 real frames and the same 16 actions from the frozen benchmark's eval
windows; only daylight windows with real motion are considered.  In the model columns the context
frames are what each model actually holds: real frame -> encoder -> decoder (no pixel copy), and
the imagined steps are decoded predicted latents.  Writes the side-by-side gif and mp4, a filmstrip
png, per-step error curves for both models, and the imagined reward traces.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval as E  # noqa: E402
import train as T  # noqa: E402

CTX, HOR = E.CTX, E.HOR
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_sota(code_dir, ckpt):
    """Import the flow-matching recipe under its own name (it has train.py / kernels.py too)."""
    sys.path.insert(0, str(code_dir))
    spec = importlib.util.spec_from_file_location("sota_train", code_dir / "train.py")
    sota = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sota)
    sys.path.pop(0)
    return sota, sota.load_model(str(ckpt))


@nnx.jit
def jepa_reconstruct(model, decoder, frames_u8):
    """Real frames -> JEPA encoder -> the off-budget pixel decoder, uint8 [B, T, H, W, 3]."""
    b, t = frames_u8.shape[:2]
    emb = model.encode(frames_u8)
    frames = decoder(emb.reshape(-1, emb.shape[-1])).reshape(b, t, T.H, T.W, 3)
    return jnp.clip(frames * 255.0 + 0.5, 0, 255).astype(jnp.uint8)


def pick_scenes(frames, actions, k, brightness_q=0.8):
    """Daylight windows (bright context) where the future actually moves."""
    bright = frames[:, :CTX].astype(np.float32).mean((1, 2, 3, 4))
    motion = np.abs(np.diff(frames[:, CTX:].astype(np.float32), axis=1)).mean((1, 2, 3, 4))
    day = bright >= np.quantile(bright, brightness_q)
    score = np.where(day, motion, -1.0)
    return np.argsort(-score)[:k]


def upscale(frame, scale):
    return np.asarray(Image.fromarray(frame).resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST))


def compose(gt, jepa, sota, scale=5, gap=14, header=44, footer=34):
    """Frames of the side-by-side video: columns GT | JEPA | flow-matching, imagined steps outlined."""
    labels = ("ground truth", "LeWM (JEPA)", "flow-matching WM (ours)")
    font = ImageFont.truetype(FONT, 20)
    small = ImageFont.truetype(FONT, 16)
    cell = 63 * scale
    width = 3 * cell + 4 * gap
    height = header + cell + footer
    out = []
    for t in range(CTX + HOR):
        canvas = Image.new("RGB", (width, height), (250, 250, 248))
        draw = ImageDraw.Draw(canvas)
        imagined = t >= CTX
        for j, (label, video) in enumerate(zip(labels, (gt, jepa, sota))):
            x = gap + j * (cell + gap)
            tile = Image.fromarray(upscale(video[t], scale))
            canvas.paste(tile, (x, header))
            color = (196, 40, 40) if (imagined and j > 0) else (30, 30, 30)
            draw.text((x, 12), label, fill=color, font=font)
            if imagined and j > 0:
                draw.rectangle([x - 3, header - 3, x + cell + 2, header + cell + 2], outline=(196, 40, 40), width=4)
        status = (f"context {t + 1}/{CTX}  (model columns: real frame through encoder and decoder)"
                  if not imagined else f"imagined step {t - CTX + 1}/{HOR}")
        draw.text((gap, header + cell + 8), status, fill=(90, 90, 90), font=small)
        out.append(np.asarray(canvas))
    return out


def filmstrip(gt, jepa, sota, steps, scale=3, gap=6):
    labels = ("ground truth", "LeWM (JEPA)", "flow-matching WM")
    font = ImageFont.truetype(FONT, 15)
    cell = 63 * scale
    left = 175
    width = left + len(steps) * (cell + gap)
    height = 28 + 3 * (cell + gap)
    canvas = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    for c, t in enumerate(steps):
        text = f"t={t + 1}" if t < CTX else f"+{t - CTX + 1}"
        draw.text((left + c * (cell + gap) + cell // 2 - 12, 6), text, fill=(90, 90, 90), font=font)
    for r, (label, video) in enumerate(zip(labels, (gt, jepa, sota))):
        y = 28 + r * (cell + gap)
        draw.text((6, y + cell // 2 - 8), label, fill=(30, 30, 30), font=font)
        for c, t in enumerate(steps):
            x = left + c * (cell + gap)
            canvas.paste(Image.fromarray(upscale(video[t], scale)), (x, y))
            if t >= CTX and r > 0:
                draw.rectangle([x - 2, y - 2, x + cell + 1, y + cell + 1], outline=(196, 40, 40), width=2)
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jepa", type=Path, required=True)
    parser.add_argument("--sota-code", type=Path, required=True)
    parser.add_argument("--sota-ckpt", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/tinywmcraft_500k/eval.npz"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/compare"))
    parser.add_argument("--windows", type=int, default=2048)
    parser.add_argument("--curves", type=int, default=512, help="windows for the error curves")
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = E.load_data(args.data)
    frames, actions = E.window_batch(data, E.sample_indices(data, args.windows))
    bundle, _ = T.load_model(args.jepa)
    sota, model = load_sota(args.sota_code, args.sota_ckpt)

    def roll_both(fr, ac, seed):
        jepa = T.rollout(bundle, fr[:, :CTX], ac[:, :CTX], ac[:, CTX:], seed)
        vid, rew, done = sota.imagine(model, fr[:, :CTX], ac[:, :CTX], ac[:, CTX:], jax.random.PRNGKey(seed))
        vid = np.asarray(vid)
        if vid.dtype != np.uint8:
            vid = np.clip(vid * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return jepa, vid, np.asarray(rew), np.asarray(done)

    # 1. per-step error curves, daylight subset and everything.
    bright = frames[:, :CTX].astype(np.float32).mean((1, 2, 3, 4))
    curves = {}
    for name, sel in (("all", np.arange(args.curves)),
                      ("daylight", np.argsort(-bright)[:args.curves])):
        if args.curves == 0:
            break
        errs = {"persistence": [], "jepa": [], "flow": []}
        for i in range(0, len(sel), 64):
            idx = sel[i:i + 64]
            fr, ac = frames[idx], actions[idx]
            jepa, vid, _, _ = roll_both(fr, ac, args.seed + i)
            real = fr[:, CTX:].astype(np.float32) / 255.0
            for key, pred in (("persistence", np.repeat(fr[:, CTX - 1:CTX], HOR, 1)), ("jepa", jepa), ("flow", vid)):
                errs[key].append(((pred.astype(np.float32) / 255.0 - real) ** 2).mean((2, 3, 4)))
        curves[name] = {k: np.concatenate(v).mean(0).tolist() for k, v in errs.items()}
        print(f"[curves] {name}: " + ", ".join(f"{k} step16 {v[-1]:.5f}" for k, v in curves[name].items()), flush=True)
    if curves:
        (args.out / "curves.json").write_text(json.dumps(curves, indent=1))

    # 2. the scenes: same window, same actions, both models, plus imagined rewards.
    chosen = pick_scenes(frames, actions, args.scenes)
    fr, ac = frames[chosen], actions[chosen]
    jepa, vid, rew, done = roll_both(fr, ac, args.seed + 999)
    jepa_ctx = np.asarray(jepa_reconstruct(bundle.model, bundle.decoder, jnp.asarray(fr[:, :CTX])))
    flow_ctx = np.asarray(sota.decode(model, sota.encode(model, fr[:, :CTX])))
    real_ctx = fr[:, :CTX].astype(np.float32) / 255.0
    recon_mse = {name: ((ctx.astype(np.float32) / 255.0 - real_ctx) ** 2).mean((1, 2, 3, 4))
                 for name, ctx in (("jepa", jepa_ctx), ("flow", flow_ctx))}
    real_rewards = data["rewards"][np.arange(CTX, CTX + HOR)[None] + E.sample_indices(data, args.windows)[chosen][:, :1] - 1,
                                   E.sample_indices(data, args.windows)[chosen][:, 1:]]
    jepa_rewards = None
    if bundle.heads is not None:
        emb, preds = T._rollout_latents(bundle.model, jnp.asarray(fr[:, :CTX]),
                                        T.outgoing_actions(ac[:, :CTX], ac[:, CTX:]), horizon=HOR)
        # reward of the transition into imagined frame j: latent of the frame before it, incoming action.
        before = jnp.concatenate([emb[:, -1:], preds[:, :-1]], 1)
        r_hat, _, _ = bundle.heads(before.reshape(-1, before.shape[-1]), jnp.asarray(ac[:, CTX:]).reshape(-1))
        jepa_rewards = np.asarray(r_hat.mean(-1)).reshape(len(chosen), HOR)
    import imageio.v2 as imageio
    for s in range(len(chosen)):
        gt = fr[s]
        j = np.concatenate([jepa_ctx[s], jepa[s]])
        f = np.concatenate([flow_ctx[s], vid[s]])
        video = compose(gt, j, f)
        imageio.mimsave(args.out / f"scene{s}.gif", video + video[-1:] * 4, duration=1000 / 4, loop=0)
        png_dir = args.out / f"scene{s}_frames"
        png_dir.mkdir(exist_ok=True)
        for t, im in enumerate(video):
            Image.fromarray(im).save(png_dir / f"{t:03d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "4", "-i", str(png_dir / "%03d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "18", str(args.out / f"scene{s}.mp4")], check=True)
        Image.fromarray(filmstrip(gt, j, f, [CTX - 1, CTX, CTX + 1, CTX + 3, CTX + 7, CTX + 11, CTX + 15])).save(
            args.out / f"scene{s}_strip.png")
        record = {"window": int(chosen[s]), "actions": ac[s, CTX:].tolist(),
                  "real_reward": real_rewards[s].tolist(), "flow_reward": rew[s].tolist(),
                  "flow_done": done[s].tolist(),
                  "context_recon_mse": {k: float(v[s]) for k, v in recon_mse.items()}}
        if jepa_rewards is not None:
            record["jepa_reward"] = jepa_rewards[s].tolist()
        (args.out / f"scene{s}.json").write_text(json.dumps(record, indent=1))
        print(f"[scene {s}] window {chosen[s]} brightness {bright[chosen[s]]:.1f} "
              f"real reward sum {real_rewards[s].sum():.2f} flow {rew[s].sum():.2f} "
              f"context recon mse jepa {recon_mse['jepa'][s]:.5f} flow {recon_mse['flow'][s]:.5f}", flush=True)


if __name__ == "__main__":
    main()
