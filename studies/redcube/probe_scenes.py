"""The benchmark's scenes with the position probe's estimate drawn on every model frame, to eyeball the probe.

Same probes as bench.py, fit on training frames; same scene windows (bench.json "scenes"), fresh rollouts.
JEPA, translucent yellow box = the per-tile latent probe read off the latent that produced the frame (encoded real frame in the
context, predicted latent afterwards; the decoder is only for the picture); flow matching, translucent yellow box = the pixel
CNN read off the decoded frame itself.  Both estimates of both models are printed and saved.

    python -m studies.redcube.probe_scenes --config color_frame_wind --root /dev/shm/redcube/color_frame_wind \
        --flow-code <recipe dir> --out runs/redcube/color_frame_wind
    ... --arms tile_factored lewm6 --prefix probe_arms_scene     # one JEPA column per arm (labels from bench.ARMS)
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

import eval as E
import train as T
from studies.compare import jepa_reconstruct
from studies.redcube import env as R
from studies.redcube import flow as F
from studies.redcube.bench import ARMS, CTX, compose, encode_flow, encode_jepa, filmstrip, fit_probes, flow_rollout, jepa_rollout

YELLOW = np.array([255, 230, 0], np.uint8)
STRIP_STEPS = [CTX - 1, CTX, CTX + 1, CTX + 3, CTX + 7, CTX + 11, CTX + 15]


def boxed(video, estimates, colour=YELLOW, alpha=0.55):
    """Alpha-blend a one-pixel outline outside the estimated cube; each corner is blended once."""
    if not 0 <= alpha <= 1:
        raise ValueError('box alpha must be in [0, 1]')
    out = video.copy()
    for t, (x, y) in enumerate(np.clip(np.rint(estimates), 0, R.W - 1).astype(int)):
        x0, y0, x1, y1 = max(x - 1, 0), max(y - 1, 0), min(x + R.CUBE, R.W - 1), min(y + R.CUBE, R.H - 1)
        mask = np.zeros((R.H,R.W),bool)
        mask[y0,x0:x1+1] = mask[y1,x0:x1+1] = True
        mask[y0:y1+1,x0] = mask[y0:y1+1,x1] = True
        out[t,mask] = np.rint((1-alpha)*out[t,mask].astype(float)+alpha*colour).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=sorted(R.CONFIGS))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--flow-code", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=E.N_WIN)
    parser.add_argument("--metric-windows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", nargs="+", default=["tile_factored"])
    parser.add_argument("--prefix", default="probe_scene")
    parser.add_argument("--box-alpha", type=float, default=0.55)
    parser.add_argument("--bench-results", type=Path, default=None, help="source bench.json when writing to a new directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = E.load_data(args.root / "eval.npz")
    sel = E.sample_indices(data, args.windows)[:args.metric_windows]
    scenes = json.loads((args.bench_results or args.out / "bench.json").read_text())["scenes"]
    frames, actions = E.window_batch(data, sel)
    times = sel[:, :1] + np.arange(frames.shape[1])[None]
    pos = data["pos"][times, sel[:, 1:]].astype(np.float32)
    frames, actions, pos = frames[scenes], actions[scenes], pos[scenes]

    bundles = {arm: T.load_model(args.root / arm / "model_probe.npz")[0] for arm in args.arms}
    flow_train, _, flow_model = F.load(args.flow_code, args.root / "flow" / "model.npz", R.N_ACTIONS)
    latent_probes, pixel_probe = fit_probes(bundles, flow_train, flow_model, args.root)

    def estimates(probe, context_latents, pred):
        d = pred.shape[-1]
        return np.concatenate([probe(context_latents.reshape(-1, d)).reshape(-1, CTX, 2),
                               probe(pred.reshape(-1, d)).reshape(-1, pred.shape[1], 2)], 1)

    columns = {}
    for arm, bundle in bundles.items():
        video, pred = jepa_rollout(bundle, frames[:, :CTX], actions[:, :CTX], actions[:, CTX:])
        context = np.asarray(jepa_reconstruct(bundle.model, bundle.decoder, jnp.asarray(frames[:, :CTX])))
        columns[ARMS[arm]] = {"video": np.concatenate([context, video], 1),
                              "latent": estimates(latent_probes[arm], encode_jepa(bundle, frames)[:, :CTX], pred), "box": "latent"}
    video, pred = flow_rollout(flow_train, flow_model, frames[:, :CTX], actions[:, :CTX], actions[:, CTX:], jax.random.PRNGKey(args.seed))
    context = np.asarray(flow_train.decode(flow_model, flow_train.encode(flow_model, jnp.asarray(frames[:, :CTX]))))
    columns["flow matching"] = {"video": np.concatenate([context, video], 1),
                                "latent": estimates(latent_probes["flow"], encode_flow(flow_train, flow_model, frames)[:, :CTX], pred),
                                "box": "pixel"}
    for c in columns.values():
        c["pixel"] = pixel_probe(c["video"]).reshape(c["video"].shape[:2] + (2,))

    labels = ["ground truth"] + list(columns)
    numbers = []
    for s in range(len(scenes)):
        cols = [boxed(frames[s], pos[s], alpha=args.box_alpha)] + [boxed(c["video"][s], c[c["box"]][s], alpha=args.box_alpha) for c in columns.values()]
        gif = compose(cols, labels)
        imageio.mimsave(args.out / f"{args.prefix}{s}.gif", gif + gif[-1:] * 4, duration=1000 / 4, loop=0)
        Image.fromarray(filmstrip(cols, labels, STRIP_STEPS)).save(args.out / f"{args.prefix}{s}_strip.png")
        numbers.append({"window": int(scenes[s]), "true": pos[s].tolist(),
                        **{f"{name} {kind}": c[kind][s].tolist() for name, c in columns.items() for kind in ("latent", "pixel")}})
        for t in range(frames.shape[1]):
            line = "  ".join(f"{name}: latent ({c['latent'][s, t, 0]:5.1f}, {c['latent'][s, t, 1]:5.1f}) err {np.abs(c['latent'][s, t] - pos[s, t]).sum():4.1f}"
                             f" pixel ({c['pixel'][s, t, 0]:5.1f}, {c['pixel'][s, t, 1]:5.1f}) err {np.abs(c['pixel'][s, t] - pos[s, t]).sum():4.1f}"
                             for name, c in columns.items())
            print(f"scene {s} t={t:2d} {'ctx ' if t < CTX else 'pred'} true ({pos[s, t, 0]:3.0f}, {pos[s, t, 1]:3.0f})  {line}", flush=True)
    (args.out / f"{args.prefix}s.json").write_text(json.dumps(numbers))
    np.savez(args.out / f"{args.prefix}_raw.npz", frames=frames, pos=pos,
             **{f'{name}_{kind}': c[kind] for name,c in columns.items() for kind in ('video','latent','pixel')})
    print(f"[done] -> {args.out}/{args.prefix}*.gif", flush=True)


if __name__ == "__main__":
    main()
