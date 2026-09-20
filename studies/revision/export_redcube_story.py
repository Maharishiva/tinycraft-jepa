"""Fixed-window, frozen-model red-cube visual evidence; no fitting or selection by outcomes.

Prepare uses only recorded metadata and writes selection.json before any model is loaded.
Export uses the pinned inputs and frozen checkpoint hashes. Render is CPU-only.
The original RGB and native latent values are preserved alongside browser assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

HISTORY, HORIZON, BATCH = 8, 16, 64
ROWS = [
    ("real_future", "Recorded future", "#087E83"),
    ("jepa_encoded_real", "JEPA · real-frame reconstruction", "#CD663E"),
    ("jepa_future", "JEPA · predicted future", "#CD663E"),
    ("flow_encoded_real", "Flow · real-frame reconstruction", "#7860A5"),
    ("flow_future", "Flow · one sampled future", "#7860A5"),
]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def ahash(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def first_windows(data, count, gate=False):
    """Lexicographic first qualifying window from each distinct recorded episode.

    Qualification is metadata-only continuity over all 24 frames, plus an already
    present wall for gate. This intentionally illustrates uninterrupted forecasts;
    it is not an unbiased evaluation sample or a policy-success estimate.
    """
    dones = data["dones"].astype(bool)
    total, envs = dones.shape
    span = HISTORY + HORIZON
    cumulative = np.concatenate([np.zeros((1, envs), np.int64), np.cumsum(dones, axis=0)])
    starts = np.arange(1, total - span + 2)
    eligible = cumulative[starts + span - 1] == cumulative[starts]
    if gate:
        if "wall_x" not in data:
            raise ValueError("gate selection requires recorded wall_x")
        eligible &= data["wall_x"][starts + HISTORY - 1] != -99
    rows, cols = np.nonzero(eligible)
    selected, seen = [], set()
    for row, env in zip(rows, cols):
        start = int(starts[row])
        episode = int(data["episode"][start, env])
        episode_id = (int(env), episode)
        if episode_id not in seen:
            selected.append((start, int(env)))
            seen.add(episode_id)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} eligible distinct episodes")
    return np.asarray(selected, np.int64), int(eligible.sum())


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "selection.json").exists() or (args.out / "inputs.npz").exists():
        raise ValueError("refusing to replace pinned selection")
    with np.load(args.eval, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    ids, eligible = first_windows(data, args.scenes, args.config == "gate")
    times = ids[:, :1] + np.arange(HISTORY + HORIZON)
    envs = ids[:, 1:]
    frames = data["frames"][times, envs]
    outgoing = data["actions"][times[:, :-1], envs].astype(np.int32)
    incoming = data["actions"][times - 1, envs].astype(np.int32)
    reset_before = data["dones"][times - 1, envs].astype(bool)
    incoming[reset_before[:, 0], 0] = 0  # Archive's incoming-action BOS convention.
    resets_after = data["dones"][times[:, :-1], envs].astype(bool)
    assert not resets_after.any()
    assert np.array_equal(incoming[:, 1:], outgoing)
    assert frames.shape == (args.scenes, HISTORY + HORIZON, 63, 63, 3)
    payload = {"ids": ids, "frame_times": times, "frames": frames,
               "outgoing_actions": outgoing, "incoming_actions": incoming,
               "future_actions": outgoing[:, HISTORY - 1:],
               "reset_before_frame": reset_before,
               "reset_after_frame": resets_after,
               "valid_mask": np.ones((args.scenes, HORIZON), bool)}
    for name in ("pos", "wind", "episode", "wall_x", "gap", "step", "texture"):
        if name in data:
            payload[name] = data[name][times, envs]
    np.savez_compressed(args.out / "inputs.npz", **payload)
    meta = json.loads(str(data["meta"])) if "meta" in data else {}
    source_files = {"eval": args.eval, "jepa_checkpoint": args.jepa, "flow_checkpoint": args.flow}
    selected = {
        "case_id": args.config, "history": HISTORY, "horizon": HORIZON,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_rule": "First lexicographic (start_time, environment_index) window from each distinct (environment_index, episode_id), with no reset across its 24 frames" + (" and a wall already present in the last context frame." if args.config == "gate" else "."),
        "selection_caveat": "Continuity conditions on future survival. These are illustrations, not an unbiased estimate of model or policy quality. No image, movement amount, or model output is used for selection.",
        "eligible_windows": eligible, "scene_ids": ids.tolist(),
        "data_metadata": meta, "source_files": {k: {"path": str(p), "sha256": sha(p)} for k, p in source_files.items()},
        "input_packet_sha256": sha(args.out / "inputs.npz"),
        "array_sha256": {k: ahash(v) for k, v in payload.items()},
        "flow_sample_seed": args.seed, "flow_nfe": args.nfe,
        "action_convention": "outgoing_actions[s,t] moves frames[s,t] to frames[s,t+1]. future_actions starts at t=7 and has 16 actions. incoming_actions is retained for the original flow API.",
        "ground_truth_randomness": "The archive does not store every transition PRNG key. Its collection seed/metadata, archive hash, exact indices, actions, observations and state labels are retained. Flow sampling keys are stored separately; flow is not expected to reproduce the archive's particular noise realization.",
        "source_sha256": sha(__file__),
    }
    dump(args.out / "selection.json", selected)
    print(json.dumps({"prepared": args.config, "ids": ids.tolist(), "input_sha256": selected["input_packet_sha256"]}), flush=True)


def export(args):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import train as T
    from studies.redcube import flow as F

    start = time.monotonic()
    selection = json.loads((args.out / "selection.json").read_text())
    if (args.out / "frames.npz").exists():
        raise ValueError("refusing to replace a frozen export")
    if sha(args.out / "inputs.npz") != selection["input_packet_sha256"]:
        raise ValueError("pinned input packet changed")
    for key, path in (("jepa_checkpoint", args.jepa), ("flow_checkpoint", args.flow)):
        if sha(path) != selection["source_files"][key]["sha256"]:
            raise ValueError(f"checkpoint changed: {key}")
    with np.load(args.out / "inputs.npz") as a:
        inputs = {k: a[k] for k in a.files}
    n = len(inputs["frames"])
    pad = lambda a: np.pad(a, [(0, BATCH - n)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    frames = pad(inputs["frames"])
    incoming = pad(inputs["incoming_actions"])
    future_actions = pad(inputs["future_actions"])
    outgoing = np.concatenate([pad(inputs["outgoing_actions"]), np.zeros((BATCH, 1), np.int32)], axis=1)
    bundle, jmeta = T.load_model(args.jepa)
    if bundle.decoder is None or bundle.model.history != HISTORY or bundle.model.latent_dim != 512:
        raise ValueError("requires the native 512-dimensional JEPA and its frozen decoder, history8")
    ftrain, _, fmodel = F.load(args.flow_code, args.flow, 4)
    fmeta = ftrain.load_meta(str(args.flow))

    @nnx.jit
    def encode_jepa(model, u8):
        return model.encode(u8)

    @nnx.jit
    def decode_jepa(decoder, z):
        return jnp.clip(decoder(z) * 255.0 + 0.5, 0, 255).astype(jnp.uint8)

    def jdecode(z):
        # All decoder invocations see 512 native latent vectors, including real,
        # predicted and context vectors. Padding changes no represented value.
        shape = z.shape[:-1]
        flat = np.asarray(z).reshape(-1, z.shape[-1])
        result = []
        for offset in range(0, len(flat), BATCH * HISTORY):
            part = flat[offset:offset + BATCH * HISTORY]
            count = len(part)
            part = np.pad(part, ((0, BATCH * HISTORY - count), (0, 0)), mode="edge")
            result.append(np.asarray(decode_jepa(bundle.decoder, jnp.asarray(part)))[:count])
        return np.concatenate(result).reshape(*shape, 63, 63, 3)

    ctx = jnp.asarray(frames[:, :HISTORY])
    jcontext, jfuture = T._rollout_latents(bundle.model, ctx, jnp.asarray(outgoing), horizon=HORIZON)
    direct_context = encode_jepa(bundle.model, ctx)
    context_parity = float(np.max(np.abs(np.asarray(jcontext).astype(np.float32) - np.asarray(direct_context).astype(np.float32))))
    if context_parity != 0:
        raise ValueError(f"canonical context encoder parity failed: {context_parity}")
    # Match the identical B64 x H8 encoding shape for every real future frame.
    jreal, freal = [], []
    for h in range(HORIZON):
        history = jnp.asarray(frames[:, h + 1:h + HISTORY + 1])
        jreal.append(np.asarray(encode_jepa(bundle.model, history))[:, -1])
        freal.append(np.asarray(ftrain.encode(fmodel, history))[:, -1])
    jreal = np.stack(jreal, 1)
    freal = np.stack(freal, 1)
    fcontext = ftrain.encode(fmodel, ctx)
    fg, fs = nnx.split(fmodel.df)
    sample_key = jax.random.PRNGKey(selection["flow_sample_seed"])
    rollout_key, _ = jax.random.split(sample_key)

    @jax.jit
    def flow_predict(state, context, actions_in, actions_future, key):
        latent = ftrain._rollout_latents(nnx.merge(fg, state), context, actions_in,
                                        actions_future.T, key, selection["flow_nfe"], fmodel.win)
        return latent.transpose(1, 0, 2, 3, 4)

    ffuture = flow_predict(fs, fcontext, jnp.asarray(incoming[:, :HISTORY]), jnp.asarray(future_actions), rollout_key)
    payload = {
        "context": inputs["frames"][:, :HISTORY], "real_future": inputs["frames"][:, HISTORY:],
        "jepa_context": jdecode(np.asarray(jcontext)[:n]),
        "jepa_encoded_real": jdecode(jreal[:n]),
        "jepa_future": jdecode(np.asarray(jfuture)[:n]),
        "flow_context": np.asarray(ftrain.decode(fmodel, fcontext))[:n],
        "flow_encoded_real": np.asarray(ftrain.decode(fmodel, jnp.asarray(freal)))[:n],
        "flow_future": np.asarray(ftrain.decode(fmodel, ffuture))[:n],
        "jepa_real_latents": jreal[:n].astype(np.float32), "jepa_predicted_latents": np.asarray(jfuture)[:n].astype(np.float32),
        "flow_real_latents": freal[:n].astype(np.float32), "flow_predicted_latents": np.asarray(ffuture)[:n].astype(np.float32),
        "flow_sample_key": np.asarray(sample_key), "flow_rollout_key": np.asarray(rollout_key),
        **{k: v for k, v in inputs.items() if k != "frames"},
    }
    for k, v in payload.items():
        if v.dtype.kind in "fc" and not np.isfinite(v).all():
            raise ValueError(f"nonfinite array {k}")
    np.savez_compressed(args.out / "frames.npz", **payload)
    manifest = {**selection, "claim_status": "illustration_only", "claim_text": "Recorded-action futures and reconstructions from fixed checkpoints; these images do not by themselves establish missing information or planning performance.",
        "jepa_metadata": jmeta, "flow_metadata": fmeta,
        "decoder_budget_note": "JEPA decoder fitted after world-model training on frozen real-frame latents; its stored configuration and frame/update count are in jepa_metadata.decoder. This diagnostic decoder is not a claim of an optimal inverse. Flow reconstruction uses the checkpoint's jointly trained codec.",
        "encoder_context_batch": BATCH, "encoder_history": HISTORY, "jepa_decoder_frame_batch": BATCH * HISTORY,
        "encoder_context_max_abs_difference": context_parity,
        "latent_units": "Unmodified checkpoint values, saved losslessly in float32 (BF16 values embed exactly). No normalization, amplification, subspace scaling, probe overlay, or manual recoloring.",
        "flow_codec_real_frame_semantics": "Temporal codec sees the actual previous/current RGB pair; each displayed real future is encoded as the last frame of an 8-frame rolling history at batch64. No future frame enters a forecast.",
        "flow_sampling": "One seed fixed before export, no selection among samples; batch64 padded with repeated last selected input; original flow rollout key split and one key per future step.",
        "export_source_sha256": sha(__file__),
        "source_modules": {str(p): sha(p) for p in (Path(T.__file__), Path(F.__file__), Path(ftrain.__file__), Path(args.flow_code) / "eval.py", Path("probe.py"))},
        "packet_sha256": sha(args.out / "frames.npz"), "array_sha256": {k: ahash(v) for k, v in payload.items()},
        "versions": {"jax": jax.__version__, "numpy": np.__version__, "devices": [str(d) for d in jax.devices()]},
        "seconds": round(time.monotonic() - start, 2),
    }
    dump(args.out / "case_manifest.json", manifest)
    print(json.dumps({"exported": args.config, "seconds": manifest["seconds"], "packet_sha256": manifest["packet_sha256"]}), flush=True)


def render(args):
    from PIL import Image, ImageDraw, ImageFont
    with np.load(args.out / "frames.npz") as a:
        data = {k: a[k] for k in a.files}
    manifest = json.loads((args.out / "case_manifest.json").read_text())
    if sha(args.out / "frames.npz") != manifest["packet_sha256"]:
        raise ValueError("frozen frame packet changed")
    assets = args.out / "images"
    assets.mkdir(exist_ok=True)
    fonts = [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/System/Library/Fonts/Supplemental/Arial.ttf")]
    fontpath = next((p for p in fonts if p.exists()), None)
    font = lambda s: ImageFont.truetype(str(fontpath), s) if fontpath else ImageFont.load_default()
    scenes = []
    for s in range(len(data["context"])):
        record = {"scene": s, "id": data["ids"][s].tolist(), "rows": {}, "context": []}
        # Native-resolution PNGs are exact array values; viewer controls scaling.
        for key, _, _ in ROWS:
            record["rows"][key] = []
            for t, frame in enumerate(data[key][s]):
                name = f"scene{s}_{key}_{t + 1:02d}.png"
                Image.fromarray(frame).save(assets / name)
                record["rows"][key].append("images/" + name)
        for t, frame in enumerate(data["context"][s]):
            name = f"scene{s}_context_{t:02d}.png"
            Image.fromarray(frame).save(assets / name)
            record["context"].append("images/" + name)
        record["actions"] = data["future_actions"][s].tolist()
        record["valid_mask"] = data["valid_mask"][s].tolist()
        record["frame_times"] = data["frame_times"][s, HISTORY:].tolist()
        record["ground_truth_position"] = data["pos"][s, HISTORY:].tolist() if "pos" in data else None
        steps = [0, 1, 3, 7, 15]
        cell, left, gap, top = 126, 260, 10, 52
        canvas = Image.new("RGB", (left + len(steps) * (cell + gap) + 14, top + len(ROWS) * (cell + 18) + 62), "#fbfaf7")
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 15), args.config.replace("_", " "), fill="#17324D", font=font(19))
        for c, t in enumerate(steps):
            draw.text((left + c * (cell + gap), 23), f"+{t + 1} step" + ("s" if t else ""), fill="#42565d", font=font(15))
        for r, (key, label, color) in enumerate(ROWS):
            y = top + r * (cell + 18)
            draw.rounded_rectangle((16, y + 8, 22, y + cell - 8), radius=3, fill=color)
            words = label.split(" · ")
            for j, text in enumerate(words):
                draw.text((33, y + 40 + j * 21), text, fill=color if not j else "#42565d", font=font(16 if not j else 14))
            for c, t in enumerate(steps):
                image = Image.fromarray(data[key][s, t]).resize((cell, cell), Image.Resampling.NEAREST)
                canvas.paste(image, (left + c * (cell + gap), y))
        draw.text((18, canvas.height - 47), f"Scene {s + 1}: first eligible distinct episode (start t ≥ 1); recorded actions. Flow is one fixed sample, not the recorded noise.", fill="#617077", font=font(12))
        draw.text((18, canvas.height - 29), f"JEPA model seed {manifest['jepa_metadata']['cfg']['seed']}; flow model seed {manifest['flow_metadata']['cfg']['seed']}, {manifest['flow_nfe']} evaluations per step. Control compute is reported separately.", fill="#617077", font=font(12))
        canvas.save(args.out / f"scene{s}_filmstrip.png")
        record["filmstrip"] = f"scene{s}_filmstrip.png"
        scenes.append(record)
    browser = {"case_id": args.config, "history": HISTORY, "horizon": HORIZON,
               "row_labels": {k: label for k, label, _ in ROWS}, "scenes": scenes,
               "selection_rule": manifest["selection_rule"] + " Candidate starts are t >= 1, retaining the recorded incoming action at t-1.",
               "selection_candidate_start_min": 1, "selection_caveat": manifest["selection_caveat"],
               "action_names": ["left", "up", "right", "down"],
               "manifest": "case_manifest.json", "packet_sha256": manifest["packet_sha256"],
               "flow_visual_nfe": manifest["flow_nfe"], "render_source_sha256": sha(__file__),
               "jepa_model_seed": manifest["jepa_metadata"]["cfg"]["seed"],
               "flow_model_seed": manifest["flow_metadata"]["cfg"]["seed"],
               "method_colors": {"real": "#087E83", "jepa": "#CD663E", "flow": "#7860A5"}}
    dump(args.out / "browser_packet.json", browser)
    print(json.dumps({"rendered": args.config, "scenes": len(scenes), "pngs": len(list(assets.glob("*.png")))}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "export", "render"))
    p.add_argument("--config", required=True)
    p.add_argument("--eval", type=Path)
    p.add_argument("--jepa", type=Path)
    p.add_argument("--flow", type=Path)
    p.add_argument("--flow-code", type=Path, default=Path("studies/stoch/flow_recipe"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenes", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--nfe", type=int, default=8)
    args = p.parse_args()
    if args.mode in ("prepare", "export") and (args.jepa is None or args.flow is None):
        p.error("prepare/export require --jepa and --flow")
    if args.mode == "prepare" and args.eval is None:
        p.error("prepare requires --eval")
    {"prepare": prepare, "export": export, "render": render}[args.mode](args)


if __name__ == "__main__":
    main()
