"""One column per environment: the three models' imagined frames under the real frames, a t-SNE of each model's
real-frame latents coloured by cube x, and the benchmark's position error over the imagined steps.

  python -m studies.redcube.infographic --root /dev/shm/redcube --runs runs/redcube \
      --flow-code <recipe dir> --configs off_still off_wind --out runs/redcube/infographic_black.png
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw

import eval as E
import train as T
from studies.compare import jepa_reconstruct
from studies.redcube import env as R
from studies.redcube import flow as F
from studies.redcube.bench import CTX, encode_flow, encode_jepa, flow_rollout, font, jepa_rollout
from studies.redcube.latent_viz import tsne

TITLES = {"off_still": "Чёрный фон, без ветра", "off_wind": "Чёрный фон, с ветром",
          "gray_frame_still": "Серый шум на кадр, без ветра", "gray_frame_wind": "Серый шум на кадр, с ветром",
          "gray_episode_still": "Серый шум на эпизод, без ветра", "gray_episode_wind": "Серый шум на эпизод, с ветром",
          "color_frame_still": "Цветной шум на кадр, без ветра", "color_frame_wind": "Цветной шум на кадр, с ветром",
          "color_episode_still": "Цветной шум на эпизод, без ветра", "color_episode_wind": "Цветной шум на эпизод, с ветром"}
MODELS = (("lewm6", "LeWM, class token", "#e07b39"), ("tile_factored", "JEPA, grid latent", "#7b4ea3"),
          ("flow", "flow matching", "#2a9d8f"))
STRIP_STEPS = (CTX - 2, CTX - 1, CTX, CTX + 1, CTX + 3, CTX + 7, CTX + 15)
GREY, INK, PAPER = "#9a9a9a", "#1c1c1c", (250, 250, 248)


def strip(gt, videos, scale=3, gap=6, left=225, header=36):
    """Rows: real environment, then one per model; columns: the last two context frames and imagined steps
    1, 2, 4, 8, 16; red outline on imagined frames."""
    tile = R.W * scale
    width = left + len(STRIP_STEPS) * (tile + gap) + gap
    height = header + (1 + len(videos)) * (tile + gap) + gap
    canvas = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    big, small = font(21), font(16)
    for j, t in enumerate(STRIP_STEPS):
        x = left + gap + j * (tile + gap)
        draw.text((x, 8), f"context {t + 1}" if t < CTX else f"шаг {t - CTX + 1}", fill=(70, 70, 70), font=small)
    for r, (label, video) in enumerate([("реальная среда", gt)] + videos):
        y = header + gap + r * (tile + gap)
        draw.text((10, y + tile // 2 - 12), label, fill=(30, 30, 30), font=big)
        for j, t in enumerate(STRIP_STEPS):
            x = left + gap + j * (tile + gap)
            canvas.paste(Image.fromarray(video[t]).resize((tile, tile), Image.NEAREST), (x, y))
            if r > 0 and t >= CTX:
                draw.rectangle([x - 2, y - 2, x + tile + 1, y + tile + 1], outline=(196, 40, 40), width=2)
    return np.asarray(canvas)


def load_config(root, runs, flow_code, cfg, n_windows, seed):
    data = E.load_data(root / cfg / "eval.npz")
    bench = json.loads((runs / cfg / "bench.json").read_text())
    arms = json.loads((runs / cfg / "allarms" / "bench.json").read_text())
    oracle_path = runs / cfg / "oracle.json"
    oracle = json.loads(oracle_path.read_text()) if oracle_path.exists() else None
    sel = E.sample_indices(data, E.N_WIN)[:512]
    scene = sel[bench["scenes"][0]:bench["scenes"][0] + 1]
    frames, actions = E.window_batch(data, scene)
    many, _ = E.window_batch(data, sel[:n_windows])
    times = sel[:n_windows, :1] + np.arange(many.shape[1])[None]
    pos = data["pos"][times, sel[:n_windows, 1:]].astype(np.float32).reshape(-1, 2)
    key = jax.random.PRNGKey(seed)
    videos, latents, curves = [], {}, {}
    for arm, name, _ in MODELS:
        if arm == "flow":
            flow_train, _, flow_model = F.load(flow_code, root / cfg / "flow" / "model.npz", R.N_ACTIONS)
            context = np.asarray(flow_train.decode(flow_model, flow_train.encode(flow_model, jnp.asarray(frames[:, :CTX]))))
            video, _ = flow_rollout(flow_train, flow_model, frames[:, :CTX], actions[:, :CTX], actions[:, CTX:], key)
            latents[arm] = encode_flow(flow_train, flow_model, many).reshape(len(pos), -1)
            curves[arm] = bench["pixel_probe"]["flow"]["predicted"]
        else:
            bundle = T.load_model(root / cfg / arm / "model_probe.npz")[0]
            context = np.asarray(jepa_reconstruct(bundle.model, bundle.decoder, jnp.asarray(frames[:, :CTX])))
            video, _ = jepa_rollout(bundle, frames[:, :CTX], actions[:, :CTX], actions[:, CTX:])
            latents[arm] = encode_jepa(bundle, many).reshape(len(pos), -1)
            curves[arm] = (bench if arm in bench["probe_error"] else arms)["probe_error"][arm]["predicted"]
        videos.append((name, np.concatenate([context, video], 1)[0]))
    return {"cfg": cfg, "strip": strip(frames[0], videos), "latents": latents, "pos": pos, "curves": curves,
            "persistence": bench["cube"]["persistence"]["position_error"],
            "oracle": oracle["actions_last_wind"] if oracle and cfg.endswith("wind") else None}


def prototypes(z):
    mu = z.mean(0)
    _, s, vt = np.linalg.svd(z - mu, full_matrices=False)
    std = s / np.sqrt(len(z))
    big = int((std > 1e-2 * std[0]).sum())
    white = (z - mu) @ vt[:big].T / std[:big]
    return len(np.unique(np.round(white * 2), axis=0)), big


def draw_column(fig, cell, panel, n_tsne, seed, first):
    inner = cell.subgridspec(4, 3, height_ratios=[0.3, 4.1, 3.0, 3.2], hspace=0.34, wspace=0.14)
    band = fig.add_subplot(inner[0, :])
    band.axis("off")
    band.text(0, 0.4, TITLES[panel["cfg"]], fontsize=19, fontweight="bold", va="center", color=INK)

    ax = fig.add_subplot(inner[1, :])
    ax.imshow(panel["strip"])
    ax.axis("off")
    ax.set_title("Прогноз на 16 шагов от одних и тех же 8 кадров context", fontsize=11, loc="left", color=INK)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(panel["pos"]), min(n_tsne, len(panel["pos"])), replace=False)
    for j, (arm, name, colour) in enumerate(MODELS):
        ax = fig.add_subplot(inner[2, j])
        z = panel["latents"][arm]
        emb = tsne(z[idx], seed=seed)
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=panel["pos"][idx, 0], cmap="viridis", s=4, lw=0, vmin=0, vmax=R.W - 1)
        n_proto, big = prototypes(z)
        ax.set_xticks([]), ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color("#d0d0d0")
        ax.set_title(name, fontsize=11.5, fontweight="bold", color=colour, loc="left")
        ax.text(0, -0.03, f"{z.shape[1]}-d latent, {n_proto} различных значений\nв {big} направлениях с дисперсией",
                transform=ax.transAxes, fontsize=8, color="#555", va="top")
        if j == 0:
            ax.set_ylabel("t-SNE latent настоящих кадров", fontsize=10)
        if j == 2:
            cb = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.03)
            cb.set_label("x кубика, px", fontsize=9)

    ax = fig.add_subplot(inner[3, :])
    steps = np.arange(1, 17)
    ymax = 30
    ax.plot(steps, panel["persistence"], color=GREY, lw=1.6, ls=":")
    if panel["oracle"] is not None:
        ax.plot(steps, panel["oracle"], color=INK, lw=1, ls="-.")
    for k, (arm, name, colour) in enumerate(MODELS):
        curve = np.asarray(panel["curves"][arm])
        ax.plot(steps, curve, color=colour, lw=2.6)
        if curve.min() > ymax:
            ax.text(15.8, ymax * (0.95 - 0.07 * k), f"{name}: {curve.mean():.0f} px, выше графика", color=colour,
                    fontsize=9.5, va="top", ha="right", fontweight="bold")
    ax.set_ylim(0, ymax)
    ax.set_xlim(1, 16)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_xlabel("шаг прогноза", fontsize=10)
    ax.set_ylabel("ошибка позиции кубика, px", fontsize=10)
    ax.set_title("Ошибка позиции по шагам: probe по latent у JEPA и LeWM, pixel CNN по кадру у flow matching, 512 окон",
                 fontsize=10.5, loc="left", color=INK, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if first:
        handles = [Line2D([], [], color=c, lw=2.6) for _, _, c in MODELS]
        names = [n for _, n, _ in MODELS]
        handles.append(Line2D([], [], color=GREY, lw=1.6, ls=":"))
        names.append("копия последнего кадра")
        if panel["oracle"] is not None:
            handles.append(Line2D([], [], color=INK, lw=1, ls="-."))
            names.append("известные действия + последний ветер")
        ax.legend(handles, names, loc="upper left", frameon=False, fontsize=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs/redcube"))
    parser.add_argument("--flow-code", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--tsne", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    panels = [load_config(args.root, args.runs, args.flow_code, cfg, args.windows, args.seed) for cfg in args.configs]
    fig = plt.figure(figsize=(8.6 * len(panels), 12.4), facecolor="#faf9f6")
    outer = fig.add_gridspec(1, len(panels), wspace=0.16, left=0.045, right=0.985, top=0.97, bottom=0.05)
    for i, panel in enumerate(panels):
        draw_column(fig, outer[i], panel, args.tsne, args.seed, first=True)
        print(f"[done] {panel['cfg']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, facecolor=fig.get_facecolor())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
