"""One gif over several configurations from their page scene gifs (bench.compose layout: ground truth / JEPA /
flow matching columns, 4x tiles, 12 px gaps, 40 px header): configurations become columns, the three videos rows.

    python -m studies.redcube.compose_scenes --configs off_still off_wind gray_frame_still gray_frame_wind \
        gray_episode_still gray_episode_wind --out runs/redcube/bw_scenes.gif
    ... --source probe --out runs/redcube/bw_probe_scenes.gif      # with the probe boxes
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from studies.redcube.bench import CTX, HOR, font

SRC_SCALE, SRC_GAP, SRC_HEADER, SIDE = 4, 12, 40, 63
ROWS = ("ground truth", "JEPA", "flow matching")
NAMES = {"off_still": "black, still", "off_wind": "black, wind",
         "gray_frame_still": "gray noise / frame, still", "gray_frame_wind": "gray noise / frame, wind",
         "gray_episode_still": "gray noise / episode, still", "gray_episode_wind": "gray noise / episode, wind",
         "color_frame_still": "colour noise / frame, still", "color_frame_wind": "colour noise / frame, wind",
         "color_episode_still": "colour noise / episode, still", "color_episode_wind": "colour noise / episode, wind"}


def tiles(gif):
    """(T, columns, 63, 63, 3) uint8: the columns of every frame, back at native resolution."""
    im = Image.open(gif)
    out = []
    cell = SIDE * SRC_SCALE
    columns = (im.size[0] - SRC_GAP) // (cell + SRC_GAP)
    for t in range(im.n_frames):
        im.seek(t)
        frame = im.convert("RGB")
        row = []
        for j in range(columns):
            x = SRC_GAP + j * (cell + SRC_GAP)
            row.append(np.asarray(frame.crop((x, SRC_HEADER, x + cell, SRC_HEADER + cell)).resize((SIDE, SIDE), Image.NEAREST)))
        out.append(np.stack(row))
    return np.stack(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/redcube"))
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--source", choices=("scene", "probe", "probe_arms"), default="scene",
                        help="scene: the page scene gifs; probe: the probe_scene gifs, whose model tiles carry the "
                             "probe read-out (yellow box = JEPA latent probe, white box = flow pixel CNN); probe_arms: "
                             "the probe_arms_scene gifs with a column per JEPA arm (grid JEPA, class-token LeWM)")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prefix = {"scene": "scene", "probe": "probe_scene", "probe_arms": "probe_arms_scene"}[args.source]
    videos = {c: tiles(args.runs / c / f"{prefix}{args.scene}.gif") for c in args.configs}
    rows = {"scene": ROWS,
            "probe": ("ground truth", "JEPA\nyellow: latent probe", "flow matching\nwhite: pixel CNN"),
            "probe_arms": ("ground truth", "JEPA, grid latent\nyellow: latent probe", "LeWM, class token\nyellow: latent probe",
                           "flow matching\nwhite: pixel CNN")}[args.source]
    big, small = font(16), font(13)
    cell, gap, left, header, footer = SIDE * args.scale, 8, 118 if args.source == "scene" else 176, 46, 26
    width = left + len(args.configs) * (cell + gap) + gap
    height = header + len(rows) * (cell + gap) + footer
    frames = []
    for t in range(CTX + HOR):
        canvas = Image.new("RGB", (width, height), (250, 250, 248))
        draw = ImageDraw.Draw(canvas)
        imagined = t >= CTX
        for j, c in enumerate(args.configs):
            x = left + gap + j * (cell + gap)
            for line, text in enumerate(NAMES[c].split(", ")):
                draw.text((x, 6 + 18 * line), text, fill=(30, 30, 30), font=small)
            for r in range(len(rows)):
                y = header + r * (cell + gap)
                tile = np.repeat(np.repeat(videos[c][t, r], args.scale, 0), args.scale, 1)
                canvas.paste(Image.fromarray(tile), (x, y))
                if imagined and r > 0:
                    draw.rectangle([x - 2, y - 2, x + cell + 1, y + cell + 1], outline=(196, 40, 40), width=2)
        for r, label in enumerate(rows):
            head, _, note = label.partition("\n")
            y = header + r * (cell + gap) + cell // 2 - (18 if note else 9)
            draw.text((8, y), head, fill=(30, 30, 30), font=big)
            if note:
                draw.text((8, y + 22), note, fill=(90, 90, 90), font=small)
        status = f"context {t + 1}/{CTX}" if not imagined else f"imagined step {t - CTX + 1}/{HOR}"
        draw.text((left + gap, height - footer + 4), status, fill=(90, 90, 90), font=small)
        frames.append(canvas)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=250, loop=0)
    print(f"{args.out} {width}x{height} {len(frames)} frames")


if __name__ == "__main__":
    main()
