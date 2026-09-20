"""Animated versions of the paper's frame strips for the blog.

    python -m studies.stoch.gifs imagination --context 0 --out studies/stoch/article/assets/imagination.gif
    python -m studies.stoch.gifs agents --episodes runs/stoch/gate/episodes.npz --out studies/stoch/article/assets/agents.gif

`imagination`: the eight context frames, then the eight imagined steps of every model from the same context (the
rows of Figure 1A), side by side and animated.  `agents`: one real-gate episode per trained policy from the same
reset, animated frame by frame; a panel freezes with a red frame when its agent dies.
"""
import argparse
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUNS = Path(__file__).resolve().parents[2] / "runs" / "stoch"
SCALE = 3
BG = (250, 250, 248)
INK = (30, 30, 30)
RED = (200, 40, 30)


def font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default(size=size)


def up(frame):
    return Image.fromarray(np.asarray(frame, np.uint8)).resize((frame.shape[1] * SCALE, frame.shape[0] * SCALE), Image.NEAREST)


def sheet(panels, titles, header, footer, cols, dead=None):
    """One gif frame: `panels` are 63x63x3 arrays laid out in `cols` columns with a title above each."""
    w, h = 63 * SCALE, 63 * SCALE
    pad, head_h, foot_h = 10, 30, 26
    title_h = 14 * max(t.count("\n") + 1 for t in titles) + 6
    rows = (len(panels) + cols - 1) // cols
    W = cols * (w + pad) + pad
    H = head_h + rows * (h + title_h + pad) + foot_h
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((pad, 6), header, fill=INK, font=font(15))
    for i, (panel, title) in enumerate(zip(panels, titles)):
        r, c = divmod(i, cols)
        x, y = pad + c * (w + pad), head_h + r * (h + title_h + pad)
        for j, line in enumerate(title.split("\n")):
            d.text((x, y + j * 14), line, fill=INK, font=font(12))
        im.paste(up(panel), (x, y + title_h))
        if dead is not None and dead[i]:
            d.rectangle([x - 2, y + title_h - 2, x + w + 1, y + title_h + h + 1], outline=RED, width=3)
            d.text((x + 6, y + title_h + 6), "dead", fill=RED, font=font(13))
    d.text((pad, H - foot_h + 6), footer, fill=INK, font=font(13))
    return im


def save(frames, path, duration, hold_last=8):
    frames = frames + [frames[-1]] * hold_last
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=False)
    print(f"[gif] {path} {len(frames)} frames {frames[0].size}")


def imagination(a):
    z = np.load(RUNS / "gate" / "latent_check.npz")
    c = a.context
    top = int(z["gap"][c]) == 0
    rows = [("true future\n(gap %s)" % ("top" if top else "bottom"), z["recorded"][c]),
            ("other future\n(gap %s)" % ("bottom" if top else "top"), z["counterfactual"][c]),
            ("JEPA latent of the\ntrue future, decoded", z["grid_recorded_future_encoded"][c]),
            ("JEPA rollout,\noriginal predictor", z["grid_deterministic"][c]),
            ("JEPA rollout,\nresidual head, one sample", z["grid_sample"][c]),
            ("JEPA rollout,\nresidual head, mean of 8", z["grid_mean_of_8"][c]),
            ("flow rollout,\none sample", z["flow_sample"][c]),
            ("flow rollout,\nmean of 8", z["flow_mean_of_8"][c])]
    ctx = z["context"][c]
    out = []
    for t in range(ctx.shape[0]):
        out.append(sheet([ctx[t]] * len(rows), [r[0] for r in rows], f"context frame {t + 1} of 8: the wall approaches, its gap is not drawn yet",
                         "the same eight real frames are given to every model", 4))
    for t in range(rows[0][1].shape[0]):
        out.append(sheet([r[1][t] for r in rows], [r[0] for r in rows], f"imagined step {t + 1} of 8",
                         "top row: the two real futures and the JEPA latent of the true one, decoded; bottom row: what each model imagines", 4))
    save(out, a.out, a.duration)


def agents(a):
    z = np.load(a.episodes, allow_pickle=True)
    n = int(z["n"])
    labels = ["\n".join(textwrap.wrap(str(z[f"label_{i}"]), 30)) for i in range(n)]
    frames = [z[f"frames_{i}"] for i in range(n)]
    died = [bool(z[f"died_{i}"]) for i in range(n)]
    walls = [z[f"wall_{i}"] for i in range(n)]
    T = max(len(f) for f in frames)
    out = []
    for t in range(T):
        panels, dead = [], []
        for f in frames:
            k = min(t, len(f) - 1)
            panels.append(f[k]); dead.append(k == len(f) - 1 and t >= len(f) - 1)
        dead = [d and died[i] for i, d in enumerate(dead)]
        passed = [int(((w[:min(t, len(w) - 1)] != -99) & (w[1:min(t, len(w) - 1) + 1] == -99)).sum()) for w in walls]
        titles = [f"{lab}\nwalls passed: {p}" for lab, p in zip(labels, passed)]
        out.append(sheet(panels, titles, f"real gate, step {t} of 128, same reset and the same wall draws for every agent",
                         "an agent survives by retreating before the wall's gap is drawn and moving once it is", a.cols, dead))
    save(out, a.out, a.duration, hold_last=16)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("imagination"); q.add_argument("--context", type=int, default=0); q.add_argument("--duration", type=int, default=350)
    q.add_argument("--out", type=Path, required=True); q.set_defaults(fn=imagination)
    q = sub.add_parser("agents"); q.add_argument("--episodes", type=Path, default=RUNS / "gate" / "episodes.npz"); q.add_argument("--duration", type=int, default=110)
    q.add_argument("--cols", type=int, default=3); q.add_argument("--out", type=Path, required=True); q.set_defaults(fn=agents)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
