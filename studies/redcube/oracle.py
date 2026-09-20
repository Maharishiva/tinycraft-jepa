"""Model-free reference curves for the cube position over the 16-step horizon.

  actions_only      the cube moved by the known actions, wind ignored
  actions_last_wind the known actions plus the wind of the last context frame held constant
  persistence       the last context position (must match the benchmark's persistence curve)

  python -m studies.redcube.oracle --root /dev/shm/redcube/color_frame_wind --out runs/redcube/color_frame_wind
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import eval as E
import train as T
from studies.redcube import env as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = np.load(args.root / "eval.npz")
    pos, wind, actions, dones = d["pos"].astype(np.int64), d["wind"].astype(np.int64), d["actions"].astype(np.int64), d["dones"]
    starts = T.valid_starts(dones, E.CTX + E.HOR)
    t0, env = starts[:, 0] + E.CTX - 1, starts[:, 1]
    steps = np.arange(1, E.HOR + 1)
    true = pos[t0[:, None] + steps[None], env[:, None]]
    moves = np.asarray(R.MOVES)[actions[t0[:, None] + steps[None] - 1, env[:, None]]].cumsum(1)
    p0 = pos[t0, env][:, None]
    w_last = wind[t0, env][:, None]
    err = lambda p: np.abs(p - true).sum(-1).mean(0).tolist()
    out = {"windows": int(len(starts)),
           "persistence": err(p0),
           "actions_only": err(p0 + moves),
           "actions_last_wind": err(p0 + moves + w_last * steps[None, :, None]),
           "wind_abs_mean": float(np.abs(wind).mean())}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "oracle.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (np.round(v, 2).tolist() if isinstance(v, list) else v) for k, v in out.items()}))


if __name__ == "__main__":
    main()
