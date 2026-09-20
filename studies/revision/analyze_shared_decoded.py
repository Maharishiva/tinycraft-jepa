"""Aggregate the shared-CNN decoded-RGB campaign: every final.json, no selection.

Rows are world x family x training source; cells are mean and sample standard
deviation over the policy seeds that finished, with the per-seed values listed.
"""
import argparse
import json
from pathlib import Path
import statistics

WORLDS = ("gate", "gray_frame_still", "gray_frame_wind", "gray_episode_wind")
METRICS = ("survival", "walls_passed", "control_length", "return")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/revision/shared_decoded"))
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    status = json.loads((a.root / "queue" / "status.json").read_text())["jobs"]
    rows = []
    for world in WORLDS:
        for source in ("real", "dream"):
            for family in ("jepa", "flow"):
                finals = []
                for seed in range(3):
                    job = f"{world}_{family}_{source}_p{seed}"
                    if job not in status:
                        continue
                    path = a.root / "policies" / world / family / source / f"p{seed}" / "final.json"
                    if status[job]["status"] == "complete" and path.exists():
                        finals.append((seed, json.loads(path.read_text())))
                if not finals and not any(f"{world}_{family}_{source}_p{s}" in status for s in range(3)):
                    continue
                row = dict(world=world, family=family, source=source, n=len(finals),
                           statuses={s: status.get(f"{world}_{family}_{source}_p{s}", {}).get("status") for s in range(3)})
                for m in METRICS:
                    values = [f["metrics"][m] for _, f in finals]
                    row[m] = dict(values={s: v for (s, _), v in zip(finals, values)},
                                  mean=statistics.fmean(values) if values else None,
                                  sd=statistics.stdev(values) if len(values) > 1 else None)
                rows.append(row)
    print("| world | family | source | n | survival | walls | control length | return |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        def cell(m, digits):
            if r[m]["mean"] is None:
                return "-"
            sd = f" ± {r[m]['sd']:.{digits}f}" if r[m]["sd"] is not None else ""
            per = ", ".join(f"{v:.{digits}f}" for v in r[m]["values"].values())
            return f"{r[m]['mean']:.{digits}f}{sd} ({per})"
        print(f"| {r['world']} | {r['family']} | {r['source']} | {r['n']} | {cell('survival', 3)} | "
              f"{cell('walls_passed', 2)} | {cell('control_length', 1)} | {cell('return', 3)} |")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(rows=rows, statuses=status), indent=2))


if __name__ == "__main__":
    main()
