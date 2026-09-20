#!/usr/bin/env python3
"""Extract native-size historical panels; never rerun models or alter pixels.

Requires Pillow and NumPy. Writes only assets/historical beside this script.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "assets" / "historical"
CONDITIONS = ("gray_frame_still", "gray_frame_wind", "gray_episode_wind")
MODELS = ("gt", "jepa", "flow")
FRAME_COUNT, SCENE_COUNT, NATIVE, SCALE = 24, 3, 63, 4


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(path: Path) -> str:
    return path.relative_to(HERE).as_posix()


def write_png(path: Path, pixels: np.ndarray) -> dict:
    Image.fromarray(pixels).save(path, optimize=True)
    with Image.open(path) as saved:
        if not np.array_equal(np.asarray(saved.convert("RGB")), pixels):
            raise RuntimeError(f"PNG did not preserve exported RGB bytes: {path}")
    return {"path": local_path(path), "sha256": sha(path),
            "width": int(pixels.shape[1]), "height": int(pixels.shape[0])}


def load_panels(path: Path) -> tuple[np.ndarray, dict]:
    """Return model × timeline × y × x × RGB, proving exact 4x blocks."""
    with Image.open(path) as gif:
        if gif.size != (804, 322):
            raise ValueError(f"Unexpected source layout in {path}: {gif.size}")
        frames, durations = [], []
        for index in range(gif.n_frames):
            gif.seek(index)
            durations.append(gif.info.get("duration"))
            frames.append(np.asarray(gif.convert("RGB")).copy())
    # Pillow has coalesced the four identical end-hold frames in these GIFs.
    # Accept explicit padding too, only after proving it duplicates frame 23.
    if len(frames) > FRAME_COUNT:
        if not all(np.array_equal(frame, frames[FRAME_COUNT - 1])
                   for frame in frames[FRAME_COUNT:]):
            raise ValueError(f"Non-padding frames after timeline in {path}")
    if len(frames) < FRAME_COUNT:
        raise ValueError(f"Incomplete timeline in {path}: {len(frames)}")
    panels = np.empty((len(MODELS), FRAME_COUNT, NATIVE, NATIVE, 3), np.uint8)
    for t, frame in enumerate(frames[:FRAME_COUNT]):
        for column in range(len(MODELS)):
            x, y, size = 12 + column * 264, 40, NATIVE * SCALE
            crop = frame[y:y + size, x:x + size]
            native = crop[::SCALE, ::SCALE]
            restored = np.repeat(np.repeat(native, SCALE, axis=0), SCALE, axis=1)
            if not np.array_equal(crop, restored):
                differing = int(np.count_nonzero(crop != restored))
                raise ValueError(f"Nonconstant 4x blocks: {path}, frame {t}, "
                                 f"column {column}, differing RGB bytes {differing}")
            panels[column, t] = native
    return panels, {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha(path),
        "canvas": [804, 322],
        "encodedFrames": len(frames),
        "sourceDurationsMs": durations,
        "timelineFramesExported": FRAME_COUNT,
        "timelineDurationMs": 250,
        "endHoldOmittedMs": max(0, sum(d or 0 for d in durations) - FRAME_COUNT * 250),
        "exact4xBlockValidation": "passed for all three panels at all 24 steps",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": 1,
        "artifactKind": "historical GIF panel export; no new evaluation",
        "conditions": {},
        "contract": {
            "nativeSize": [NATIVE, NATIVE],
            "spriteSize": [FRAME_COUNT * NATIVE, SCENE_COUNT * NATIVE],
            "spriteLayout": "24 timeline frames horizontally, 3 scenes vertically",
            "spriteCrop": "x=timelineIndex*63, y=sceneIndex*63, width=63, height=63",
            "contextIndices": [0, 7],
            "forecastIndices": [8, 23],
            "timeline": [{"index": t, "phase": "context" if t < 8 else "forecast",
                          "label": f"Context {t + 1}/8" if t < 8 else f"Future +{t - 7}",
                          "displayDurationMs": 250} for t in range(FRAME_COUNT)],
            "modelStreams": {
                "gt": "Recorded real frames throughout all 24 steps",
                "jepa": "Decoded real context (indices 0-7), then decoded JEPA predictions (8-23)",
                "flow": "Decoded real context (indices 0-7), then sampled decoded flow predictions (8-23)",
            },
            "decodedRealFutureAvailable": False,
            "actionsAvailable": False,
            "extraction": "Crop 252x252 at (12+264*column,40), then select each 4th pixel; "
                          "every source crop is verified to consist of identical 4x4 RGB blocks",
            "pixelCaveat": "Source GIFs already have palette quantization. PNGs exactly preserve "
                           "the RGB pixels decoded from those GIFs, not original model float outputs. "
                           "Do not use this export to recompute quantitative image metrics.",
            "timingCaveat": "The source GIF's extra 1000ms final hold is omitted. "
                            "No content-based deduplication of the 24 timeline steps is performed.",
        },
        "exporter": {"path": "submission/html-redcube/export_historical.py", "sha256": sha(Path(__file__))},
    }
    for condition in CONDITIONS:
        source_dir = ROOT / "runs" / "redcube" / condition
        bench_path = source_dir / "bench.json"
        bench = json.loads(bench_path.read_text())
        selected = bench["scenes"]
        if len(selected) != SCENE_COUNT:
            raise ValueError(f"Expected three selected windows for {condition}: {selected}")
        if "tile_factored" not in bench["benchmark"] or "flow" not in bench["benchmark"]:
            raise ValueError(f"Missing expected model identities in {bench_path}")
        scene_panels, gif_sources = [], []
        for scene in range(SCENE_COUNT):
            panels, source = load_panels(source_dir / f"scene{scene}.gif")
            source.update({"sceneIndex": scene, "selectedWindow": selected[scene]})
            scene_panels.append(panels)
            gif_sources.append(source)
        streams = np.stack(scene_panels)  # scene, model, time, y, x, RGB
        entry = {
            "scenes": SCENE_COUNT, "context": 8, "horizon": 16,
            "keys": {}, "posters": {}, "selectedWindows": selected,
            "source": {
                "gifs": gif_sources,
                "benchmark": {"path": bench_path.relative_to(ROOT).as_posix(), "sha256": sha(bench_path)},
                "jepaArm": "tile_factored",
                "flowArm": "flow",
                "selectionRule": "Highest cumulative GT position L1 travel between the 16 future frames, "
                                 "among the benchmark metric-window subset; not random examples.",
                "selectionCode": "studies/redcube/bench.py:501-518",
                "selectionFormula": "argsort(-abs(diff(pos[:,8:],axis=1)).sum((1,2)))[:3]",
                "selectionScopeCaveat": "Current code defaults to the first 512 sampled valid evaluation "
                                       "windows. Exact original CLI and sampled (time,stream) indices are "
                                       "not embedded in these legacy files. selectedWindows are row indices "
                                       "into that sampled window batch, not episode IDs.",
            },
            "assets": {},
        }
        for model_index, model in enumerate(MODELS):
            sheet = np.concatenate([np.concatenate(streams[s, model_index], axis=1)
                                    for s in range(SCENE_COUNT)], axis=0)
            asset = write_png(OUT / f"{condition}-{model}.png", sheet)
            entry["keys"][model] = asset["path"]
            entry["assets"][model] = asset
            entry["posters"][model] = {}
            for label, time_index in (("context", 7), ("future4", 11)):
                poster = write_png(OUT / f"{condition}-{model}-{label}.png", streams[0, model_index, time_index])
                poster.update({"sceneIndex": 0, "timelineIndex": time_index,
                               "label": "Context 8/8" if time_index == 7 else "Future +4"})
                entry["posters"][model][label] = poster
        payload["conditions"][condition] = entry
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    (OUT / "manifest.json").write_text(encoded)
    (OUT / "historical-story.js").write_text("window.HISTORICAL_STORY = " + encoded.rstrip() + ";\n")
    print(json.dumps({"conditions": list(payload["conditions"]), "sprites": 9, "posters": 18,
                      "validatedNativePanels": SCENE_COUNT * FRAME_COUNT * len(MODELS) * len(CONDITIONS),
                      "manifest": str(OUT / "manifest.json"),
                      "javascript": str(OUT / "historical-story.js")}, indent=2))


if __name__ == "__main__":
    main()
