"""Full-checkpoint canonical codec parity; no training or outcome selection."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from studies.revision import decoded_models as D
from studies.revision import warm_context_eval as W
from studies.revision import refit_full_label_heads as H


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("jepa", "flow"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--heads", type=Path)
    p.add_argument("--flow-code", type=Path, default=Path("studies/stoch/flow_recipe"))
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--config", choices=W.WORLDS, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError("do not overwrite codec parity evidence")
    adapter = D.load_jepa(args.model, args.heads) if args.family == "jepa" else D.load_flow(args.model, args.flow_code)
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")
    packet = W.load_or_create_packet(args.packet, args.config)
    rows = []
    for count in (256, 512):
        raw = jnp.asarray(np.tile(packet.raw_context, (count // 256, 1, 1, 1, 1)))

        @jax.jit
        def context(frames):
            z = adapter.encode(frames)
            return z[:, -1], adapter.decode(z)[:, -1]

        @jax.jit
        def current(previous, now):
            if args.family == "jepa":
                z = adapter.encode(now[:, None])[:, -1]
            else:
                pairs = jnp.concatenate([previous, now], axis=-1) if adapter.wm.model.enc.temporal else now
                z = adapter._encode_pairs(pairs)
            return z, adapter.posterior_step(previous, now)

        z_context, rgb_context = jax.device_get(context(raw))
        z_current, rgb_current = jax.device_get(current(raw[:, -2], raw[:, -1]))
        latent_max = float(np.max(np.abs(z_context - z_current)))
        pixel_max = int(np.max(np.abs(rgb_context.astype(np.int16) - rgb_current.astype(np.int16))))
        rows.append(dict(batch=count, latent_max_abs=latent_max, pixel_max_abs=pixel_max,
                         exact=bool(np.array_equal(z_context, z_current) and np.array_equal(rgb_context, rgb_current))))
        if args.family == "jepa":
            current_frames = np.asarray(raw[:, -1])
            z_head = np.asarray(H.make_cached_encoder(adapter)(jnp.asarray(current_frames)))
            difference = float(np.max(np.abs(z_head - z_current)))
            rows.append(dict(batch=count, path="head-cache-versus-adapter", latent_max_abs=difference,
                             exact=bool(np.array_equal(z_head, z_current))))
    adapter.assert_frozen()
    result = {"family": args.family, "config": args.config, "packet_fingerprint": packet.fingerprint(),
              "adapter": adapter.provenance(), "cases": rows, "passed": all(row["exact"] for row in rows),
              "source_sha256": {str(path): D.sha256(path) for path in (Path(__file__), Path(D.__file__), Path(W.__file__), Path(H.__file__))}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"passed": result["passed"], "cases": rows}), flush=True)
    if not result["passed"]:
        raise SystemExit("canonical posterior parity failed; do not silently relax this guard")


if __name__ == "__main__":
    main()
