"""Shared stock-CNN control of frozen JEPA and Flow through decoded RGB.

Every imagination start is an actual eight-frame archive context. Both actors
observe their own posterior reconstructions in burn-in and real evaluation, and
their own model reconstructions in imagination. Gate decoded-real controls use
exact transitions/signals with the identical posterior observation interface.
No reward shaping, latent scaling, fabricated reset history, or model update.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from studies.revision import decoded_models as D
from studies.revision import decoded_dream as DD
from studies.revision import matched_ppo as M
from studies.revision import warm_context_eval as W
from studies.stoch import env as S
from studies.stoch import rl as P

jax.config.update("jax_default_matmul_precision", "tensorfloat32")
WORLDS = ("gray_frame_still", "gray_frame_wind", "gray_episode_wind", "gate")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sample_contexts(rng, data, table, batch):
    ids = rng.integers(0, len(table), batch)
    t, e = table[ids, 0], table[ids, 1]
    ti = t[:, None] + np.arange(P.CTX)
    ei = np.broadcast_to(e[:, None], ti.shape)
    frames = data["frames"][ti, ei]
    incoming = data["actions"][ti - 1, ei].astype(np.int32)
    identity = np.stack([t, e, t + P.CTX - 1], axis=1).astype(np.int32)
    return jnp.asarray(frames), jnp.asarray(incoming), identity


def state_at(data, episode_age, identities):
    t, e = identities[:, 2], identities[:, 1]
    n = len(t)
    return S.State(jnp.asarray(data["pos"][t, e], jnp.int32),
                   jnp.asarray(data["wind"][t, e], jnp.int32),
                   jnp.zeros((n, 2), jnp.uint32),
                   jnp.asarray(episode_age[t, e], jnp.int32),
                   jnp.asarray(data["episode"][t, e], jnp.int32),
                   jnp.asarray(data["wall_x"][t, e], jnp.int32),
                   jnp.asarray(data["gap"][t, e], jnp.int32))


def make_real_collector(adapter, spec, args):
    """Exact gate dynamics/signals; same decoded RGB and policy history contract."""
    if args.config != "gate":
        raise ValueError("the prospective decoded-real positive controls are gate only")
    B, H = args.batch, args.hor
    step_env = jax.vmap(S.make(spec)[1])

    @nnx.jit
    def collect(ac, raw, incoming, state, key):
        decoded = adapter.posterior(raw)
        g = DD._burn(ac, decoded, incoming, B)

        def step(carry, k):
            state, real, stopped, g, previous_action, observation = carry
            ka, ks, _ = jax.random.split(k, 3)
            logits, values, ng = ac.step(observation, previous_action, g)
            action = jax.random.categorical(ka, logits).astype(jnp.int32)
            frame, nxt, reward, done = step_env(state, action, jax.random.split(ks, B))
            choose = lambda old, new: jnp.where(stopped.reshape((B,) + (1,) * (new.ndim - 1)), old, new)
            nxt = jax.tree.map(choose, state, nxt)
            frame = choose(real, frame)
            reward, done = jnp.where(stopped, 0.0, reward), stopped | done
            next_observation = adapter.posterior_step(real, frame)
            output = (observation, previous_action, g, action, P.logprob(logits, action),
                      P.hl_expect(values), reward, done.astype(jnp.float32))
            return (nxt, frame, done, ng, action, next_observation), output

        initial = (state, raw[:, -1], jnp.zeros(B, bool), g, incoming[:, -1], decoded[:, -1])
        (_, _, _, g, previous_action, observation), outputs = jax.lax.scan(step, initial, jax.random.split(key, H))
        return DD._finish(ac, outputs, g, previous_action, observation, B, args.gamma, args.lam)
    return collect


def run(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("refusing to overwrite existing experiment artifacts")
    expected_updates = 150 if args.config == "gate" else 50
    if not args.smoke and (args.updates != expected_updates or args.batch != 512 or args.hor != 48
                           or args.chunk != 16 or args.mb != 128 or args.epochs != 4):
        raise ValueError("scientific budgets are fixed prospectively for each world")
    if args.smoke and args.updates != 2:
        raise ValueError("computational smoke must use exactly two updates")
    if args.hor % args.chunk or args.batch % args.mb or args.policy_seed not in (0, 1, 2):
        raise ValueError("invalid registered policy/optimizer shape")
    for name, value in dict(lr=2.5e-4, gamma=.99, lam=.9, clip=.2, vf=.5, ent=.01).items():
        if getattr(args, name) != value:
            raise ValueError(f"registered shared optimizer requires {name}={value}")
    root = Path(__file__).resolve().parents[2]
    if args.flow_code.resolve() != (root / "studies/stoch/flow_recipe").resolve():
        raise ValueError("registered comparison requires the pinned native Flow recipe")
    if args.family == "jepa" and args.train_source == "dream" and not args.smoke and args.heads is None:
        raise ValueError("registered JEPA imagination requires the full-training-label head refit")
    args.out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    spec = S.CONFIGS[args.config]
    fields = ["frames", "actions", "dones"]
    if args.train_source == "real":
        fields += ["pos", "wind", "episode", "wall_x", "gap"]
    with np.load(args.data) as archive:
        data = {name: archive[name] for name in fields}
        data_meta = json.loads(str(archive["meta"]))
    recorded_spec = dict(data_meta["spec"])
    recorded_spec.setdefault("gate", False)
    if recorded_spec != asdict(spec) or data_meta["split"] != "train" or data["actions"].size != 500_000:
        raise ValueError("wrong world or training archive")
    data["actions"] = data["actions"].astype(np.int32)
    _, uniform = P.start_tables(data["dones"], *data["actions"].shape, P.CTX)
    table = np.stack(uniform, axis=1)
    if len(table) == 0:
        raise ValueError("no valid recorded contexts")
    ages = M.episode_steps(data["episode"], data["dones"]) if args.train_source == "real" else None
    reconstruction = M.validate_reconstruction(spec, data, ages) if args.train_source == "real" else None
    adapter = (D.load_jepa(args.model, args.heads) if args.family == "jepa"
               else D.load_flow(args.model, args.flow_code, S.N_ACTIONS, nfe=4))
    data_sha = sha256(args.data)
    if adapter.meta.get("data_sha256") != data_sha:
        raise ValueError("checkpoint training archive differs from this exact comparison archive")
    if args.heads is not None:
        head_meta = adapter.meta.get("heads_override_metadata", {})
        if head_meta.get("data_sha256") != data_sha or head_meta.get("heads", {}).get("training_data_sha256") != data_sha:
            raise ValueError("JEPA head override is not fitted on this exact training archive")
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")
    ac = P.AC(nnx.Rngs(args.policy_seed + 2), obs="pixels", dim=S.H * S.W * 3)
    optimizer = nnx.Optimizer(ac, optax.chain(optax.clip_by_global_norm(1.0),
        optax.adamw(args.lr, weight_decay=0.01)), wrt=nnx.Param)
    update = M.make_update(args)
    collector = (DD.make_dream(adapter, args.batch, args.hor, args.gamma, args.lam)
                 if args.train_source == "dream" else make_real_collector(adapter, spec, args))
    packet = W.load_or_create_packet(args.eval_packet, args.config)
    evaluate = W.make_evaluator(spec, adapter, packet)
    source_names = ("studies/revision/shared_decoded_ppo.py", "studies/revision/decoded_models.py",
                    "studies/revision/decoded_dream.py", "studies/revision/warm_context_eval.py",
                    "studies/revision/matched_ppo.py", "studies/stoch/rl.py", "studies/stoch/env.py",
                    "studies/redcube/env.py", "studies/redcube/flow.py", "train.py", "probe.py",
                    "studies/stoch/flow_recipe/train.py", "studies/stoch/flow_recipe/eval.py",
                    "studies/stoch/flow_recipe/kernels.py")
    provenance = {"protocol": "shared_decoded_cnn_v1", "args": vars(args),
                  "model": adapter.provenance(), "data_sha256": data_sha,
                  "data_metadata": data_meta, "source_sha256": {name: sha256(root / name) for name in source_names},
                  "policy_params": P.n_params(ac), "policy_observation": "own posterior/predicted uint8 RGB",
                  "input_shape": [63, 63, 3], "world_model_coordinates": 512,
                  "real_state_reconstruction": reconstruction,
                  "start_protocol": "uniform actual8-frame contexts; no reset-padding starts",
                  "context_table_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
                  "eval_packet_fingerprint": packet.fingerprint(), "warm_eval_protocol": packet.metadata,
                  "jax_version": jax.__version__, "matmul_precision": str(jax.config.jax_default_matmul_precision),
                  "xla_flags": os.environ.get("XLA_FLAGS"), "devices": [str(device) for device in jax.devices()],
                  "checkpoint_selection": "fixed final update; no intermediate score selection"}
    if args.family == "flow":
        cfg = adapter.meta["cfg"]
        provenance["supervision_stages"] = {
            "tokenizer": "pixel reconstruction; frozen during dynamics fitting",
            "dynamics": "joint native denoising, reward and termination losses",
            "reward_weight": cfg["reward_weight"], "done_weight": cfg["done_weight"],
            "head_warm_fraction": cfg["head_warm"], "later_head_trunk_gradient_scale": cfg["head_trunk"],
            "initial_head_trunk_gradient_scale": 1.0, "dynamics_steps": cfg["wm_steps"],
            "label_opportunity": cfg["data_budget"],
            "scope": "same label opportunity does not isolate sampling or prediction loss"}
    else:
        provenance["supervision_stages"] = {
            "world_model": "reward-free joint latent prediction plus SIGReg",
            "decoder": "posthoc pixel fit on frozen encoder", "decoder_frames": adapter.meta["decoder"]["frames"],
            "task_heads": "posthoc supervised fit; encoder, predictor and decoder frozen",
            "head_transitions": adapter.meta["heads"].get("transitions"),
            "head_override_used": args.heads is not None,
            "scope": "complete frozen recipes; not an objective-only ablation"}
    (args.out / "config.json").write_text(json.dumps(provenance, default=str, indent=2))
    print("[init] " + json.dumps({"family": args.family, "source": args.train_source,
          "policy_params": P.n_params(ac), "contexts": len(table), "setup_seconds": time.time() - began}), flush=True)
    host = np.random.default_rng(args.policy_seed + 11)
    key = jax.random.PRNGKey(args.policy_seed + 7)
    starts = []
    train_start = time.time()
    for u in range(1, args.updates + 1):
        started = time.time()
        raw, incoming, identities = sample_contexts(host, data, table, args.batch)
        starts.append(identities)
        key, rollout_key = jax.random.split(key)
        if args.train_source == "real":
            batch, diag = collector(ac, raw, incoming, state_at(data, ages, identities), rollout_key)
        else:
            batch, diag = collector(ac, raw, incoming, rollout_key)
        diag = np.asarray(diag)
        for _ in range(args.epochs):
            for ci in range(args.hor // args.chunk):
                for bi in range(args.batch // args.mb):
                    loss = np.asarray(update(ac, optimizer, batch, jnp.asarray(ci), jnp.asarray(bi)))
        if not np.isfinite(diag).all() or not np.isfinite(loss).all():
            raise FloatingPointError("nonfinite loss/diagnostic; no automatic retuning or replacement")
        row = {"update": u, "positions": u * args.batch * args.hor,
               "seconds": round(time.time() - started, 3), "reward": float(diag[0]), "done": float(diag[1]),
               "rollout_return": float(diag[2]), "alive": float(diag[3]),
               "loss": float(loss[0]), "pg": float(loss[1]), "vf": float(loss[2]), "entropy": float(loss[3])}
        with (args.out / "ledger.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print("[update] " + json.dumps(row), flush=True)
    training_seconds = time.time() - train_start
    adapter.assert_frozen()
    outcome = evaluate(ac)
    summary = W.save_evaluation(outcome, packet, args.out / "evaluation", policy_seed=args.policy_seed,
                                provenance={"family": args.family, "train_source": args.train_source,
                                            "model_sha256": adapter.checkpoint_sha256})
    adapter.assert_frozen()
    final = {"family": args.family, "world": args.config, "train_source": args.train_source,
             "policy_seed": args.policy_seed, "smoke": args.smoke, "updates": args.updates,
             "budget_complete": True, "nominal_positions": args.updates * args.batch * args.hor,
             "optimizer_steps": args.updates * args.epochs * (args.hor // args.chunk) * (args.batch // args.mb),
             "training_seconds": training_seconds, "total_seconds": time.time() - began,
             "metrics": {name: summary[name] for name in ("return", "survival", "total_length", "control_length", "walls_passed", "n_ep")},
             "eval_packet_fingerprint": packet.fingerprint(), "model_sha256": adapter.checkpoint_sha256}
    P.save_ac(ac, args.out / "model.pkl", final)
    final["policy_sha256"] = sha256(args.out / "model.pkl")
    np.savez_compressed(args.out / "starts.npz", indices=np.stack(starts), columns=np.array(["context_start_t", "env", "last_frame_t"]))
    (args.out / "final.json").write_text(json.dumps(final, indent=2))
    print("[final] " + json.dumps(final), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=WORLDS, required=True)
    p.add_argument("--family", choices=("jepa", "flow"), required=True)
    p.add_argument("--train-source", choices=("dream", "real"), default="dream")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--heads", type=Path)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--flow-code", type=Path, default=Path("studies/stoch/flow_recipe"))
    p.add_argument("--eval-packet", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--policy-seed", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--updates", type=int, required=True)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--hor", type=int, default=48)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--mb", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=.99)
    p.add_argument("--lam", type=float, default=.9)
    p.add_argument("--clip", type=float, default=.2)
    p.add_argument("--vf", type=float, default=.5)
    p.add_argument("--ent", type=float, default=.01)
    args = p.parse_args()
    if args.train_source == "real" and args.config != "gate":
        p.error("the registered real-transition controls are gate only")
    run(args)


if __name__ == "__main__":
    main()
