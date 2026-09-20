"""Frozen, native-width JEPA behind the original Craftax pixel-PPO env API.

This adapter changes neither rewards nor latents. The reward and termination
readouts consume (z_t, a_t); the one-step predictor consumes outgoing actions.
The policy sees decoded uint8 RGB converted to [0, 1], as in wm_gym.py.
Reset padding repeats the episode-start frame with zero outgoing actions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import train as T

EXPECTED_MODEL_SHA256 = "59cc282c9bd7a49ab1c272342220bc8e0d8ca0f1aa77bbcb4c23ae31dfd6835d"
EXPECTED_DATA_SHA256 = "09f2cb41a38b0f34937a89c4122f6b25bbb3094e34a077b3a572504269d0a1b2"
FORMAT = "frozen_jepa_inventory_v1"


def file_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def weights_digest(module):
    """Hash parameter and batch-statistic arrays, excluding inference RNG state."""
    digest = hashlib.sha256()
    for kind, state in zip(("p", "b"), T._weights(module)):
        for i, leaf in enumerate(jax.tree.leaves(state)):
            array = np.ascontiguousarray(jax.device_get(leaf))
            digest.update(f"{kind}.{i}:{array.dtype}:{array.shape}:".encode())
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def assert_frozen_archive(source_path, fitted_path):
    """Require exact preservation of every stored model parameter/statistic."""
    with np.load(source_path, allow_pickle=False) as source, np.load(fitted_path, allow_pickle=False) as fitted:
        names = sorted(name for name in source.files if name.startswith(("model.p.", "model.b.")))
        actual = sorted(name for name in fitted.files if name.startswith(("model.p.", "model.b.")))
        if not names or names != actual:
            raise ValueError("Frozen model array keys differ")
        for name in names:
            a, b = source[name], fitted[name]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                raise ValueError(f"Frozen model array changed: {name}")


class TransitionHeads(nnx.Module):
    """Original probe reward/done architectures, without an unused value head."""

    def __init__(self, dim, hidden, rngs, actions=17):
        self.actions = actions
        self.reward = nnx.List([
            nnx.Linear(dim + actions, hidden, rngs=rngs, **T.MP),
            nnx.Linear(hidden, hidden, rngs=rngs, **T.MP),
            nnx.Linear(hidden, 1, rngs=rngs, **T.MP),
        ])
        self.done = nnx.List([
            nnx.Linear(dim + actions, hidden, rngs=rngs, **T.MP),
            nnx.Linear(hidden, 1, rngs=rngs, **T.MP),
        ])

    @staticmethod
    def _run(layers, x):
        for layer in layers[:-1]:
            x = nnx.gelu(layer(x))
        return layers[-1](x).astype(jnp.float32)[..., 0]

    def __call__(self, latent, action):
        joint = jnp.concatenate([
            latent, jax.nn.one_hot(action, self.actions, dtype=latent.dtype)
        ], axis=-1)
        return self._run(self.reward, joint), self._run(self.done, joint)


def validate_model(model, meta):
    if (model.latent_dim, model.history, model.n_actions) != (512, 8, 17):
        raise ValueError("Inventory protocol requires native 512-D, history-8, 17-action JEPA")
    if meta["cfg"].get("num_preds", 1) != 1:
        raise ValueError("Inventory adapter requires a one-step predictor")
    if meta.get("data_sha256") != EXPECTED_DATA_SHA256:
        raise ValueError("Frozen JEPA was not trained on the registered archive")


def load_bundle(path):
    bundle, meta = T.load_model(path)
    validate_model(bundle.model, meta)
    if meta.get("inventory_probe", {}).get("format") != FORMAT:
        raise ValueError("Expected a refitted inventory bundle; historical heads are not accepted")
    if bundle.decoder is None:
        raise ValueError("Inventory bundle lacks a decoder")
    spec = meta["inventory_heads"]
    with np.load(path, allow_pickle=False) as archive:
        heads = T._restore(TransitionHeads(512, spec["hidden"], T.rngs(), 17), archive, "inventory_heads")
    expected = meta["inventory_probe"]["frozen_model_weights_sha256"]
    if weights_digest(bundle.model) != expected:
        raise ValueError("Frozen model weight digest differs from fitting manifest")
    return bundle.model, bundle.decoder, heads, meta


def validate_registered_bundle(path):
    """Admit only the completed, fixed-recipe refit to production policy training.

    This deliberately requires the final sidecar: an interrupted fitting process
    may have written an NPZ before its byte-equality/round-trip checks completed.
    """
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"]))
    hashes = {"model": EXPECTED_MODEL_SHA256, "data": EXPECTED_DATA_SHA256}
    for record in (manifest, meta.get("inventory_probe", {})):
        if record.get("format") != FORMAT or record.get("status") != "complete" or record.get("source_sha256") != hashes:
            raise ValueError("Not a completed refit of the registered model and data")
    fixed = {"steps": 8000, "batch": 256, "lr": 1e-3, "hidden": 256, "depth": 6,
             "head_steps": 6000, "head_batch": 1024, "head_hidden": 512, "head_lr": 1e-3, "seed": 0}
    if any(manifest.get("arguments", {}).get(name) != value for name, value in fixed.items()):
        raise ValueError("Fitted readout recipe differs from the registered defaults")
    for name, expected in (
        ("decoder", {"steps": 8000, "batch": 256, "lr": 1e-3, "hidden": 256, "depth": 6, "tokens": 16, "frames": 500_000}),
        ("inventory_heads", {"steps": 6000, "batch": 1024, "lr": 1e-3, "hidden": 512, "transitions": 500_000}),
    ):
        if any(meta.get(name, {}).get(key) != value for key, value in expected.items()):
            raise ValueError(f"Registered {name} metadata mismatch")
    cfg = meta.get("cfg", {})
    if any(cfg.get(key) != value for key, value in {"seed": 0, "tokens": 16, "token_dim": 32, "history": 8, "n_actions": 17, "num_preds": 1}.items()):
        raise ValueError("Registered frozen JEPA configuration mismatch")
    if not manifest.get("frozen_array_verification") or not manifest.get("saved_bundle_roundtrip_verified"):
        raise ValueError("Successful frozen-array and reload checks are absent")
    digest = file_sha256(path)
    if manifest.get("artifacts_sha256", {}).get(path.name) != digest:
        raise ValueError("Readout bundle differs from the completed fitting manifest")
    _, _, _, loaded_meta = load_bundle(path)
    if loaded_meta["inventory_probe"]["frozen_model_weights_sha256"] != manifest.get("frozen_model_weights_sha256"):
        raise ValueError("Frozen model digest disagrees between bundle and sidecar")
    return {"path": str(path), "bundle_sha256": digest, "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path), "source_sha256": hashes,
            "fixed_arguments": fixed, "frozen_model_weights_sha256": manifest["frozen_model_weights_sha256"]}


class WorldModel:
    """Rebuild NNX modules inside the active JAX trace, like GenericWM.fresh."""

    def __init__(self, path):
        model, decoder, heads, self.meta = load_bundle(path)
        self.win, self.latent_dim = model.history, model.latent_dim
        self.mgraph, self.mstate = nnx.split(model)
        self.dgraph, self.dstate = nnx.split(decoder)
        self.hgraph, self.hstate = nnx.split(heads)

    def encode(self, frames_u8):
        return nnx.merge(self.mgraph, self.mstate).encode(frames_u8, train=False)

    def decode(self, latent):
        pixels = nnx.merge(self.dgraph, self.dstate)(latent)
        # Only pixel quantization: no clipping of rewards or latent states.
        return jnp.clip(pixels * 255.0 + 0.5, 0, 255).astype(jnp.uint8)

    def predict(self, latent_history, outgoing_actions):
        model = nnx.merge(self.mgraph, self.mstate)
        return model.predict(latent_history, model.act_embed(outgoing_actions), train=False)[:, -1]

    def read(self, current_latent, action):
        reward, logit_done = nnx.merge(self.hgraph, self.hstate)(current_latent, action)
        return reward, jax.nn.sigmoid(logit_done)

    def parameter_sha256(self):
        """Digest every frozen WM/readout parameter and BatchNorm statistic."""
        digests = [weights_digest(nnx.merge(graph, state)) for graph, state in (
            (self.mgraph, self.mstate), (self.dgraph, self.dstate), (self.hgraph, self.hstate)
        )]
        return hashlib.sha256("".join(digests).encode()).hexdigest()


def episode_start_indices(dones):
    dones = np.asarray(dones)
    if dones.ndim != 2 or not dones.shape[0] or not dones.shape[1]:
        raise ValueError("Expected nonempty time-by-environment dones")
    starts = np.concatenate([np.ones((1, dones.shape[1]), bool), dones[:-1] > 0.5], axis=0)
    return np.stack(np.nonzero(starts), axis=-1).astype(np.int32)


def reset_frames(path):
    with np.load(path, allow_pickle=False) as archive:
        indices = episode_start_indices(archive["dones"])
        return np.asarray(archive["frames"])[indices[:, 0], indices[:, 1]]


def transition(wm, state, action, done_key, max_steps):
    """Pure transition helper, exposed for action/reward-alignment tests."""
    reward, pdone = wm.read(state["seq"][:, -1], action)
    outgoing = state["aseq"].at[:, -1].set(action)
    latent = wm.predict(state["seq"], outgoing)
    seq = jnp.concatenate([state["seq"][:, 1:], latent[:, None]], axis=1)
    aseq = jnp.concatenate([outgoing[:, 1:], jnp.zeros_like(outgoing[:, :1])], axis=1)
    timestep = state["t"] + 1
    done = (jax.random.uniform(done_key, action.shape) < pdone) | (timestep >= max_steps)
    obs = wm.decode(latent).astype(jnp.float32) / 255.0
    return obs, {"seq": seq, "aseq": aseq, "t": timestep}, reward, done, {}


def make_env(wm_path, data_path=None, nfe=4, max_steps=1024, once=False, chunk=64,
             *, reset_frames_u8=None, reset_contexts_u8=None, reset_actions=None):
    """Return (wm, vreset, vstep) with the original float-pixel PPO signatures.

    nfe is accepted solely for call compatibility; JEPA predicts once per action.
    vreset(keys, indices=None) accepts explicit reset-bank indices for pairing.
    Optional genuine bootstrap contexts contain 8 frames and 7 outgoing actions.
    The production comparison uses the same episode-start frames for both WMs.
    """
    del nfe
    if once:
        raise ValueError("Action-based reward modification is forbidden")
    if max_steps <= 0 or chunk <= 0:
        raise ValueError("max_steps and chunk must be positive")
    if reset_frames_u8 is not None and reset_contexts_u8 is not None:
        raise ValueError("Provide reset frames or bootstrap contexts, not both")
    wm = WorldModel(wm_path)
    if reset_contexts_u8 is not None:
        contexts = np.asarray(reset_contexts_u8)
        if contexts.ndim != 5 or contexts.shape[1:] != (wm.win, 63, 63, 3) or contexts.dtype != np.uint8:
            raise ValueError("Bootstrap contexts must be uint8 [R,8,63,63,3]")
        frames = contexts[:, -1]
        actions = np.asarray(reset_actions) if reset_actions is not None else None
        if actions is None or actions.shape != (len(contexts), wm.win - 1):
            raise ValueError("Bootstrap contexts require [R,7] outgoing actions")
        if not np.issubdtype(actions.dtype, np.integer) or np.any((actions < 0) | (actions >= 17)):
            raise ValueError("Bootstrap actions must be integer action IDs in [0,17)")
        actions = np.concatenate([actions.astype(np.int32), np.zeros((len(contexts), 1), np.int32)], 1)
    else:
        if reset_actions is not None:
            raise ValueError("Reset actions are only meaningful with genuine contexts")
        if reset_frames_u8 is None and data_path is None:
            raise ValueError("Provide data_path or a reset frame bank")
        frames = np.asarray(reset_frames_u8 if reset_frames_u8 is not None else reset_frames(data_path))
        if frames.ndim != 4 or frames.shape[1:] != (63, 63, 3) or frames.dtype != np.uint8:
            raise ValueError("Reset frames must be uint8 [R,63,63,3]")
        contexts = None
        actions = np.zeros((len(frames), wm.win), np.int32)
    if not len(frames):
        raise ValueError("Reset bank is empty")
    encode = jax.jit(wm.encode)
    chunks = []
    for start in range(0, len(frames), chunk):
        source = (contexts[start:start + chunk] if contexts is not None
                  else frames[start:start + chunk, None])
        z = encode(jnp.asarray(source))
        chunks.append(z if contexts is not None else jnp.repeat(z, wm.win, axis=1))
    rz, rf, ra = jnp.concatenate(chunks), jnp.asarray(frames), jnp.asarray(actions)

    def vreset(keys, indices=None):
        n = keys.shape[0]
        ids = jax.random.randint(keys[0], (n,), 0, len(frames)) if indices is None else jnp.asarray(indices)
        if ids.shape != (n,):
            raise ValueError("Explicit reset indices must have shape [number of keys]")
        return rf[ids].astype(jnp.float32) / 255.0, {
            "seq": rz[ids], "aseq": ra[ids], "t": jnp.zeros((n,), jnp.int32)
        }

    def vstep(keys, state, action):
        # Match the original wrapper's third split for sampled termination.
        _, _, done_key = jax.random.split(keys[0], 3)
        return transition(wm, state, action, done_key, max_steps)

    return wm, vreset, vstep
