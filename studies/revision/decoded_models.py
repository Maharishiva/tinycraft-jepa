"""Frozen native world models behind a shared reconstructed-RGB interface.

No latent rescaling, extra policy inputs, learned-head replacement for Flow, or
codec retraining is performed here. The driver must set the same XLA_FLAGS before
importing JAX for both arms (the native Flow recipe uses
``--xla_gpu_enable_triton_gemm=false``), and explicitly set
``jax_default_matmul_precision=tensorfloat32`` after loading either arm.
"""
import hashlib
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import train as T
from studies.redcube import flow as F


RGB_SHAPE = (63, 63, 3)
LATENT_SHAPE = (4, 4, 32)
CODEC_BATCH = 512


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def state_sha256(state):
    """Host-side exact state fingerprint, including non-parameter variables."""
    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        # Dropout modules retain typed PRNG keys even in evaluation mode. Those
        # cannot be converted to NumPy directly, but must also remain frozen.
        dtype, shape = str(leaf.dtype), tuple(leaf.shape)
        data = jax.random.key_data(leaf) if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key) else leaf
        array = np.asarray(jax.device_get(data))
        digest.update(repr(path).encode())
        digest.update(dtype.encode())
        digest.update(repr(shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _frames(frames):
    if frames.ndim != 5 or frames.shape[-3:] != RGB_SHAPE or min(frames.shape[:2]) < 1:
        raise ValueError("posterior requires nonempty [B,T,63,63,3] frames")
    if frames.dtype != jnp.uint8:
        raise ValueError("posterior requires native uint8 RGB frames")


def _canonical_rows(apply, rows):
    """Run a framewise codec at fixed batch 512, then discard padding.

    BF16 kernels can change numerically with the batch shape. Context encoding,
    current-state encoding and imagined-state decoding therefore use the SAME
    physical codec batch shape. Padding is isolated to independent codec rows;
    it never becomes a history slot, actor observation or dynamics input.
    """
    count = rows.shape[0]
    if count < 1:
        raise ValueError("a codec call requires at least one row")
    chunks = (count + CODEC_BATCH - 1) // CODEC_BATCH
    padded = jnp.pad(rows, ((0, chunks * CODEC_BATCH - count),) + ((0, 0),) * (rows.ndim - 1))
    batches = padded.reshape((chunks, CODEC_BATCH) + rows.shape[1:])
    result = jax.lax.map(apply, batches)
    return result.reshape((chunks * CODEC_BATCH,) + result.shape[2:])[:count]


def _step_frames(previous_real, current_real):
    if previous_real.shape != current_real.shape or current_real.ndim != 4:
        raise ValueError("posterior_step requires two equally shaped [B,63,63,3] arrays")
    _frames(previous_real[:, None])
    _frames(current_real[:, None])


class FrozenAdapter:
    """Weights are graph/state snapshots; only the separately supplied AC trains."""

    def _seal(self, paths):
        self.checkpoint_paths = {name: Path(path) for name, path in paths.items()}
        self.checkpoint_sha256 = {name: sha256(path) for name, path in self.checkpoint_paths.items()}
        self.frozen_hashes = self.state_hashes()

    def state_hashes(self):
        return {name: state_sha256(state) for name, state in self._states().items()}

    def assert_frozen(self, check_files=True):
        current = self.state_hashes()
        if current != self.frozen_hashes:
            raise AssertionError(f"frozen {self.family} model state changed")
        if check_files:
            current_files = {name: sha256(path) for name, path in self.checkpoint_paths.items()}
            if current_files != self.checkpoint_sha256:
                raise AssertionError(f"frozen {self.family} checkpoint file changed")
        return current

    def posterior(self, frames):
        """[B,T,63,63,3] real uint8 -> same-shaped decoded uint8 RGB."""
        return self.decode(self.encode(frames))

    def posterior_step(self, previous_real, current_real):
        """Reconstruct a real step; previous_real MUST be an actual real frame.

        At genuine episode reset only, the caller passes the reset frame twice.
        During an episode, never pass the preceding decoded reconstruction here.
        """
        _step_frames(previous_real, current_real)
        return self.posterior(jnp.stack([previous_real, current_real], axis=1))[:, -1]

    def provenance(self):
        return dict(family=self.family, latent_shape=list(LATENT_SHAPE),
                    actor_observation="own decoded uint8 RGB in context, real steps and imagination",
                    checkpoint_paths={k: str(v) for k, v in self.checkpoint_paths.items()},
                    checkpoint_sha256=self.checkpoint_sha256, frozen_state_sha256=self.frozen_hashes,
                    metadata=self.meta, quantization=self.quantization,
                    codec_batch=CODEC_BATCH,
                    codec_batching="framewise fixed-512 chunks; zero pad final chunk, discard before dynamics")


class JepaAdapter(FrozenAdapter):
    family = "jepa"
    quantization = "clip(decoder(z) * 255 + 0.5, 0, 255).astype(uint8), native JEPA"

    def __init__(self, bundle, meta, paths):
        model = bundle.model
        cfg = meta["cfg"]
        if (cfg.get("encoder") != "tile" or model.n_tokens != 16 or
                model.latent_dim != 512 or model.history != 8):
            raise ValueError("registered JEPA requires native tile 4x4x32 and history=8")
        if bundle.decoder is None or bundle.heads is None:
            raise ValueError("JEPA requires the frozen decoder and fitted reward/done heads")
        if model.n_actions != 4 or bundle.heads.actions != 4:
            raise ValueError("registered JEPA and its heads require four actions")
        self.meta, self.history, self.n_actions = meta, model.history, model.n_actions
        self.model_graph, self.model_state = nnx.split(model)
        self.decoder_graph, self.decoder_state = nnx.split(bundle.decoder)
        self.heads_graph, self.heads_state = nnx.split(bundle.heads)
        self._seal(paths)

    def _states(self):
        return dict(model=self.model_state, decoder=self.decoder_state, heads=self.heads_state)

    def encode(self, frames):
        _frames(frames)
        model = nnx.merge(self.model_graph, self.model_state)
        encoded = _canonical_rows(lambda chunk: model.encode(chunk[:, None])[:, 0],
                                  frames.reshape((-1,) + RGB_SHAPE))
        return jax.lax.stop_gradient(encoded.reshape(frames.shape[:2] + (512,)))

    def decode(self, z):
        if z.shape[-1] != 512:
            raise ValueError("JEPA decode requires its original flat 512-dimensional latent")
        decoder = nnx.merge(self.decoder_graph, self.decoder_state)
        def one(chunk):
            return jnp.clip(decoder(chunk) * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
        pixels = _canonical_rows(one, z.reshape((-1, 512)))
        return pixels.reshape(z.shape[:-1] + RGB_SHAPE)

    def posterior_step(self, previous_real, current_real):
        # JEPA's encoder is framewise: process only B current rows, never 2B
        # paired rows or a synthetic eight-frame context for a real step.
        _step_frames(previous_real, current_real)
        return self.posterior(current_real[:, None])[:, 0]

    def predict(self, latent_history, outgoing_actions):
        model = nnx.merge(self.model_graph, self.model_state)
        return jax.lax.stop_gradient(model.predict(latent_history, model.act_embed(outgoing_actions))[:, -1])

    def read_current(self, z, action):
        heads = nnx.merge(self.heads_graph, self.heads_state)
        reward, done_logit, _ = heads(z, action)
        return reward.mean(-1), jax.nn.sigmoid(done_logit)


class FlowAdapter(FrozenAdapter):
    family = "flow"
    quantization = "clip(native_decoder(z) * 255, 0, 255).astype(uint8), native Flow"

    def __init__(self, wm, meta, paths):
        if int(wm.nfe) != 4:
            raise ValueError("registered Flow comparison uses the native sampler at NFE=4")
        if int(wm.m.GLAT) != 4 or int(wm.m.TOK) != 32 or wm.win != wm.m.WIN:
            raise ValueError("registered Flow requires native 4x4x32 and the native head-reader window")
        self.wm, self.meta, self.history = wm, meta, 8
        self.n_actions = int(meta["cfg"]["num_actions"])
        self._seal(paths)

    def _states(self):
        return dict(encoder=self.wm.estate, decoder=self.wm.dstate, dynamics_and_heads=self.wm.state)

    def encode(self, frames):
        _frames(frames)
        # Form native temporal pairs BEFORE flattening/chunking. Only the first
        # context frame has the native duplicated-frame BOS convention.
        if self.wm.model.enc.temporal:
            previous = jnp.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
            pairs = jnp.concatenate([previous, frames], axis=-1)
        else:
            pairs = frames
        return self._encode_pairs(pairs)

    def _encode_pairs(self, pairs):
        enc = nnx.merge(self.wm.egdef, self.wm.estate)
        def one(chunk):
            # Exactly the native _encode normalization and standardization;
            # pairs are already assembled, so no second temporal concatenation.
            z = enc(self.wm.m._to01(chunk))
            return (z - self.wm.zmean) / self.wm.zstd
        encoded = _canonical_rows(one, pairs.reshape((-1,) + pairs.shape[-3:]))
        return jax.lax.stop_gradient(encoded.reshape(pairs.shape[:-3] + LATENT_SHAPE))

    def decode(self, z):
        if z.shape[-3:] != LATENT_SHAPE:
            raise ValueError("Flow decode requires its original 4x4x32 latent")
        pixels = _canonical_rows(self.wm.decode, z.reshape((-1,) + LATENT_SHAPE))
        return pixels.reshape(z.shape[:-3] + RGB_SHAPE)

    def posterior_step(self, previous_real, current_real):
        # One pair per actual current state: B512 is exactly one codec chunk.
        # In particular, do not encode and decode the unused previous frame.
        _step_frames(previous_real, current_real)
        pairs = (jnp.concatenate([previous_real, current_real], axis=-1)
                 if self.wm.model.enc.temporal else current_real)
        return self.decode(self._encode_pairs(pairs))

    def provenance(self):
        return dict(super().provenance(), nfe=int(self.wm.nfe), window=int(self.wm.win),
                    temporal_codec=bool(self.wm.model.enc.temporal),
                    reward_done="original jointly fitted native Flow heads")


def load_jepa(model_path, heads_path=None):
    """Load model_probe, optionally using heads from a new full T.save bundle.

    The optional bundle must contain the identical model and decoder. This
    rejects accidentally changing the representation while replacing task heads.
    """
    bundle, meta = T.load_model(model_path)
    paths = dict(model=model_path)
    if heads_path is not None:
        head_bundle, head_meta = T.load_model(heads_path)
        for name in ("model", "decoder"):
            before, after = getattr(bundle, name), getattr(head_bundle, name)
            if before is None or after is None or state_sha256(nnx.state(before)) != state_sha256(nnx.state(after)):
                raise ValueError(f"head override changed or omitted frozen {name}")
        if head_bundle.heads is None:
            raise ValueError("head override bundle has no fitted heads")
        bundle = T.Bundle(bundle.model, bundle.decoder, head_bundle.heads)
        meta = dict(meta, heads=head_meta.get("heads"), heads_override_metadata=head_meta)
        paths["heads"] = heads_path
    return JepaAdapter(bundle, meta, paths)


def load_flow(model_path, code_dir="studies/stoch/flow_recipe", n_actions=4, nfe=4):
    """Load the actual recipe and native checkpoint, including its native heads."""
    module, _, model = F.load(code_dir, model_path, n_actions)
    meta = module.load_meta(model_path)
    if int(meta["cfg"]["num_actions"]) != int(n_actions):
        raise ValueError("Flow checkpoint action count disagrees with environment")
    return FlowAdapter(F.WM(module, model, nfe), meta, dict(model=model_path))
