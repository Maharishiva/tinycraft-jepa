"""The flow-matching world model (tinywmcraft recipe) loaded from its own directory.

The recipe ships train.py / eval.py / kernels.py whose names collide with this repository's, so
they are imported under private names, and the recipe's eval sees its own `train` while it loads.
"""
import importlib.util
import sys
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_modules(code_dir):
    """(train, eval) modules of the recipe."""
    code_dir = Path(code_dir)
    sys.path.insert(0, str(code_dir))
    sys.modules.pop("kernels", None)
    train = import_file("flow_train", code_dir / "train.py")
    saved = sys.modules.get("train")
    sys.modules["train"] = train
    ev = import_file("flow_eval", code_dir / "eval.py")
    if saved is None:
        del sys.modules["train"]
    else:
        sys.modules["train"] = saved
    sys.path.pop(0)
    return train, ev


def load(code_dir, ckpt, n_actions):
    """Recipe modules and the checkpoint; the recipe's global action count is set to the environment's."""
    train, ev = load_modules(code_dir)
    train.A_N = n_actions
    return train, ev, train.load_model(str(ckpt))


class WM:
    """The checkpoint split into graph definitions and states so its pieces can be rebuilt inside another jit
    (the recipe jits its codec, and flax will not nest a module built outside the trace)."""

    def __init__(self, train, model, nfe):
        self.m, self.model, self.nfe = train, model, nfe
        self.win, self.ctx_signal = model.win, train.CTX_SIGNAL
        self.gdef, self.state = nnx.split(model.df)
        self.egdef, self.estate = nnx.split(model.enc)
        self.dgdef, self.dstate = nnx.split(model.dec)
        self.zmean, self.zstd = model.df.zmean.value, model.df.zstd.value
        unwrap = lambda f: f.__wrapped__ if hasattr(f, "__wrapped__") else f
        self._encode, self._decode = unwrap(train._encode), unwrap(train._decode)

    def encode(self, u8):
        return self._encode(nnx.merge(self.egdef, self.estate), self.zmean, self.zstd, u8)

    def decode(self, z):
        return self._decode(nnx.merge(self.dgdef, self.dstate), self.zmean, self.zstd, z)

    def taus(self, batch, n, real):
        """Signal level per slot: real context clean, generated frames at CTX_SIGNAL, the future at zero."""
        i = jnp.arange(self.win)
        return jnp.broadcast_to(jnp.where(i < n, jnp.where(i < real, 1.0, self.ctx_signal), 0.0), (batch, self.win))

    def flow(self, wstate, seq, taus, aseq, n, key):
        return self.m._flow_one(nnx.merge(self.gdef, wstate), seq, taus, aseq, n, key, n_steps=self.nfe)

    def read(self, wstate, seq, aseq, key):
        """Reward per frame and termination probability per frame from the model's own heads."""
        return self.m._read_heads(nnx.merge(self.gdef, wstate), seq, aseq, key)

    def imagine(self, ctx_frames, ctx_actions, actions, key):
        """Decoded rollout (B, T, H, W, 3) uint8 plus imagined rewards and done probabilities, outside jit."""
        video, rewards, dones = self.m.imagine(self.model, ctx_frames, ctx_actions, actions, key)
        video = np.asarray(video)
        if video.dtype != np.uint8:
            video = np.clip(video * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return video, np.asarray(rewards), np.asarray(dones)
