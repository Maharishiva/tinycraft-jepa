"""Optional native-grid policy sensitivity for the matched PPO diagnostic.

The default factory returns the unmodified latent MLP actor-critic. The optional
CNN sees exactly the same 512 coordinates, restored to their existing 4x4x32
spatial layout. It does not decode, resize, replicate, scale, or otherwise add
observations. Its two SAME convolutions preserve the native spatial resolution.
The recurrent core, actor, critic, activation functions and initializers follow
``studies.stoch.rl.AC``. All diagnostic arms must use the same selected torso.

This is an architecture sensitivity, not a parameter-matched architecture
comparison: the native-grid torso has 317,824 parameters versus the original
MLP's 393,984, while their recurrent cores and heads have identical shapes.
Only use a grid whose spatial ordering is established by the encoder metadata;
a global latent vector must not be assigned an artificial grid.
"""

import math
import pickle
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from studies.stoch import rl as P


TORSOS = ("mlp", "native_grid")
NATIVE_GRID = (4, 4, 32)


class NativeGridAC(P.AC):
    """The existing recurrent actor-critic with a smaller native-grid torso."""

    def __init__(self, rngs, dim=math.prod(NATIVE_GRID), grid_shape=NATIVE_GRID):
        self.grid_shape = tuple(int(x) for x in grid_shape)
        if len(self.grid_shape) != 3 or any(x <= 0 for x in self.grid_shape):
            raise ValueError("grid_shape must contain three positive dimensions")
        self.obs, self.dim = "latent", int(dim)
        if self.dim != math.prod(self.grid_shape):
            raise ValueError("latent dimension must equal the native grid size")
        height, width, channels = self.grid_shape
        self.conv0 = nnx.Conv(channels, 64, (3, 3), padding="SAME", rngs=rngs, **P.MP)
        self.conv1 = nnx.Conv(64, 64, (3, 3), padding="SAME", rngs=rngs, **P.MP)
        self.fc = nnx.Linear(height * width * 64, P.PEMB, rngs=rngs, **P.MP)
        self.gru = P.GRU(P.PEMB + P.A_N, P.PHID, rngs)
        self.l1 = nnx.Linear(P.PEMB + P.PHID, 256, rngs=rngs, **P.FP)
        self.actor = nnx.Linear(256, P.A_N, kernel_init=nnx.initializers.normal(0.01),
                                rngs=rngs, **P.FP)
        self.critic = nnx.Linear(256, P.VBINS, rngs=rngs, **P.FP)

    def embed(self, obs):
        if obs.ndim != 2 or obs.shape[-1] != self.dim:
            raise ValueError(f"expected flat native latents with shape (batch, {self.dim})")
        x = obs.reshape((obs.shape[0],) + self.grid_shape).astype(P.CDT)
        x = nnx.relu(self.conv0(x))
        x = nnx.relu(self.conv1(x))
        return nnx.relu(self.fc(x.reshape((x.shape[0], -1)))).astype(jnp.float32)


def make_ac(rngs, obs_dim, torso="mlp", grid_shape=NATIVE_GRID):
    """Construct a policy without changing the original MLP initialization path."""
    if torso == "mlp":
        return P.AC(rngs, "latent", int(obs_dim))
    if torso == "native_grid":
        return NativeGridAC(rngs, int(obs_dim), grid_shape)
    raise ValueError(f"unknown torso {torso!r}; choose one of {TORSOS}")


def policy_spec(ac):
    """Record enough architecture information to reconstruct either actor."""
    if isinstance(ac, NativeGridAC):
        return {"torso": "native_grid", "obs": ac.obs, "dim": ac.dim,
                "grid_shape": list(ac.grid_shape)}
    if type(ac) is P.AC:
        # Preserve compatibility with original pixel/flat-pixel checkpoints too.
        return {"torso": "mlp" if ac.obs != "pixels" else "pixels",
                "obs": ac.obs, "dim": ac.dim}
    raise TypeError(f"unsupported actor-critic class {type(ac).__name__}")


def save_ac(ac, path, meta):
    """Save parameters, metadata and the constructor; compatible with old MLPs."""
    spec = policy_spec(ac)
    payload = {"state": jax.device_get(nnx.split(ac)[1]), "meta": meta,
               "obs": ac.obs, "dim": ac.dim, "policy_spec": spec}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_ac(path):
    """Restore this module's checkpoints or original ``P.save_ac`` checkpoints."""
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    spec = payload.get("policy_spec")
    if spec is None:
        ac = P.AC(nnx.Rngs(0), payload.get("obs", "pixels"),
                  payload.get("dim", math.prod(NATIVE_GRID)))
    elif spec["torso"] == "native_grid":
        if spec["obs"] != "latent":
            raise ValueError("native-grid policies require latent observations")
        ac = NativeGridAC(nnx.Rngs(0), spec["dim"], spec["grid_shape"])
    elif spec["torso"] in ("mlp", "pixels"):
        ac = P.AC(nnx.Rngs(0), spec["obs"], spec["dim"])
    else:
        raise ValueError(f"unsupported saved torso {spec['torso']!r}")
    nnx.update(ac, jax.tree.map(jnp.asarray, payload["state"]))
    return ac, payload["meta"]
