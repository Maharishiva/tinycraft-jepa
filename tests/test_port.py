"""Checks that the JAX port matches the reference LeWM semantics."""
import sys
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train as T


def reference_sigreg(proj, directions, knots=17):
    """Numpy transcription of `module.SIGReg.forward` from lucas-maes/le-wm."""
    t = np.linspace(0, 3, knots, dtype=np.float64)
    dt = 3 / (knots - 1)
    weights = np.full((knots,), 2 * dt)
    weights[[0, -1]] = dt
    window = np.exp(-(t ** 2) / 2.0)
    weights = weights * window
    a = directions / np.linalg.norm(directions, axis=0, keepdims=True)
    x_t = (proj @ a)[..., None] * t
    err = (np.cos(x_t).mean(-3) - window) ** 2 + (np.sin(x_t).mean(-3)) ** 2
    return float(((err @ weights) * proj.shape[-2]).mean())


def test_sigreg_matches_reference():
    key = jax.random.PRNGKey(0)
    proj = np.asarray(jax.random.normal(jax.random.PRNGKey(1), (4, 64, 16)), np.float64)
    directions = np.asarray(jax.random.normal(key, (16, 32)), np.float64)
    ours = float(T.sigreg(jnp.asarray(proj, jnp.float32), key, 17, 32))
    theirs = reference_sigreg(proj, directions)
    assert abs(ours - theirs) < 1e-4 * max(1.0, abs(theirs)), (ours, theirs)


def test_sigreg_detects_collapse():
    key = jax.random.PRNGKey(2)
    gaussian = jax.random.normal(jax.random.PRNGKey(3), (4, 512, 16))
    statistic = lambda x: float(T.sigreg(x, key, 17, 64))
    assert statistic(gaussian) < statistic(gaussian * 0.05)
    assert statistic(gaussian) < statistic(jnp.zeros_like(gaussian))
    assert statistic(gaussian) < statistic(gaussian * 3.0)


def _tiny(seed=0, wake_adaln=False):
    cfg = T.Config(emb_dim=32, enc_depth=1, enc_mlp=32, pred_depth=2, pred_mlp=32,
                   mlp_hidden=32, pred_heads=2, pred_dim_head=8, dropout=0.0)
    model = T.LeWM(cfg, T.rngs(seed))
    if wake_adaln:
        for i, block in enumerate(model.predictor.blocks):
            block.ada.kernel[...] = 0.2 * jax.random.normal(jax.random.PRNGKey(50 + i),
                                                            block.ada.kernel.shape)
    return cfg, model


def test_predictor_is_zero_initialised():
    cfg, model = _tiny()
    emb = jax.random.normal(jax.random.PRNGKey(4), (2, cfg.history, cfg.emb_dim))
    act_emb = model.act_embed(jnp.zeros((2, cfg.history), jnp.int32))
    # AdaLN-zero: every block is the identity at initialisation.
    assert np.allclose(model.predictor(emb, act_emb),
                       model.predictor.norm(emb + model.predictor.pos[...]), atol=1e-4)


def test_predictor_is_causal():
    cfg, model = _tiny(wake_adaln=True)
    emb = jax.random.normal(jax.random.PRNGKey(4), (2, cfg.history, cfg.emb_dim))
    act_emb = model.act_embed(jnp.zeros((2, cfg.history), jnp.int32))
    base = model.predictor(emb, act_emb)
    noise = jax.random.normal(jax.random.PRNGKey(9), emb[:, -1].shape)
    after = model.predictor(emb.at[:, -1].set(3.0 * noise), act_emb)
    assert np.allclose(base[:, :-1], after[:, :-1], atol=1e-4)
    assert not np.allclose(base[:, -1], after[:, -1], atol=1e-4)


def test_actions_condition_the_predictor():
    cfg, model = _tiny(wake_adaln=True)
    emb = jax.random.normal(jax.random.PRNGKey(6), (2, cfg.history, cfg.emb_dim))
    one = model.predictor(emb, model.act_embed(jnp.zeros((2, cfg.history), jnp.int32)))
    two = model.predictor(emb, model.act_embed(jnp.full((2, cfg.history), 5, jnp.int32)))
    assert not np.allclose(one, two, atol=1e-4)


def test_outgoing_action_alignment():
    context = np.arange(1, 9, dtype=np.int32)[None]
    future = np.arange(11, 27, dtype=np.int32)[None]
    out = np.asarray(T.outgoing_actions(context, future))[0]
    assert out.shape == (24,)
    # frame k is left with out[k]; eval.py hands over incoming actions, so out[7] must be the
    # first future action (the one that produces the first generated frame).
    assert out[0] == context[0, 1] and out[6] == context[0, 7]
    assert out[7] == future[0, 0] and out[22] == future[0, 15]


def test_window_starts_skip_resets():
    dones = np.zeros((10, 2), np.int64)
    dones[4, 0] = 1
    table = T.valid_starts(dones, span=4)
    starts = {(int(s), int(e)) for s, e in table}
    assert (2, 0) not in starts and (4, 0) not in starts
    assert (5, 0) in starts and (4, 1) in starts


def test_rollout_uses_history_window_only():
    cfg = T.Config(emb_dim=32, enc_depth=1, enc_mlp=32, pred_depth=1, pred_mlp=32,
                   mlp_hidden=32, pred_heads=2, pred_dim_head=8, dropout=0.0)
    model = T.LeWM(cfg, T.rngs(0))
    frames = np.asarray(jax.random.randint(jax.random.PRNGKey(7), (2, 8, 63, 63, 3), 0, 255), np.uint8)
    context = np.zeros((2, 8), np.int32)
    future = np.zeros((2, 4), np.int32)
    emb, preds = T._rollout_latents(model, jnp.asarray(frames), T.outgoing_actions(context, future),
                                    horizon=4)
    assert emb.shape == (2, 8, cfg.emb_dim) and preds.shape == (2, 4, cfg.emb_dim)
    assert np.isfinite(np.asarray(preds)).all()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error}")
    raise SystemExit(failures)
