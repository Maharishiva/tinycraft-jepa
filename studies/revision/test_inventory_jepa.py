"""CPU smoke: python -m unittest studies.revision.test_inventory_jepa -v."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import train as T
from studies.revision import inventory_jepa as A
from studies.revision import inventory_jepa_probe as P


class FakeWM:
    """Integer-valued dynamics make outgoing actions and current rewards visible."""

    win = 8

    def __init__(self, path=None):
        pass

    def encode(self, pixels):
        return jnp.broadcast_to(pixels[..., 0, 0, 0, None].astype(jnp.float32), pixels.shape[:2] + (512,))

    def predict(self, history, outgoing):
        increment = (outgoing * jnp.arange(1, 9)).sum(axis=1)
        return history[:, -1] + increment[:, None]

    def read(self, current, action):
        return current[:, 0] * 100 + action, (action == 16).astype(jnp.float32)

    def decode(self, latent):
        return jnp.broadcast_to(latent[:, 0, None, None, None].astype(jnp.uint8), (len(latent), 63, 63, 3))


class InventoryAdapterTests(unittest.TestCase):
    def test_history_and_pretransition_reward(self):
        state = {"seq": jnp.broadcast_to(jnp.arange(8)[None, :, None], (1, 8, 512)).astype(jnp.float32),
                 "aseq": jnp.asarray([[0, 1, 2, 3, 4, 5, 6, 0]], jnp.int32), "t": jnp.zeros(1, jnp.int32)}
        step = jax.jit(lambda s, a: A.transition(FakeWM(), s, a, jax.random.PRNGKey(1), 1024))
        obs, after, reward, done, info = step(state, jnp.asarray([3], jnp.int32))
        outgoing = np.asarray([0, 1, 2, 3, 4, 5, 6, 3])
        expected_next = 7 + int(outgoing @ np.arange(1, 9))
        np.testing.assert_array_equal(np.asarray(reward), [703])
        np.testing.assert_array_equal(np.asarray(after["aseq"]), [[1, 2, 3, 4, 5, 6, 3, 0]])
        np.testing.assert_array_equal(np.asarray(after["seq"])[:, :-1], np.asarray(state["seq"])[:, 1:])
        np.testing.assert_array_equal(np.asarray(after["seq"])[:, -1], np.full((1, 512), expected_next))
        np.testing.assert_allclose(obs, expected_next / 255.0)
        self.assertFalse(bool(done[0]))
        self.assertEqual(info, {})
        _, twice, reward2, _, _ = step(after, jnp.asarray([2], jnp.int32))
        np.testing.assert_array_equal(np.asarray(reward2), [100 * expected_next + 2])
        np.testing.assert_array_equal(np.asarray(twice["aseq"]), [[2, 3, 4, 5, 6, 3, 2, 0]])
        # Neither step mutates the prior action or latent history.
        np.testing.assert_array_equal(np.asarray(state["aseq"]), [[0, 1, 2, 3, 4, 5, 6, 0]])

    def test_native_done_and_episode_cap(self):
        state = {"seq": jnp.zeros((2, 8, 512)), "aseq": jnp.zeros((2, 8), jnp.int32),
                 "t": jnp.asarray([0, 1023], jnp.int32)}
        _, after, reward, done, _ = A.transition(FakeWM(), state, jnp.asarray([16, 1]), jax.random.PRNGKey(4), 1024)
        np.testing.assert_array_equal(np.asarray(done), [True, True])
        np.testing.assert_array_equal(np.asarray(after["t"]), [1, 1024])
        np.testing.assert_array_equal(np.asarray(reward), [16, 1])

    def test_reset_indices_and_repeated_frame_padding(self):
        frames = np.stack([np.full((63, 63, 3), v, np.uint8) for v in (10, 20, 30)])
        with patch.object(A, "WorldModel", FakeWM):
            _, reset, step = A.make_env("unused", reset_frames_u8=frames)
        keys = jax.random.split(jax.random.PRNGKey(6), 2)
        obs, state = jax.jit(reset)(keys, jnp.asarray([2, 0]))
        np.testing.assert_allclose(obs[:, 0, 0, 0], np.asarray([30, 10]) / 255.0)
        np.testing.assert_array_equal(state["seq"][:, :, 0], np.asarray([[30] * 8, [10] * 8]))
        np.testing.assert_array_equal(state["aseq"], np.zeros((2, 8)))
        default_obs, _ = reset(keys)
        ids = jax.random.randint(keys[0], (2,), 0, 3)
        np.testing.assert_allclose(default_obs, frames[np.asarray(ids)].astype(np.float32) / 255.0)
        # Actual JAX scan tracing through the wrapper is supported.
        def body(carry, unused):
            _, new_state, _, _, _ = step(keys, carry, jnp.asarray([1, 2], jnp.int32))
            return new_state, new_state["t"]
        end, _ = jax.jit(lambda s: jax.lax.scan(body, s, None, length=2))(state)
        np.testing.assert_array_equal(end["t"], [2, 2])

    def test_episode_start_convention(self):
        dones = np.asarray([[0, 1], [1, 0], [1, 1]], np.float32)
        np.testing.assert_array_equal(A.episode_start_indices(dones), [[0, 0], [0, 1], [1, 1], [2, 0]])

    def test_genuine_bootstrap_context_alignment(self):
        values = np.asarray([np.arange(8), np.arange(8) + 10], np.uint8)
        contexts = np.broadcast_to(values[:, :, None, None, None], (2, 8, 63, 63, 3)).copy()
        actions = np.asarray([np.arange(7), np.arange(7) + 8], np.int32)
        with patch.object(A, "WorldModel", FakeWM):
            _, reset, step = A.make_env("unused", reset_contexts_u8=contexts, reset_actions=actions)
        keys = jax.random.split(jax.random.PRNGKey(7), 1)
        obs, state = reset(keys, jnp.asarray([1]))
        np.testing.assert_allclose(obs[:, 0, 0, 0], [17 / 255])
        np.testing.assert_array_equal(state["seq"][:, :, 0], [np.arange(8) + 10])
        np.testing.assert_array_equal(state["aseq"], [[8, 9, 10, 11, 12, 13, 14, 0]])
        _, new_state, reward, _, _ = jax.jit(step)(keys, state, jnp.asarray([1], jnp.int32))
        np.testing.assert_array_equal(reward, [1701])
        np.testing.assert_array_equal(new_state["aseq"], [[9, 10, 11, 12, 13, 14, 1, 0]])

    def test_t_plus_one_current_frame_alignment(self):
        frames = np.arange(4 * 2 * 63 * 63 * 3, dtype=np.int64).reshape(4, 2, 63, 63, 3).astype(np.uint8)
        actions = np.zeros((3, 2), np.int32)
        rewards, dones = np.arange(6).reshape(3, 2).astype(np.float32), np.zeros((3, 2), np.float32)
        aligned = P.current_frame_bank(frames, actions, rewards, dones, expected_count=6)
        np.testing.assert_array_equal(aligned, frames[:-1])
        self.assertTrue(np.shares_memory(frames, aligned))
        with self.assertRaisesRegex(ValueError, "T\\+1"):
            P.current_frame_bank(frames[:-1], actions, rewards, dones, expected_count=6)

    def test_refuses_reward_modification(self):
        with self.assertRaisesRegex(ValueError, "forbidden"):
            A.make_env("unused", once=True)

    def test_freeze_guard_detects_any_array_change(self):
        with tempfile.TemporaryDirectory() as folder:
            source, fitted = Path(folder) / "source.npz", Path(folder) / "fitted.npz"
            arrays = {"model.p.0": np.asarray([1.0, -0.0], np.float32), "model.b.0": np.asarray([0.5], np.float32)}
            np.savez(source, **arrays)
            np.savez(fitted, **arrays, **{"decoder.p.0": np.ones(1)})
            A.assert_frozen_archive(source, fitted)
            arrays["model.b.0"] = np.asarray([0.6], np.float32)
            np.savez(fitted, **arrays)
            with self.assertRaisesRegex(ValueError, "changed"):
                A.assert_frozen_archive(source, fitted)

    def test_reward_done_head_roundtrip_and_frozen_features(self):
        z = jnp.arange(16 * 512, dtype=jnp.float32).reshape(16, 512) / 10000
        a = jnp.arange(16, dtype=jnp.int32)
        rewards = jnp.linspace(-0.1, 1.0, 16)
        dones = (a == 15).astype(jnp.float32)
        features_before = np.asarray(z).copy()
        heads, history = P.train_heads(z, a, rewards, dones, steps=4, batch=8, hidden=16, block=4)
        self.assertEqual(history.shape, (4, 2))
        self.assertTrue(np.isfinite(history).all())
        np.testing.assert_array_equal(features_before, np.asarray(z))
        self.assertFalse(hasattr(heads, "value"))
        arrays = {}
        for kind, state in zip(("p", "b"), T._weights(heads)):
            arrays.update({f"inventory_heads.{kind}.{i}": np.asarray(leaf)
                           for i, leaf in enumerate(jax.tree.leaves(state))})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "heads.npz"
            np.savez(path, **arrays)
            with np.load(path) as archive:
                restored = T._restore(A.TransitionHeads(512, 16, T.rngs(99)), archive, "inventory_heads")
            self.assertEqual(A.weights_digest(heads), A.weights_digest(restored))
            expected = heads(z, a)
            actual = restored(z, a)
            for x, y in zip(expected, actual):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    def test_frame_cache_preserves_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "archive.npz"
            frames = np.full((3, 2, 63, 63, 3), 17, np.uint8)
            np.savez_compressed(path, frames=frames)
            digest = A.file_sha256(path)
            first = P.materialize_frames(path, Path(folder) / "cache", digest)
            second = P.materialize_frames(path, Path(folder) / "cache", digest)
            self.assertIsInstance(first, np.memmap)
            np.testing.assert_array_equal(first, frames)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(digest, A.file_sha256(path))

    def test_real_nnx_bundle_nested_collector_and_freeze(self):
        # Small widths keep CPU smoke cheap; latent512/history8 and every module
        # boundary are unchanged. This fixture is never an experiment artifact.
        cfg = T.Config(encoder="tile", predictor="factored", emb_dim=8,
                       enc_depth=1, enc_heads=1, enc_res=1, pred_depth=1,
                       pred_heads=1, pred_width=8, tokens=16, token_dim=32,
                       token_hidden=16, history=8)
        model = T.LeWM(cfg, T.rngs(2))
        decoder = P.P.Decoder(512, 8, 1, T.rngs(3), tokens=16)
        heads = A.TransitionHeads(512, 8, T.rngs(4))
        meta = {"cfg": asdict(cfg), "data_sha256": A.EXPECTED_DATA_SHA256,
                "decoder": {"hidden": 8, "depth": 1, "tokens": 16},
                "inventory_heads": {"hidden": 8},
                "inventory_probe": {"format": A.FORMAT,
                    "frozen_model_weights_sha256": A.weights_digest(model)}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.npz"
            P.save_bundle(path, model, decoder, heads, meta)
            frames = np.full((2, 63, 63, 3), 127, np.uint8)
            wm, reset, step = A.make_env(path, reset_frames_u8=frames, chunk=2)
            before = wm.parameter_sha256()
            keys = jax.random.split(jax.random.PRNGKey(1), 2)
            _, initial = reset(keys)
            actor = nnx.Linear(1, 17, rngs=T.rngs(5))

            @nnx.jit
            def collect(actor, state):
                def body(carry, unused):
                    logits = actor(jnp.ones((2, 1)))
                    actions = jnp.argmax(logits, axis=-1).astype(jnp.int32)
                    obs, nxt, reward, done, _ = step(keys, carry, actions)
                    return nxt, (obs, reward, done)
                return jax.lax.scan(body, state, None, length=2)

            final, (obs, reward, done) = collect(actor, initial)
            np.testing.assert_array_equal(np.asarray(final["t"]), [2, 2])
            self.assertEqual(obs.shape, (2, 2, 63, 63, 3))
            self.assertTrue(np.isfinite(np.asarray(obs)).all())
            self.assertTrue(np.isfinite(np.asarray(reward)).all())
            self.assertEqual(done.dtype, jnp.bool_)
            self.assertEqual(before, wm.parameter_sha256())


if __name__ == "__main__":
    unittest.main()
