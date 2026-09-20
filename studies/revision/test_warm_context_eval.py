"""CPU-only protocol tests; synthetic policies are never scientific results.

JAX_PLATFORMS=cpu python -m unittest studies.revision.test_warm_context_eval -v
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from studies.revision import warm_context_eval as W
from studies.stoch import env as S


class Identity:
    def posterior(self, frames):
        return frames

    def posterior_step(self, previous_real, current_real):
        return current_real


class TemporalTestCodec:
    """An exactly checkable temporal codec that changes every real frame."""
    def posterior(self, frames):
        previous = jnp.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
        return ((previous.astype(jnp.int32) + frames.astype(jnp.int32) + 13) % 256).astype(jnp.uint8)

    def posterior_step(self, previous_real, current_real):
        return ((previous_real.astype(jnp.int32) + current_real.astype(jnp.int32) + 13) % 256).astype(jnp.uint8)


class RecordLeft(nnx.Module):
    def step(self, frame, prev_a, g):
        increment = jnp.stack([jnp.ones_like(prev_a), prev_a,
                               frame[:, 0, 0, 0], frame[:, 3, 3, 0]], axis=-1)
        logits = jnp.broadcast_to(jnp.asarray([0., -1e10, -1e10, -1e10]), (len(frame), 4))
        return logits, jnp.zeros((len(frame), 1)), g + increment


class SafeTestPolicy(nnx.Module):
    """Only a test fixture: read cube position to guarantee the cap is reached."""
    def step(self, frame, prev_a, g):
        red = (frame[..., 0] == 255) & (frame[..., 1] == 0) & (frame[..., 2] == 0)
        # Ignore the outer red border; the cube always occupies the interior.
        red = red[:, 2:-2, 2:-2]
        x = (red * jnp.arange(2, S.W - 2)[None, None, :]).sum((1, 2)) / red.sum((1, 2))
        action = jnp.where(x < 30, 2, 0)
        logits = jnp.where(jax.nn.one_hot(action, 4) > 0, 0., -1e10)
        return logits, jnp.zeros((len(frame), 1)), g + 1


class WarmContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(d.platform != "cpu" for d in jax.devices()):
            raise RuntimeError("correctness tests must run with JAX_PLATFORMS=cpu")
        cls.packet = W.create_packet("gray_frame_still", episodes=4, candidate_batch=8)

    def test_packet_roundtrip_and_fixed_rng(self):
        other = W.create_packet("gray_frame_still", episodes=4, candidate_batch=3)
        self.assertEqual(self.packet.fingerprint(), other.fingerprint())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packet.npz"
            W.save_packet(self.packet, path)
            self.assertEqual(W.load_packet(path).fingerprint(), self.packet.fingerprint())
            with self.assertRaises(ValueError):
                W.load_or_create_packet(path, "gray_frame_wind", episodes=4)
        with self.assertRaises(ValueError):
            W.make_evaluator(S.CONFIGS["gray_frame_still"], Identity(), self.packet)

    def test_survivors_reach_native_128_not_135(self):
        result = W.make_evaluator(S.CONFIGS["gray_frame_still"], Identity(), self.packet,
                                  require_episodes=None, hidden_size=4)(SafeTestPolicy())
        result = jax.device_get(result)
        np.testing.assert_array_equal(result["control_length"], 121)
        np.testing.assert_array_equal(result["total_length"], 128)
        np.testing.assert_array_equal(result["final_state"].step, 128)
        np.testing.assert_array_equal(result["final_gru"], 128.)
        np.testing.assert_array_equal(result["return"], 0.)
        np.testing.assert_array_equal(result["done"], True)
        np.testing.assert_array_equal(result["survival"], True)

    def test_direct_replay_terminal_reward_burn_alignment_and_previous_real(self):
        packet, codec = self.packet, TemporalTestCodec()
        result = W.make_evaluator(S.CONFIGS["gray_frame_still"], codec, packet,
                                  require_episodes=None, hidden_size=4)(RecordLeft())
        result = jax.device_get(result)
        step = jax.jit(S.make(S.CONFIGS["gray_frame_still"])[1])
        contextual = np.asarray(codec.posterior(jnp.asarray(packet.raw_context)))
        for i in range(packet.episodes):
            # Independent scalar replay includes exactly the observed input
            # at each policy call, including terminal-producing action.
            state = jax.tree.map(lambda value: jnp.asarray(value[i]), packet.state)
            real = packet.raw_context[i, -1]
            decoded = contextual[i, -1]
            pa = int(packet.incoming_actions[i, -1])
            g = np.zeros(4, np.float32)
            for t in range(7):
                frame = contextual[i, t]
                g += [1, packet.incoming_actions[i, t], frame[0, 0, 0], frame[3, 3, 0]]
            total_reward = 0.
            for t in range(121):
                g += [1, pa, decoded[0, 0, 0], decoded[3, 3, 0]]
                nxt, state, reward, done = step(state, jnp.asarray(0), jnp.asarray(packet.control_transition_keys[i, t]))
                decoded = ((real.astype(np.int32) + np.asarray(nxt).astype(np.int32) + 13) % 256).astype(np.uint8)
                real, pa = np.asarray(nxt), 0
                total_reward += float(reward)
                if bool(done):
                    break
            self.assertEqual(total_reward, -1.)
            self.assertEqual(float(result["return"][i]), -1.)
            self.assertEqual(int(result["control_length"][i]), t + 1)
            self.assertEqual(int(result["total_length"][i]), 7 + t + 1)
            np.testing.assert_array_equal(result["final_gru"][i], g)
            for actual, expected in zip(result["final_state"], state):
                np.testing.assert_array_equal(actual[i], expected)
            if i < 3:
                np.testing.assert_array_equal(result["trace_control_actions"][i, t + 1:], -1)
                np.testing.assert_array_equal(result["trace_control_rewards"][i, t + 1:], 0.)
                np.testing.assert_array_equal(result["trace_control_active"][i, t + 1:], False)
                np.testing.assert_array_equal(result["trace_control_rgb"][i, t + 1:],
                                              np.broadcast_to(real, result["trace_control_rgb"][i, t + 1:].shape))

    def test_prefix_admission_never_tests_future_survival(self):
        # This fixture dies at step eight for every possible action. A valid
        # seven-action prefix must still be admitted without looking ahead.
        reset, _ = S.make(S.CONFIGS["gray_frame_still"])

        def death_at_eight(state, action, key):
            state = state._replace(step=state.step + 1)
            return S.render(S.CONFIGS["gray_frame_still"], state, key), state, \
                jnp.where(state.step >= 8, -1., 0.), state.step >= 8

        with patch.object(S, "make", return_value=(reset, death_at_eight)):
            packet = W.create_packet("gray_frame_still", episodes=5, candidate_batch=3)
        np.testing.assert_array_equal(packet.prefix_ids, np.arange(5))
        self.assertEqual(packet.metadata["total_attempts"], 5)
        self.assertEqual(packet.metadata["excluded_prefix_deaths"], 0)
        np.testing.assert_array_equal(packet.state.step, 7)

    def test_saved_traces_are_first_three_full_native_episodes(self):
        packet = self.packet
        result = W.make_evaluator(S.CONFIGS["gray_frame_still"], Identity(), packet,
                                  require_episodes=None, hidden_size=4)(RecordLeft())
        with tempfile.TemporaryDirectory() as directory:
            summary = W.save_evaluation(result, packet, directory, policy_seed=123)
            with np.load(Path(directory) / "traces.npz") as trace:
                self.assertEqual(trace["native_rgb"].shape, (3, 129, 63, 63, 3))
                self.assertEqual(trace["native_actions"].shape, (3, 128))
                np.testing.assert_array_equal(trace["prefix_ids"], packet.prefix_ids[:3])
                np.testing.assert_array_equal(trace["native_rgb"][:, :8], packet.raw_context[:3])
            with np.load(Path(directory) / "outcomes.npz") as outcomes:
                np.testing.assert_array_equal(outcomes["control_action_keys"], packet.control_action_keys)
            self.assertEqual(summary["training_policy_seed"], 123)
            self.assertEqual(summary["packet_fingerprint"], packet.fingerprint())


if __name__ == "__main__":
    unittest.main()
