"""Independent CPU checks for the exact one-step wind/variance diagnostic.

Run with ``python -m unittest studies.revision.test_oracle_variance -v``.
The evaluator must keep its numerical primitives importable without JAX/GPU
initialization. These tests neither load a checkpoint nor fit any model.
"""
import itertools
import unittest

import numpy as np

from studies.revision import oracle_variance as O


MOVES = np.asarray([[-3, 0], [0, -3], [3, 0], [0, 3]], dtype=np.int64)


class BranchLawTests(unittest.TestCase):
    def test_all_wind_states_have_correct_mass_mean_and_variance(self):
        wind = np.asarray(list(itertools.product(range(-2, 3), repeat=2)))
        pos = np.full_like(wind, 28)
        next_wind, next_pos, weights = O.branch_law(wind, pos, MOVES)

        self.assertEqual(next_wind.shape, (25, 9, 2))
        self.assertEqual(next_pos.shape, (25, 4, 9, 2))
        self.assertEqual(weights.shape, (9,))
        np.testing.assert_allclose(weights.sum(), 1.0, rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            np.sort(weights), [0.01] * 4 + [0.08] * 4 + [0.64],
            rtol=0, atol=1e-14,
        )
        self.assertTrue(np.all(weights > 0))
        self.assertTrue(np.all(np.abs(next_wind) <= 2))

        mean = np.einsum("k,nkd->nd", weights, next_wind)
        expected_mean = np.where(np.abs(wind) == 2, 0.95 * wind, wind)
        np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=1e-13)
        centered = next_wind - mean[:, None]
        variance = np.einsum("k,nkd->nd", weights, centered ** 2)
        expected_variance = np.where(np.abs(wind) == 2, 0.09, 0.20)
        np.testing.assert_allclose(variance, expected_variance, rtol=0, atol=1e-13)
        cross_covariance = np.einsum(
            "k,nk,nk->n", weights, centered[:, :, 0], centered[:, :, 1]
        )
        np.testing.assert_allclose(cross_covariance, 0.0, rtol=0, atol=1e-13)
        expected_pos = pos[:, None, None] + MOVES[None, :, None] + next_wind[:, None]
        np.testing.assert_array_equal(next_pos, expected_pos)

    def test_clipping_preserves_mass_of_duplicate_branches(self):
        next_wind, _, weights = O.branch_law(
            np.asarray([[2, 2]]), np.asarray([[28, 28]]), MOVES
        )
        mass = {}
        for outcome, probability in zip(next_wind[0], weights):
            key = tuple(int(x) for x in outcome)
            mass[key] = mass.get(key, 0.0) + float(probability)
        self.assertEqual(set(mass), {(1, 1), (1, 2), (2, 1), (2, 2)})
        for outcome, expected in {
            (1, 1): 0.01, (1, 2): 0.09, (2, 1): 0.09, (2, 2): 0.81
        }.items():
            self.assertAlmostEqual(mass[outcome], expected, places=13)

    def test_wind_inference_uses_previous_outgoing_action(self):
        actions = np.asarray([[0, 1, 2, 3, 0, 2, 1], [2, 3, 0, 1, 2, 0, 3]])
        winds = np.asarray([
            [[0, 0], [1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]],
            [[0, 0], [0, -1], [-1, -1], [-1, 0], [-2, 1], [-1, 1], [-1, 0]],
        ])
        positions = np.empty((2, 8, 2), dtype=np.int64)
        positions[:, 0] = 28
        for step in range(7):
            positions[:, step + 1] = positions[:, step] + MOVES[actions[:, step]] + winds[:, step]
        # The input is a genuine within-episode context. Reset detection is the
        # caller's responsibility; positions alone do not reliably identify it.
        self.assertTrue(np.all((positions >= 2) & (positions <= 55)))
        np.testing.assert_array_equal(
            O.infer_wind(positions, actions, MOVES), winds[:, -1]
        )

    def test_safety_checks_every_branch_and_episode_cap(self):
        positions = np.full((6, 4, 9, 2), 28, dtype=np.int64)
        positions[0] = [2, 55]  # The bounds are inclusive.
        positions[1, 3, 8, 0] = 1
        positions[2, 0, 0, 1] = 56
        steps = np.asarray([126, 20, 20, 127, 128, 0])
        np.testing.assert_array_equal(
            O.safe_all_branches(positions, steps),
            [True, False, False, False, False, True],
        )


class CostIdentityTests(unittest.TestCase):
    def assert_identity(self, value, atol=1e-11):
        np.testing.assert_allclose(
            value["exact"], value["mean"] + value["variance"][:, :, None],
            rtol=1e-12, atol=atol,
        )
        self.assertLessEqual(float(np.max(np.abs(value["identity_residual"]))), atol)

    def test_two_branch_analytic_misranking(self):
        # Action A has outcomes -1/+1; action B is always 0.1. The goal is 0.
        branch = np.asarray([[[[-1.0], [1.0]], [[0.1], [0.1]]]])
        goal = np.zeros((1, 1, 1))
        result = O.costs(branch, goal, np.asarray([0.5, 0.5]))
        np.testing.assert_allclose(result["mean"][0, :, 0], [0.0, 0.01], atol=1e-14)
        np.testing.assert_allclose(result["variance"][0], [1.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(result["exact"][0, :, 0], [1.0, 0.01], atol=1e-14)
        self.assert_identity(result)
        decision = O.decision_metrics(result["mean"], result["exact"])
        self.assertTrue(decision["robust_disagreement"][0, 0])
        for name in ("regret_min", "regret_max", "regret_uniform"):
            self.assertAlmostEqual(float(decision[name][0, 0]), 0.99, places=13)

    def test_iterating_conditional_mean_misses_nonlinear_recursion_mean(self):
        # x_{t+1}=x_t^2+xi_t, x_0=0, independent fair xi_t in {-1,+1}.
        innovations = np.asarray(list(itertools.product([-1.0, 1.0], repeat=2)))
        probability = np.full(4, 0.25)
        x1 = innovations[:, 0]
        x2 = x1 ** 2 + innovations[:, 1]
        self.assertEqual(float(probability @ x1), 0.0)
        self.assertEqual(float(probability @ x2), 1.0)
        iterated_mean = 0.0
        for _ in range(2):
            iterated_mean = iterated_mean ** 2
        self.assertEqual(iterated_mean, 0.0)
        self.assertNotEqual(iterated_mean, float(probability @ x2))

    def test_centered_variance_survives_large_common_offset(self):
        offset = 1e8
        branch = np.asarray([[[[offset - 1], [offset + 1]],
                              [[offset + 0.1], [offset + 0.1]]]], dtype=np.float64)
        goal = np.full((1, 1, 1), offset, dtype=np.float64)
        result = O.costs(branch, goal, np.asarray([0.5, 0.5]))
        np.testing.assert_allclose(result["variance"], [[1.0, 0.0]], rtol=0, atol=1e-14)
        represented_bias_cost = (float(offset + 0.1) - offset) ** 2
        np.testing.assert_allclose(
            result["mean"][0, :, 0], [0.0, represented_bias_cost],
            rtol=1e-12, atol=1e-14,
        )
        self.assert_identity(result)

    def test_identity_position_is_an_action_constant_variance_control(self):
        wind = np.asarray(list(itertools.product(range(-2, 3), repeat=2)))
        pos = np.full_like(wind, 28)
        _, branch, weights = O.branch_law(wind, pos, MOVES)
        goal_positions = np.asarray(list(itertools.product([16, 28, 42], repeat=2)))
        goals = np.broadcast_to(goal_positions[None], (25, 9, 2))
        result = O.costs(branch, goals, weights)
        # costs uses coordinate-mean squared distance, hence division by D=2.
        expected = np.where(np.abs(wind) == 2, 0.09, 0.20).sum(axis=1) / 2
        np.testing.assert_allclose(
            result["variance"], np.broadcast_to(expected[:, None], (25, 4)),
            rtol=0, atol=1e-12,
        )
        self.assert_identity(result)
        decision = O.decision_metrics(result["mean"], result["exact"])
        np.testing.assert_array_equal(decision["mean_optimal"], decision["exact_optimal"])
        self.assertFalse(np.any(decision["robust_disagreement"]))
        for name in ("regret_min", "regret_max", "regret_uniform"):
            np.testing.assert_allclose(decision[name], 0.0, rtol=0, atol=1e-10)

    def test_affine_position_encoder_preserves_action_constant_variance(self):
        wind = np.asarray([[-2, -2], [0, 0], [2, -1]])
        pos = np.asarray([[24, 28], [28, 28], [32, 24]])
        _, branch, weights = O.branch_law(wind, pos, MOVES)
        matrix = np.asarray([[1.0, 0.5, 0.0], [2.0, -1.0, 3.0]])
        offset = np.asarray([11.0, -7.0, 4.0])
        goals = np.broadcast_to(np.asarray([[16, 16], [28, 28], [42, 42]])[None], (3, 3, 2))
        result = O.costs(branch @ matrix + offset, goals @ matrix + offset, weights)
        np.testing.assert_allclose(
            result["variance"], np.broadcast_to(result["variance"][:, :1], (3, 4)),
            rtol=0, atol=1e-11,
        )
        self.assert_identity(result)
        decision = O.decision_metrics(result["mean"], result["exact"])
        np.testing.assert_array_equal(decision["mean_optimal"], decision["exact_optimal"])
        self.assertFalse(np.any(decision["robust_disagreement"]))
        for name in ("regret_min", "regret_max", "regret_uniform"):
            np.testing.assert_allclose(decision[name], 0.0, rtol=0, atol=1e-10)

    def test_realizable_nonlinear_encoder_can_change_action_ranking(self):
        # A legitimate encoded goal at physical position (30, 28). This uses
        # the actual four moves and nine-branch law, not an off-manifold goal.
        _, positions, weights = O.branch_law(np.asarray([[0, 0]]), np.asarray([[28, 28]]), MOVES)
        branch = np.exp(0.4 * (positions[..., 0] - 28))[..., None]
        goal = np.asarray([[[np.exp(0.8)]]])
        result = O.costs(branch, goal, weights)
        self.assert_identity(result)
        decision = O.decision_metrics(result["mean"], result["exact"])
        np.testing.assert_array_equal(np.flatnonzero(decision["mean_optimal"][0, :, 0]), [2])
        np.testing.assert_array_equal(np.flatnonzero(decision["exact_optimal"][0, :, 0]), [1, 3])
        self.assertTrue(decision["robust_disagreement"][0, 0])
        self.assertAlmostEqual(float(decision["regret_min"][0, 0]), 0.2051326677698868, places=11)

    def test_zero_innovation_has_no_variance_or_ranking_difference(self):
        values = np.asarray([[[1.0, -2.0], [3.0, 0.0], [-1.0, 4.0], [0.0, 0.0]]])
        branch = np.repeat(values[:, :, None], 9, axis=2)
        _, _, weights = O.branch_law(np.asarray([[0, 0]]), np.asarray([[28, 28]]), MOVES)
        goals = np.asarray([[[0.0, 0.0], [2.0, 1.0]]])
        result = O.costs(branch, goals, weights)
        self.assert_identity(result)
        np.testing.assert_allclose(result["variance"], 0.0, rtol=0, atol=1e-25)
        decision = O.decision_metrics(result["mean"], result["exact"])
        np.testing.assert_array_equal(decision["mean_optimal"], decision["exact_optimal"])
        self.assertFalse(np.any(decision["robust_disagreement"]))


class DecisionTests(unittest.TestCase):
    def test_ties_report_min_max_and_uniform_regret(self):
        mean = np.asarray([[[0.0, 0.0], [0.0, 0.0], [2.0, 2.0]]])
        exact = np.asarray([[[2.0, 3.0], [0.0, 2.0], [1.0, 0.0]]])
        result = O.decision_metrics(mean, exact)
        np.testing.assert_array_equal(result["mean_optimal"], [[[True, True], [True, True], [False, False]]])
        np.testing.assert_array_equal(result["exact_optimal"], [[[False, False], [True, False], [False, True]]])
        np.testing.assert_array_equal(result["robust_disagreement"], [[False, True]])
        np.testing.assert_allclose(result["regret_min"], [[0.0, 2.0]], rtol=0, atol=1e-14)
        np.testing.assert_allclose(result["regret_max"], [[2.0, 3.0]], rtol=0, atol=1e-14)
        np.testing.assert_allclose(result["regret_uniform"], [[1.0, 2.5]], rtol=0, atol=1e-14)

    def test_roundoff_argmin_flip_is_not_robust_disagreement(self):
        mean = np.asarray([[[0.0], [5e-11]]])
        exact = np.asarray([[[5e-11], [0.0]]])
        self.assertNotEqual(int(np.argmin(mean)), int(np.argmin(exact)))
        result = O.decision_metrics(mean, exact, tolerance_factor=1e-10)
        self.assertTrue(result["mean_optimal"].all())
        self.assertTrue(result["exact_optimal"].all())
        self.assertFalse(result["robust_disagreement"][0, 0])

    def test_tolerance_scales_with_cost_magnitude(self):
        mean = np.asarray([[[1e5], [1e5 + 5e-6]]])
        exact = np.asarray([[[1e5 + 5e-6], [1e5]]])
        result = O.decision_metrics(mean, exact, tolerance_factor=1e-10)
        self.assertTrue(result["mean_optimal"].all())
        self.assertTrue(result["exact_optimal"].all())
        self.assertFalse(result["robust_disagreement"][0, 0])

    def test_tie_metrics_do_not_depend_on_action_order(self):
        mean = np.asarray([[[0.0, 0.0], [0.0, 0.0], [2.0, 2.0]]])
        exact = np.asarray([[[2.0, 3.0], [0.0, 2.0], [1.0, 0.0]]])
        order = np.asarray([2, 0, 1])
        original = O.decision_metrics(mean, exact)
        permuted = O.decision_metrics(mean[:, order], exact[:, order])
        for name in ("mean_optimal", "exact_optimal"):
            np.testing.assert_array_equal(permuted[name], original[name][:, order])
        for name in ("robust_disagreement", "regret_min", "regret_max", "regret_uniform"):
            np.testing.assert_array_equal(permuted[name], original[name])


if __name__ == "__main__":
    unittest.main()
