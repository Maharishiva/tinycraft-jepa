"""CPU-only regression tests; no JAX, Craftax, policy import or GPU allocation.

python -m unittest discover -s studies/revision -p test_inventory_eval.py -v
"""
import copy
import unittest

import numpy as np

from studies.revision.inventory_eval import ACHIEVEMENTS, INVENTORY, TOOLS, aggregate_traces, freeze_inactive


def synthetic(achievements, health, done, *, poison_reward=None, lava=None):
    """Construct raw traces with the exact pre-step freezing helper used live."""
    n, steps, _ = achievements.shape
    if lava is None:
        lava = np.zeros((n, steps), bool)
    initial = {"achievements": np.zeros((n, 22), bool), "health": np.full(n, 9),
               "inventory": np.zeros((n, 12), int)}
    old_ach = initial["achievements"].copy()
    old_health, old_lava = initial["health"].copy(), np.zeros(n, bool)
    active = np.ones(n, bool)
    trace = {k: [] for k in ("actions", "rewards", "native_done", "active_before", "alive_after", "achievements", "health", "in_lava", "policy_logits_finite")}
    score = np.full(n, np.nan)
    for t in range(steps):
        candidate = achievements[:, t]
        reward = candidate.sum(1) - old_ach.sum(1) + .1 * (health[:, t] - old_health)
        if poison_reward is not None:
            reward = np.where(active, reward, poison_reward)
        frozen = freeze_inactive(old_ach, candidate, active)
        h = freeze_inactive(old_health, health[:, t], active)
        l = freeze_inactive(old_lava, lava[:, t], active)
        d = active & done[:, t]
        score = np.where(d, np.expm1(np.log(101.) * frozen.mean(1)), score)
        values = dict(actions=np.where(active, 11, -1), rewards=np.where(active, reward, 0.),
                      native_done=d, active_before=active, alive_after=(h > 0) & ~l,
                      policy_logits_finite=np.ones(n, bool),
                      achievements=frozen, health=h, in_lava=l)
        for k, v in values.items():
            trace[k].append(v.copy())
        old_ach, old_health, old_lava = frozen, h, l
        active = active & ~done[:, t]
    raw = {k: np.stack(v, 1) for k, v in trace.items()}
    inventory = np.zeros((n, 12), int)
    for a, field in TOOLS.items():
        inventory[:, INVENTORY.index(field)] = old_ach[:, ACHIEVEMENTS.index(a)]
    endpoint = dict(achievements=old_ach, health=old_health, in_lava=old_lava,
                    timestep=raw["active_before"].sum(1), inventory=inventory,
                    native_terminal_info_score=score)
    return raw, initial, endpoint


class NativeInventoryEvaluationTests(unittest.TestCase):
    def test_terminal_acquisition_and_poisoned_post_terminal_steps(self):
        ach = np.zeros((1, 4, 22), bool)
        ach[0, :, 5] = True
        ach[0, 1:, 13] = True
        ach[0, 2:, :] = True  # ignored extra achievements after death
        raw, initial, end = synthetic(ach, np.array([[9, 0, 999, 999]]),
                                      np.array([[False, True, False, False]]), poison_reward=100.)
        rows, summary, first = aggregate_traces(raw, initial, end, horizon=4)
        self.assertEqual(rows[0]["executed_steps"], 2)
        self.assertEqual(first[0, 5], 1)
        self.assertEqual(first[0, 13], 2)
        self.assertEqual(first[0, 14], -1)
        self.assertAlmostEqual(rows[0]["native_return"], 1.1)
        self.assertFalse(rows[0]["survived_endpoint"])
        self.assertEqual(summary["four_tool_success"]["MAKE_STONE_PICKAXE"]["successes"], 1)
        np.testing.assert_array_equal(raw["actions"], [[11, 11, -1, -1]])

    def test_mixed_endings_and_terminal_at_cap_precedence(self):
        ach = np.zeros((3, 4, 22), bool)
        health = np.array([[9, 0, 0, 0], [9, 9, 9, 9], [9, 9, 9, 0]])
        done = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], bool)
        raw, initial, end = synthetic(ach, health, done)
        rows, _, _ = aggregate_traces(raw, initial, end, horizon=4)
        self.assertEqual([r["executed_steps"] for r in rows], [2, 4, 4])
        self.assertEqual([r["native_done"] for r in rows], [True, False, True])
        self.assertEqual([r["evaluator_truncated"] for r in rows], [False, True, False])
        self.assertIsNone(rows[1]["native_terminal_info_score"])

    def test_first_acquisition_is_one_based_and_persistent(self):
        ach = np.zeros((1, 5, 22), bool)
        ach[0, :, 0] = True
        ach[0, 2:, 1] = True
        ach[0, 4:, 5] = True
        args = synthetic(ach, np.array([[9, 8, 9, 8, 0]]), np.array([[0, 0, 0, 0, 1]], bool))
        rows, _, first = aggregate_traces(*args, horizon=5)
        np.testing.assert_array_equal(first[0, [0, 1, 5, 14]], [1, 3, 5, -1])
        self.assertIsNone(rows[0]["first_acquisition_step"]["MAKE_STONE_SWORD"])
        self.assertAlmostEqual(rows[0]["native_return"], 2.1)

    def test_reward_identity_rejects_corruption(self):
        args = synthetic(np.zeros((1, 3, 22), bool), np.array([[8, 9, 9]]), np.zeros((1, 3), bool))
        aggregate_traces(*args, horizon=3)
        args[0]["rewards"][0, 1] += .2
        with self.assertRaisesRegex(AssertionError, "telescoping"):
            aggregate_traces(*args, horizon=3)

    def test_native_actions_are_not_success_flags_and_invariants_reject(self):
        args = synthetic(np.zeros((1, 2, 22), bool), np.full((1, 2), 9), np.zeros((1, 2), bool))
        _, summary, _ = aggregate_traces(*args, horizon=2)
        self.assertEqual(summary["primary"]["successes"], 0)  # action 11 was attempted
        for field, index, value in (("actions", (0, 0), 17), ("rewards", (0, 1), 1.)):
            bad = copy.deepcopy(args)
            bad[0][field][index] = value
            with self.assertRaises(AssertionError):
                aggregate_traces(*bad, horizon=2)
        bad = copy.deepcopy(args)
        bad[1]["inventory"][0, 0] = 1
        with self.assertRaises(AssertionError):
            aggregate_traces(*bad, horizon=2)
        bad = copy.deepcopy(args)
        bad[2]["inventory"][0, INVENTORY.index("wood_pickaxe")] = 1
        with self.assertRaisesRegex(AssertionError, "MAKE_WOOD_PICKAXE"):
            aggregate_traces(*bad, horizon=2)

    def test_nonmonotone_active_flags_fail(self):
        ach = np.zeros((1, 2, 22), bool)
        ach[0, 0, 0] = True
        args = synthetic(ach, np.full((1, 2), 9), np.zeros((1, 2), bool))
        with self.assertRaisesRegex(AssertionError, "monotone"):
            aggregate_traces(*args, horizon=2)

    def test_score_aggregations_differ(self):
        ach = np.zeros((2, 1, 22), bool)
        ach[0, 0, 0] = ach[1, 0, 1] = True
        args = synthetic(ach, np.zeros((2, 1), int), np.ones((2, 1), bool))
        _, summary, _ = aggregate_traces(*args, horizon=1)
        self.assertAlmostEqual(summary["achievement_rate_geometric_aggregate"], .4296629167, places=8)
        self.assertAlmostEqual(summary["mean_native_terminal_info_score"], .2334044671, places=8)

    def test_fixed_denominator_four_tools_and_permutation(self):
        bits = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 1]], bool)
        ach = np.zeros((4, 1, 22), bool)
        ach[:, 0, [ACHIEVEMENTS.index(name) for name in TOOLS]] = bits
        args = synthetic(ach, np.array([[0], [9], [9], [0]]), np.array([[1], [0], [0], [1]], bool))
        _, summary, _ = aggregate_traces(*args, horizon=1)
        self.assertEqual([r["rate"] for r in summary["four_tool_success"].values()], [.5, .5, .25, .5])
        self.assertEqual(summary["four_tool_count"]["mean"], 1.75)
        self.assertEqual(summary["all_four_tools"]["rate"], .25)
        perm = np.array([2, 0, 3, 1])
        permuted = [{k: v[perm] for k, v in part.items()} for part in args]
        _, summary2, _ = aggregate_traces(*permuted, horizon=1)
        self.assertEqual(summary["four_tool_success"], summary2["four_tool_success"])
        self.assertEqual(summary["all_four_tools"], summary2["all_four_tools"])
        for field in ("native_return", "executed_steps", "achievement_count", "four_tool_count"):
            for moment in ("mean", "sd", "sem"):
                self.assertAlmostEqual(summary[field][moment], summary2[field][moment])


if __name__ == "__main__":
    unittest.main()
