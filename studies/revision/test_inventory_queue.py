"""CPU-only semantic guards for the prospective fixed policy-training matrix."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from studies.revision import inventory_common as C
from studies.revision import run_inventory_queue as Q

ROOT = Path(__file__).parent


class RecipeTests(unittest.TestCase):
    def test_original_policy_source_is_exact_snapshot(self):
        manifest = json.loads((ROOT / 'inventory_source/SOURCE_MANIFEST.json').read_text())
        for filename, entry in manifest.items():
            self.assertEqual(C.sha256(ROOT / 'inventory_source' / filename), entry['sha256'], filename)
        self.assertEqual(manifest['agent.py']['sha256'],
                         '1bfeb69f0a6098959a409c4601e8a92219b1967f297b1e35beb96440edceb132')

    def test_original_ppo_core_byte_exact(self):
        old = (ROOT / 'inventory_source/train_ppo_real.py').read_text()
        new = (ROOT / 'inventory_ppo.py').read_text()
        def core(source):
            return source[source.index('    @nnx.jit\n    def collect'):source.index('    rng = jax.random.PRNGKey')]
        self.assertEqual(core(old), core(new))

    def test_exact_six_prescribed_runs(self):
        args = SimpleNamespace(flow_wm=Path('/flow.npz'), jepa_wm=Path('/jepa.npz'),
                               jepa_sha256='j' * 64, out=Path('/runs'), reset_bank=Path('/bank.npz'),
                               reset_bank_sha256='b' * 64)
        jobs = Q.jobs_for(args)
        self.assertEqual([(x['family'], x['policy_seed']) for x in jobs],
                         [(f, s) for s in (0, 1, 2) for f in ('flow', 'jepa')])
        for job in jobs:
            self.assertEqual(job['model_seed'], 0)
            self.assertNotIn('--ac_from', job['train_command'])
            self.assertNotIn('--wm_once', job['train_command'])
            self.assertEqual(job['eval_command'][-2:], ['--eval-seed', '123'])
            self.assertTrue(job['eval_command'][job['eval_command'].index('--checkpoint') + 1].endswith('/model.pkl'))

    def test_actual_budget(self):
        self.assertEqual(5_000_000 // (64 * 100), 781)
        self.assertEqual(781 * 64 * 100, 4_998_400)

    def test_timezone_required(self):
        with self.assertRaises(ValueError):
            Q.parse_deadline('2026-09-20T07:41:00')
        self.assertEqual(Q.parse_deadline('2026-09-20T07:41:00Z'),
                         Q.parse_deadline('2026-09-20T09:41:00+02:00'))

    def test_unreviewed_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'admission.json'
            path.write_text(json.dumps({'source_review_approved': False}))
            with self.assertRaisesRegex(ValueError, 'independent source review'):
                Q.validate_admission(path, SimpleNamespace())
            path.write_text(json.dumps({'source_review_approved': True, 'source_hashes': {}}))
            with self.assertRaisesRegex(ValueError, 'Source bytes differ'):
                Q.validate_admission(path, SimpleNamespace())

    def test_reference_reset_bank_order(self):
        import numpy as np
        done = np.array([[False, True], [True, False], [False, False]])
        t, e = np.nonzero(np.concatenate([np.ones((1, done.shape[1]), bool), done[:-1] > .5], 0))
        self.assertEqual(list(zip(t.tolist(), e.tolist())), [(0, 0), (0, 1), (1, 1), (2, 0)])

    def test_genuine_prefix_bank_excludes_only_length_or_reset_crossing(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            data, bank = Path(tmp) / 'data.npz', Path(tmp) / 'bank.npz'
            done = np.zeros((10, 2), bool)
            done[2, 0] = done[8, 1] = True
            frames = np.broadcast_to(np.arange(22, dtype=np.uint8).reshape(11, 2, 1, 1, 1),
                                     (11, 2, 63, 63, 3)).copy()
            actions = (np.arange(20).reshape(10, 2) % 17).astype(np.int32)
            # Positive metadata must never exclude or upweight a start.
            inventory = np.full((11, 2, 12), 7, np.int32)
            np.savez(data, frames=frames, actions=actions, dones=done,
                     rewards=np.ones((10, 2), np.float32), inventory=inventory)
            original_hash = C.sha256
            with patch.object(C, 'sha256', side_effect=lambda p: C.DATA_SHA256 if Path(p) == data else original_hash(p)):
                C.prepare_bank(data, bank)
            with np.load(bank, allow_pickle=False) as saved:
                self.assertEqual(saved['episode_start_time_index'].tolist(), [0, 3])
                self.assertEqual(saved['env_index'].tolist(), [1, 0])
                np.testing.assert_array_equal(saved['actions'], np.stack([actions[:7, 1], actions[3:10, 0]]))
                np.testing.assert_array_equal(saved['contexts'][:, :, 0, 0, 0], [[1, 3, 5, 7, 9, 11, 13, 15],
                                                                             [6, 8, 10, 12, 14, 16, 18, 20]])
                self.assertTrue(np.all(saved['descriptive_initial_inventory'] == 7))
            metadata = json.loads(bank.with_suffix('.json').read_text())
            self.assertEqual(metadata['candidate_episode_starts'], 4)
            self.assertEqual(metadata['excluded_incomplete'], 1)
            self.assertEqual(metadata['excluded_reset_crossing'], 1)
            self.assertFalse(metadata['outcome_filtering'])


if __name__ == '__main__':
    unittest.main()
