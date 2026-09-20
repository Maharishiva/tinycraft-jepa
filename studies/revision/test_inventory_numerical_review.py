"""Rejection tests for the identified native BF16 numerical-review admission."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from studies.revision import inventory_numerical_review as N
from studies.revision.inventory_common import sha256

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts/revision'
BANK = 'f0d20ddb488a21a309a01db72efbd99360d304e38d98c972fc843571fd3073f5'


class NumericalAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ARTIFACTS / 'inventory_flow_numerical_review.json'
        if not path.exists():
            raise unittest.SkipTest('Specific archived GPU evidence is required for these rejection checks')
        cls.review = json.loads(path.read_text())

    def validate(self, review, bank=BANK):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'review.json'
            path.write_text(json.dumps(review))
            return N.validate_review(path, reset_bank_sha256=bank)

    def arrays(self):
        with np.load(self.review['canonical_arrays']['path'], allow_pickle=False) as archive:
            needed = ['seq','tau','aseq','slots','row_keys','prepared_actual_seq','prepared_actual_tau',
                      'prepared_actual_aseq','prepared_actual_slot','prepared_actual_keys','prepared_actual_real',
                      'pre_n','pre_r','lax_map','scalar','transition_dynamic_z','transition_closed_z',
                      'canonical_native','canonical_fixed_batch','canonical_fixed_vmap','canonical_fixed_rows']
            needed += [f'noise_{i}_{kind}' for i in range(3) for kind in ('scalar','vmap')]
            return {name: np.array(archive[name], copy=True) for name in needed}

    def test_identified_review_passes_without_changing_failed_flag(self):
        result = self.validate(self.review)
        self.assertFalse(result['original_strict_parity_passed'])
        failed = json.loads(Path(self.review['failed_receipt']['path']).read_text())
        self.assertIs(failed['passed'], False)

    def test_other_model_rejected(self):
        review = copy.deepcopy(self.review)
        review['world_model_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'different world model'):
            self.validate(review)

    def test_other_reset_bank_rejected(self):
        with self.assertRaisesRegex(ValueError, 'different reset contexts'):
            self.validate(self.review, '0'*64)

    def test_unapproved_or_changed_raw_evidence_rejected(self):
        for label in ('failed_receipt','canonical_receipt','canonical_arrays','diagnostic_source'):
            review = copy.deepcopy(self.review)
            review[label]['sha256'] = '0'*64
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'specifically approved evidence'):
                self.validate(review)

    def test_pending_independent_approval_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            approval = json.loads(Path(self.review['independent_approval']['path']).read_text())
            approval['decision'] = 'pending'
            path = Path(tmp) / 'approval.json'
            path.write_text(json.dumps(approval))
            review = copy.deepcopy(self.review)
            review['independent_approval'] = {'path': str(path), 'sha256': sha256(path)}
            with self.assertRaisesRegex(ValueError, 'approval document|independent numerical approval'):
                self.validate(review)

    def test_prepared_state_or_noise_corruption_rejected(self):
        for key in ('prepared_actual_seq', 'noise_1_vmap', 'noise_2_vmap'):
            data = self.arrays()
            data[key].flat[0] += 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'not exact|Noise schedule'):
                N.measure_batch_control(data)

    def test_native_control_bound_violation_rejected(self):
        data = self.arrays()
        data['canonical_fixed_vmap'].flat[0] += 1
        with self.assertRaisesRegex(ValueError, 'exceeds the unchanged native batching control'):
            N.measure_batch_control(data)

    def test_recorded_metric_tampering_rejected(self):
        review = copy.deepcopy(self.review)
        review['measured_batch_control']['vmap']['max_abs'][0] = 0
        with self.assertRaisesRegex(ValueError, 'bounds differ from the raw arrays'):
            self.validate(review)


if __name__ == '__main__':
    unittest.main()
