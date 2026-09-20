"""Specific independently reviewed admission for the frozen Flow BF16 recipe.

This does not make the failed scalar-parity check pass. It preserves that
failure and verifies a separate, identified native batching control. No model
weights, precision, latent states, reward, or numerical tolerance are changed.
"""
import argparse
import json
from pathlib import Path

from studies.revision.inventory_common import FLOW_SHA256, sha256, source_hashes, write_json

SCHEMA = 'flow_native_batch_numerical_review_v1'
FLOW_SOURCE_SHA256 = '86790ce2dac999e4900c21993d2fcb058209b975559719623a24d065562bd402'
FAILED_RECEIPT_SHA256 = '5a8afe2aba9fb7cd567488a18178f5b43d31cf6fc5d75721fd003e761fab3f63'
CANONICAL_RECEIPT_SHA256 = 'afe3e5f20e9a4e38ecdb617765255aa079ffd488665de21eb29886f6d6ca9f3a'
CANONICAL_ARRAYS_SHA256 = '294abd7495784b09f57b64c34c352ec009132af197ff7cfd0aff32074620ac61'
DIAGNOSTIC_SOURCE_SHA256 = 'd91efa0fe61a580915aed3332b9178f9e97728a2b52748ccc51e4ca7b283ef19'
INDEPENDENT_APPROVAL_SHA256 = 'd65f4e1fe7af9c2d301c014730ca64e1b241bc3695a9ee3dbd03f9c0d22ee990'
EXPECTED_JAX_CONFIG = {'jax_default_matmul_precision': 'tensorfloat32',
                       'jax_default_prng_impl': 'threefry2x32', 'jax_enable_x64': False,
                       'jax_threefry_partitionable': True}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256(path)}


def measure_batch_control(arrays):
    """Recompute per-row native-control comparisons from the preserved arrays."""
    import numpy as np
    for name in ('seq', 'tau', 'aseq', 'slots', 'row_keys'):
        actual = {'seq': 'seq', 'tau': 'tau', 'aseq': 'aseq', 'slots': 'slot', 'row_keys': 'keys'}[name]
        _require(np.array_equal(arrays[name], arrays['prepared_actual_' + actual]),
                 f'Prepared core input is not exact: {name}')
    _require(np.array_equal(arrays['pre_n'], [8, 31, 32, 32]), 'Mixed-age test cases changed')
    _require(np.array_equal(arrays['pre_r'], [8, 8, 8, 0]), 'Mixed real-context ages changed')
    _require(np.array_equal(arrays['prepared_actual_real'], [8, 8, 7, 0]), 'Mixed-age slide bookkeeping changed')
    for i in range(3):
        _require(np.array_equal(arrays[f'noise_{i}_vmap'], arrays[f'noise_{i}_scalar']),
                 f'Noise schedule differs: {i}')
    _require(np.array_equal(arrays['lax_map'], arrays['scalar']), 'Unchanged scalar native core differs under lax.map')
    _require(np.array_equal(arrays['transition_dynamic_z'], arrays['transition_closed_z']),
             'Constant-versus-dynamic fixture mismatch is unresolved')
    _require(np.array_equal(arrays['canonical_native'], arrays['canonical_fixed_batch']),
             'Injected-noise diagnostic does not exactly reproduce the unchanged native full-batch helper')
    base = np.asarray(arrays['canonical_fixed_batch'], np.float64)
    _require(base.shape == (4, 4, 4, 32), 'Native latent shape changed')
    measured = {}
    for label, key in [('vmap', 'canonical_fixed_vmap'), ('native_scalar', 'canonical_fixed_rows')]:
        delta = (np.asarray(arrays[key], np.float64) - base).reshape(4, -1)
        _require(np.isfinite(delta).all(), 'Nonfinite native-control comparison')
        measured[label] = {'max_abs': np.abs(delta).max(1).tolist(),
                           'mae': np.abs(delta).mean(1).tolist(),
                           'rmse': np.sqrt(np.square(delta).mean(1)).tolist()}
    for metric in ('max_abs', 'mae', 'rmse'):
        _require(np.all(np.asarray(measured['vmap'][metric]) <= np.asarray(measured['native_scalar'][metric])),
                 f'Per-row vmap {metric} exceeds the unchanged native batching control')
    measured['criterion'] = 'Each row: vmap maximum/MAE/RMSE no greater than native batch4-versus-batch1 control; no tolerance added'
    measured['scope'] = 'Identified frozen seed-0 checkpoint and archived mixed-age/canonical GPU fixtures; not universal bitwise equivalence'
    return measured


def validate_review(path, *, reset_bank_sha256):
    """Reject unreviewed, altered, broader, or numerically unsupported admissions."""
    import numpy as np
    review = json.loads(Path(path).read_text())
    _require(review.get('schema') == SCHEMA and review.get('decision') == 'admit_native_batch_recipe',
             'Unknown or unapproved numerical-review decision')
    _require(review.get('world_model_sha256') == FLOW_SHA256, 'Numerical review names a different world model')
    _require(review.get('production_flow_sha256') == FLOW_SOURCE_SHA256, 'Numerical review names a changed Flow adapter')
    _require(sha256(Path(__file__).with_name('inventory_flow.py')) == FLOW_SOURCE_SHA256,
             'Production Flow adapter changed after the numerical review')
    expected = {'failed_receipt': FAILED_RECEIPT_SHA256, 'canonical_receipt': CANONICAL_RECEIPT_SHA256,
                'canonical_arrays': CANONICAL_ARRAYS_SHA256, 'diagnostic_source': DIAGNOSTIC_SOURCE_SHA256}
    for label, identity in expected.items():
        ref = review[label]
        _require(ref['sha256'] == identity and sha256(ref['path']) == identity,
                 f'Reviewed {label} bytes changed or are not the specifically approved evidence')
    approval_ref = review['independent_approval']
    _require(approval_ref['sha256'] == INDEPENDENT_APPROVAL_SHA256
             and sha256(approval_ref['path']) == INDEPENDENT_APPROVAL_SHA256,
             'Independent approval document changed or differs from the approved reviewer receipt')
    approval = json.loads(Path(approval_ref['path']).read_text())
    _require(approval.get('decision') == 'approve_native_batch_numerics' and approval.get('independent') is True
             and bool(approval.get('reviewer')) and bool(approval.get('explanation')),
             'Explicit independent numerical approval is absent')
    bindings = {'production_flow_sha256': FLOW_SOURCE_SHA256, 'world_model_sha256': FLOW_SHA256,
                'failed_receipt_sha256': FAILED_RECEIPT_SHA256, 'canonical_receipt_sha256': CANONICAL_RECEIPT_SHA256,
                'canonical_arrays_sha256': CANONICAL_ARRAYS_SHA256, 'diagnostic_source_sha256': DIAGNOSTIC_SOURCE_SHA256}
    for key, identity in bindings.items():
        _require(approval.get(key) == identity, f'Independent approval does not bind {key}')
    failed = json.loads(Path(review['failed_receipt']['path']).read_text())
    canonical = json.loads(Path(review['canonical_receipt']['path']).read_text())
    _require(failed['passed'] is False, 'Original strict failure must remain false and preserved')
    _require(failed['reset_bank_sha256'] == reset_bank_sha256 and failed['frozen_state_unchanged'] is True,
             'Failed fixture used different reset contexts or changed frozen state')
    _require(canonical['failed_receipt_sha256'] == FAILED_RECEIPT_SHA256
             and canonical['outputs_sha256'] == CANONICAL_ARRAYS_SHA256
             and canonical['source_sha256'] == DIAGNOSTIC_SOURCE_SHA256
             and canonical['flow_source_sha256'] == FLOW_SOURCE_SHA256
             and canonical['world_model_sha256'] == FLOW_SHA256,
             'Canonical evidence chain is inconsistent')
    _require(canonical['jax_config'] == EXPECTED_JAX_CONFIG, 'Reviewed numerical precision or PRNG changed')
    # Administrative queue/evaluator/protocol changes are permitted only through
    # the separate current-source review; all executed Flow dependencies stay exact.
    current = source_hashes()
    dependencies = ['studies/revision/inventory_flow.py', *[
        'studies/revision/inventory_source/' + name for name in
        ('wm_env2.py', 'wm_generic.py', 'kernels.py', 'tinywm_module.py', 'agent.py')]]
    for name in dependencies:
        _require(current[name] == failed['source_hashes'][name], f'Reviewed native dependency changed: {name}')
    with np.load(review['canonical_arrays']['path'], allow_pickle=False) as arrays:
        measured = measure_batch_control(arrays)
    _require(review['measured_batch_control'] == measured, 'Recorded numerical bounds differ from the raw arrays')
    return {'schema': SCHEMA, 'review_sha256': sha256(path), 'independent_approval_sha256': approval_ref['sha256'],
            'reviewer': approval['reviewer'], 'original_strict_parity_passed': False,
            'decision': 'admit_native_batch_recipe', 'jax_config': EXPECTED_JAX_CONFIG,
            'measured_batch_control': measured}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--failed-receipt', type=Path, required=True)
    p.add_argument('--canonical-receipt', type=Path, required=True)
    p.add_argument('--canonical-arrays', type=Path, required=True)
    p.add_argument('--diagnostic-source', type=Path, required=True)
    p.add_argument('--independent-approval', type=Path, required=True)
    p.add_argument('--reset-bank-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    import numpy as np
    with np.load(a.canonical_arrays, allow_pickle=False) as arrays:
        measured = measure_batch_control(arrays)
    review = {'schema': SCHEMA, 'decision': 'admit_native_batch_recipe',
              'world_model_sha256': FLOW_SHA256, 'production_flow_sha256': FLOW_SOURCE_SHA256,
              'failed_receipt': _reference(a.failed_receipt), 'canonical_receipt': _reference(a.canonical_receipt),
              'canonical_arrays': _reference(a.canonical_arrays), 'diagnostic_source': _reference(a.diagnostic_source),
              'independent_approval': _reference(a.independent_approval), 'measured_batch_control': measured}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    # Validate before publishing the final immutable receipt name.
    temporary = a.out.with_suffix(a.out.suffix + '.candidate')
    write_json(temporary, review)
    result = validate_review(temporary, reset_bank_sha256=a.reset_bank_sha256)
    temporary.replace(a.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
