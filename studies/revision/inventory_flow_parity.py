"""Nonreportable GPU parity fixture for the genuine-prefix Flow adapter.

Build real encoded eight-frame prefixes and native generated histories, then
compare asynchronous batched transitions with unchanged scalar native helpers.
This is an implementation test, never a policy evaluation or scientific sample.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

from studies.revision.inventory_common import FLOW_SHA256, sha256, source_hashes, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wm', type=Path, required=True)
    p.add_argument('--reset-bank', type=Path, required=True)
    p.add_argument('--reset-bank-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if sha256(a.wm) != FLOW_SHA256 or sha256(a.reset_bank) != a.reset_bank_sha256:
        raise ValueError('Parity fixture input identity mismatch')
    import numpy as np
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from studies.revision.inventory_flow import make_env, transition
    if jax.default_backend() != 'gpu':
        raise RuntimeError('The admission fixture must exercise GPU cached-attention kernels')
    a.out.mkdir(parents=True)
    began = time.monotonic()
    with np.load(a.reset_bank, allow_pickle=False) as bank:
        if str(bank['bank_format']) != 'genuine_episode_prefix8_v1':
            raise ValueError('Wrong initialization bank')
        ids = np.arange(4) % len(bank['contexts'])
        contexts, outgoing = bank['contexts'][ids], bank['actions'][ids]
    wm, reset, _ = make_env(a.wm, reset_contexts_u8=contexts, reset_actions=outgoing)
    frozen_before = wm.parameter_sha256()
    _, state0 = reset(jax.random.split(jax.random.PRNGKey(23), 4), jnp.arange(4))
    expected_incoming = np.concatenate([np.zeros((4, 1), np.int32), outgoing], 1)
    np.testing.assert_array_equal(np.asarray(state0['aseq'])[:, :8], expected_incoming)
    np.testing.assert_array_equal(np.asarray(state0['n']), np.full(4, 8))
    np.testing.assert_array_equal(np.asarray(state0['r']), np.full(4, 8))
    rollout_actions = jax.random.randint(jax.random.PRNGKey(114), (32, 4), 0, 17)

    @jax.jit
    def native_history(ctx, incoming, actions, key):
        return wm.m._rollout_latents(nnx.merge(wm.gdef, wm.state), ctx, incoming,
                                     actions, key, 4, wm.win)

    generated = np.asarray(native_history(state0['seq'][:, :8], state0['aseq'][:, :8],
                                         rollout_actions, jax.random.PRNGKey(115)))
    ages = (0, 23, 24, 32)
    state = {k: np.array(v, copy=True) for k, v in state0.items()}
    native_actions = np.asarray(rollout_actions)
    for row, age in enumerate(ages):
        frames = np.concatenate([np.asarray(state0['seq'])[row, :8], generated[:age, row]], 0)
        actions = np.concatenate([expected_incoming[row], native_actions[:age, row]], 0)
        state['seq'][row] = 0
        state['aseq'][row] = 0
        state['seq'][row, :min(len(frames), 32)] = frames[-32:]
        state['aseq'][row, :min(len(frames), 32)] = actions[-32:]
        state['n'][row] = min(8 + age, 32)
        state['r'][row] = max(8 - max(8 + age - 32, 0), 0)
        state['t'][row] = age
    state = jax.tree.map(jnp.asarray, state)
    act = jnp.asarray([3, 11, 0, 16], jnp.int32)  # arbitrary native actions, no task-success claim
    kf, kh, kd = jax.random.split(jax.random.PRNGKey(116), 3)
    actual = jax.jit(lambda s: transition(wm, s, act, kf, kh, kd))(state)
    actual = jax.tree.map(lambda x: np.asarray(x), actual)

    # Reference indexing is performed separately in host arrays. Only the frozen
    # original scalar-slot Flow helper computes the reference next latent.
    expected_state = {k: np.array(v, copy=True) for k, v in state.items()}
    slots = []
    z_expected = []
    one = jax.jit(lambda seq, taus, actions, slot, key:
                 wm.flow(wm.state, seq, taus, actions, slot, key, 4))
    row_keys = jax.random.split(kf, 4)
    for row in range(4):
        n, real = int(expected_state['n'][row]), int(expected_state['r'][row])
        if n == 32:
            expected_state['seq'][row, :-1] = expected_state['seq'][row, 1:]
            expected_state['aseq'][row, :-1] = expected_state['aseq'][row, 1:]
            real = max(real - 1, 0)
            expected_state['r'][row] = real
        slot = min(n, 31)
        slots.append(slot)
        expected_state['aseq'][row, slot] = int(act[row])
        tau = np.where(np.arange(32) < slot,
                       np.where(np.arange(32) < real, 1., wm.ctx_signal), 0.).astype(np.float32)
        z = np.asarray(one(jnp.asarray(expected_state['seq'][row:row + 1]), tau[None],
                           jnp.asarray(expected_state['aseq'][row:row + 1]), jnp.asarray(slot), row_keys[row]))[0]
        expected_state['seq'][row, slot] = z
        expected_state['n'][row] = slot + 1
        expected_state['t'][row] += 1
        z_expected.append(z)
    expected_reward, expected_pdone = jax.jit(lambda seq, actions: wm.read(wm.state, seq, actions, kh))(
        jnp.asarray(expected_state['seq']), jnp.asarray(expected_state['aseq']))
    expected_reward = np.asarray(expected_reward)[np.arange(4), slots]
    expected_pdone = np.asarray(expected_pdone)[np.arange(4), slots]
    expected_done = (np.asarray(jax.random.uniform(kd, (4,))) < expected_pdone) | (expected_state['t'] >= 1024)
    expected_pixels = np.asarray(jax.jit(wm.decode)(jnp.asarray(np.stack(z_expected)))).astype(np.float32) / 255.
    comparisons = {'observation': (actual[0], expected_pixels), 'reward': (actual[2], expected_reward),
                   'done': (actual[3], expected_done), **{k: (actual[1][k], v) for k, v in expected_state.items()}}
    checks = {}
    arrays = {}
    for name, (observed, expected) in comparisons.items():
        arrays[name + '_actual'] = observed
        arrays[name + '_reference'] = expected
        # A strict numerical check exposes changed batched-kernel behavior for review.
        exact = np.array_equal(observed, expected)
        passed = exact if observed.dtype.kind in 'biu' else np.allclose(observed, expected, rtol=1e-5, atol=1e-5)
        checks[name] = {'passed': bool(passed), 'exact': bool(exact),
                        'max_abs_error': float(np.max(np.abs(observed.astype(float) - expected.astype(float))))}
    np.savez(a.out / 'comparisons.npz', **arrays)
    result = {'passed': all(c['passed'] for c in checks.values()) and wm.parameter_sha256() == frozen_before,
              'reportable': False, 'backend': jax.default_backend(), 'source_hashes': source_hashes(),
              'parity_source_sha256': sha256(__file__), 'world_model_sha256': FLOW_SHA256,
              'reset_bank_sha256': a.reset_bank_sha256, 'checks': checks,
              'input_n': np.asarray(state['n']).tolist(), 'input_r': np.asarray(state['r']).tolist(),
              'input_t': ages, 'head_successor_slots': slots,
              'real_prefix': 'Encoded genuine8 frames,7 recorded actions; longer histories from native _rollout_latents',
              'rng': 'Scalar native _flow_one uses the identical split(flow_key,4)[row] as the vmapped adapter',
              'frozen_state_unchanged': wm.parameter_sha256() == frozen_before,
              'seconds': time.monotonic() - began, 'comparison_sha256': sha256(a.out / 'comparisons.npz')}
    write_json(a.out / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
