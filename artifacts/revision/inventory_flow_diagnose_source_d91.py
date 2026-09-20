"""Nonreportable immutable-core investigation of rejected Flow parity.

No production source, weights, latent width, or tolerance is changed. All direct
core comparisons pass actions, slots, keys, and Gaussian arrays dynamically.
"""
import argparse
import ast
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wm', type=Path, required=True)
    p.add_argument('--failed-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    import numpy as np
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from studies.revision import inventory_flow as F
    from studies.revision.inventory_common import FLOW_SHA256, sha256, write_json
    if sha256(a.wm) != FLOW_SHA256 or jax.default_backend() != 'gpu':
        raise ValueError('Expected registered checkpoint on GPU')
    a.out.mkdir(parents=True)
    started = time.monotonic()
    receipt = json.loads((a.failed_dir / 'result.json').read_text())
    with np.load(a.failed_dir / 'comparisons.npz', allow_pickle=False) as data:
        postseq = np.array(data['seq_reference'], copy=True)
        postaseq = np.array(data['aseq_reference'], copy=True)
    slots = np.asarray(receipt['head_successor_slots'], np.int32)
    kf, kh, kd = jax.random.split(jax.random.PRNGKey(116), 3)
    keys = jax.random.split(kf, len(slots))
    act = jnp.asarray([3, 11, 0, 16], jnp.int32)
    wm = F.WorldModel(a.wm)

    # Recover the relevant pretransition state. The already-dropped leading
    # frame for full windows is unrecoverable and set to zero; it is immediately
    # dropped again and cannot enter any compared core input. New/future slots
    # in unfilled prefixes are restored to their original zero padding.
    preseq, preaseq = postseq.copy(), postaseq.copy()
    for row, n in enumerate(receipt['input_n']):
        if n == 32:
            preseq[row, 1:] = postseq[row, :-1]
            preaseq[row, 1:] = postaseq[row, :-1]
            preseq[row, 0] = 0
            preaseq[row, 0] = 0
        else:
            preseq[row, n:] = 0
            preaseq[row, n:] = 0
    prestate = {'seq': preseq, 'aseq': preaseq, 'n': np.asarray(receipt['input_n'], np.int32),
                'r': np.asarray(receipt['input_r'], np.int32), 't': np.asarray(receipt['input_t'], np.int32)}
    # Independent host preparation, with every input exposed in the receipt.
    seq, aseq = preseq.copy(), preaseq.copy()
    real = np.array(prestate['r'], copy=True)
    for row, n in enumerate(prestate['n']):
        if n == 32:
            seq[row] = np.concatenate([preseq[row, 1:], preseq[row, -1:]], 0)
            aseq[row] = np.concatenate([preaseq[row, 1:], preaseq[row, -1:]], 0)
            real[row] = max(int(real[row]) - 1, 0)
        aseq[row, slots[row]] = int(act[row])
    tau = np.where(np.arange(32)[None] < slots[:, None],
                   np.where(np.arange(32)[None] < real[:, None], 1., .9), 0.).astype(np.float32)
    inputs = tuple(jnp.asarray(x) for x in (seq, tau, aseq, slots, keys))
    arrays = {**{'pre_' + k: v for k, v in prestate.items()}, 'seq': seq, 'tau': tau,
              'aseq': aseq, 'slots': slots, 'row_keys': np.asarray(keys),
              'flow_key': np.asarray(kf), 'head_key': np.asarray(kh), 'done_key': np.asarray(kd)}
    comparisons, timings, outputs = {}, {}, {}

    def difference(x, y):
        x, y = np.asarray(x), np.asarray(y)
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        return {'exact': bool(np.array_equal(x, y)), 'max_abs': float(d.max()),
                'per_row_max_abs': d.reshape(4, -1).max(1).tolist()}

    def record():
        np.savez(a.out / 'outputs.npz', **arrays, **outputs)
        result = {'reportable': False, 'world_model_sha256': FLOW_SHA256,
                  'source_sha256': sha256(__file__), 'failed_receipt_sha256': sha256(a.failed_dir/'result.json'),
                  'flow_source_sha256': sha256(F.__file__),
                  'jax_config': {k: getattr(jax.config, k) for k in ('jax_default_matmul_precision',
                     'jax_default_prng_impl', 'jax_enable_x64', 'jax_threefry_partitionable')},
                  'comparisons': comparisons, 'timings_seconds': timings,
                  'elapsed_seconds': time.monotonic()-started, 'outputs_sha256': sha256(a.out/'outputs.npz')}
        write_json(a.out / 'result.json', result)
        print(json.dumps(result, indent=2), flush=True)

    # Execute the exact production AST up to the core call; this instruments a
    # copy in memory and leaves the production source file untouched.
    tree = ast.parse(Path(F.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'transition')
    fn.name = 'prepare_actual'
    cut = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'one' for t in n.targets))
    fn.body = fn.body[:cut] + ast.parse('return seq, taus, aseq, n, r, jax.random.split(flow_key, N)').body
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {'jax': jax, 'jnp': jnp}
    exec(compile(module, '<unchanged production prefix preparation>', 'exec'), namespace)
    actual_prep = jax.jit(lambda s, ac, fk, hk, dk: namespace['prepare_actual'](wm,s,ac,fk,hk,dk))(
        jax.tree.map(jnp.asarray, prestate), act, kf, kh, kd)
    for name, actual, reference in zip(('seq', 'tau', 'aseq', 'slot', 'real', 'keys'),
                                       actual_prep, (seq, tau, aseq, slots, real, keys)):
        comparisons['prepared_' + name] = difference(actual, reference)
        arrays['prepared_actual_' + name] = np.asarray(actual)
    if not all(c['exact'] for c in comparisons.values()):
        record()
        raise RuntimeError('Core input mismatch: stop before attributing differences to batching')

    def one(z, t, action, slot, key):
        return wm.flow(wm.state, z[None], t[None], action[None], slot, key, 4)[0]
    scalar = jax.jit(one)
    batched = jax.jit(jax.vmap(one))
    mapped = jax.jit(lambda *xs: jax.lax.map(lambda row: one(*row), xs))
    for name, call in [('scalar', lambda: jnp.stack([scalar(*(x[i] for x in inputs)) for i in range(4)])),
                       ('vmap', lambda: batched(*inputs)), ('lax_map', lambda: mapped(*inputs)),
                       ('vmap_size1', lambda: jnp.concatenate([batched(*(x[i:i+1] for x in inputs)) for i in range(4)], 0))]:
        t = time.monotonic()
        outputs[name] = np.asarray(call())
        timings[name+'_compile_and_call'] = time.monotonic()-t
        t = time.monotonic(); np.asarray(call()); timings[name+'_warm_call'] = time.monotonic()-t
        if name != 'scalar':
            comparisons[name+'_vs_scalar'] = difference(outputs[name], outputs['scalar'])
        record()
    # Explicitly dynamic inputs match production; keep the rejected closure
    # fixture as a separate comparison without using it as the reference.
    dynamic_transition = jax.jit(lambda s, ac, fk, hk, dk: F.transition(wm,s,ac,fk,hk,dk))
    dynamic = dynamic_transition(jax.tree.map(jnp.asarray, prestate), act, kf, kh, kd)
    outputs['transition_dynamic_z'] = np.asarray(dynamic[1]['seq'])[np.arange(4), slots]
    comparisons['transition_dynamic_vs_direct_vmap'] = difference(outputs['transition_dynamic_z'], outputs['vmap'])
    closed = jax.jit(lambda s: F.transition(wm,s,act,kf,kh,kd))(jax.tree.map(jnp.asarray, prestate))
    outputs['transition_closed_z'] = np.asarray(closed[1]['seq'])[np.arange(4), slots]
    comparisons['transition_closed_vs_dynamic'] = difference(outputs['transition_closed_z'], outputs['transition_dynamic_z'])
    record()

    def noise(z, t, action, slot, key):
        del action, slot
        kc, kz = jax.random.split(key)
        t5 = t[None, :, None, None, None]
        context_gaussian = jax.random.normal(kc, z[None].shape)
        initial_gaussian = jax.random.normal(kz, (1,) + z.shape[-3:])
        noisy = jnp.where(t5 > .5, t5*z[None] + (1-t5)*context_gaussian, z[None])
        return noisy, context_gaussian, initial_gaussian
    def cache(z, t, action, slot, key):
        noisy = noise(z,t,action,slot,key)[0]
        return nnx.merge(wm.gdef,wm.state).forward(noisy,t[None],action[None],cache=True)
    for name, call in [('noise', noise), ('cache', cache)]:
        scalar_call = jax.jit(call)
        rows = [jax.device_get(scalar_call(*(x[i] for x in inputs))) for i in range(4)]
        reference = jax.tree.map(lambda *row: np.stack(row), *rows)
        actual = jax.device_get(jax.jit(jax.vmap(call))(*inputs))
        for i, (x, y) in enumerate(zip(jax.tree.leaves(actual),jax.tree.leaves(reference))):
            comparisons[f'{name}_{i}_vmap_vs_scalar'] = difference(x,y)
            arrays[f'{name}_{i}_vmap'], arrays[f'{name}_{i}_scalar'] = np.asarray(x),np.asarray(y)
        record()

    # Canonical equal-slot case: use the same clean context in four rows and
    # explicitly inject identical Gaussian ARRAY slices into batch4 and batch1.
    # This preserves the original core equations and removes RNG-stream changes.
    def fixed_noise(z, t, action, slot, context_gaussian, initial_gaussian):
        apply = nnx.merge(wm.gdef,wm.state)
        B = z.shape[0]
        t5 = t[...,None,None,None]
        noisy = jnp.where(t5>.5,t5*z+(1-t5)*context_gaussian,z)
        dt = 1./4
        caches = apply.forward(noisy,t,action,cache=True)
        current_action = action[:,slot][:,None]
        def step(latent, tau_s):
            x1 = apply.forward(latent[:,None],jnp.full((B,1),tau_s),current_action,
                               caches=caches,slot=slot,delta=dt)[0][:,0]
            return latent+dt*(x1-latent)/jnp.maximum(1.-tau_s,1e-4),None
        return jax.lax.scan(step,initial_gaussian,jnp.arange(4)*dt)[0]
    canonical_seq = jnp.repeat(inputs[0][:1],4,axis=0)
    canonical_tau = jnp.repeat(inputs[1][:1],4,axis=0)
    canonical_actions = jnp.repeat(inputs[2][:1],4,axis=0)
    canonical_slot = jnp.asarray(8,jnp.int32)
    canonical_key = jax.random.PRNGKey(118)
    kc,kz = jax.random.split(canonical_key)
    context_gaussian = jax.random.normal(kc,canonical_seq.shape)
    initial_gaussian = jax.random.normal(kz,(4,)+canonical_seq.shape[-3:])
    native_batch = jax.jit(lambda z,t,ac,slot,key: wm.flow(wm.state,z,t,ac,slot,key,4))(
        canonical_seq,canonical_tau,canonical_actions,canonical_slot,canonical_key)
    fixed = jax.jit(fixed_noise)
    fixed_batch = fixed(canonical_seq,canonical_tau,canonical_actions,canonical_slot,context_gaussian,initial_gaussian)
    fixed_rows = jnp.concatenate([fixed(canonical_seq[i:i+1],canonical_tau[i:i+1],canonical_actions[i:i+1],
                                canonical_slot,context_gaussian[i:i+1],initial_gaussian[i:i+1]) for i in range(4)],0)
    fixed_vmapped = jax.jit(jax.vmap(
        lambda z,t,ac,slot,ce,ie: fixed_noise(z[None],t[None],ac[None],slot,ce[None],ie[None])[0],
        in_axes=(0,0,0,None,0,0)))(canonical_seq,canonical_tau,canonical_actions,canonical_slot,
                                 context_gaussian,initial_gaussian)
    comparisons['equal_slot_native_vs_injected_same_full_batch_noise'] = difference(native_batch,fixed_batch)
    comparisons['equal_slot_full_batch_vs_scalar_same_noise_arrays'] = difference(fixed_batch,fixed_rows)
    comparisons['equal_slot_full_batch_vs_vmap_same_noise_arrays'] = difference(fixed_batch,fixed_vmapped)
    arrays.update(canonical_seq=np.asarray(canonical_seq), canonical_context_gaussian=np.asarray(context_gaussian),
                  canonical_initial_gaussian=np.asarray(initial_gaussian), canonical_native=np.asarray(native_batch),
                  canonical_fixed_batch=np.asarray(fixed_batch), canonical_fixed_rows=np.asarray(fixed_rows),
                  canonical_fixed_vmap=np.asarray(fixed_vmapped))
    decode = jax.jit(wm.decode)
    latent = jnp.asarray(outputs['scalar'])
    pixels_batch = np.asarray(decode(latent))
    pixels_scalar = np.concatenate([np.asarray(decode(latent[i:i+1])) for i in range(4)],0)
    comparisons['decode_same_latent_batch_vs_scalar'] = difference(pixels_batch,pixels_scalar)
    arrays.update(scalar_pixels=pixels_scalar,batch_pixels=pixels_batch)
    record()


if __name__ == '__main__':
    main()
