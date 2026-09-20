"""Prospective from-scratch pixel PPO in a frozen Flow or native-512 JEPA model.

Collector, GAE preparation and PPO update bodies are byte-for-byte copies of the
clean historical trainer. Both families use one fixed model seed and three paired
policy seeds. No BC, anchor, action mask, reward alteration or checkpoint selection.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from studies.revision.inventory_common import DATA_SHA256, FLOW_SHA256, sha256, source_hashes, write_json, validate_jepa_bundle
from studies.revision.inventory_numerical_review import EXPECTED_JAX_CONFIG


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=('flow', 'jepa'), required=True)
    p.add_argument('--wm', type=Path, required=True)
    p.add_argument('--wm-sha256', required=True)
    p.add_argument('--reset-bank', type=Path, required=True)
    p.add_argument('--reset-bank-sha256', required=True)
    p.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-seconds', type=float, default=3600)
    p.add_argument('--smoke-updates', type=int, default=0)
    return p


def parameter_hash(module, nnx, jax, np):
    h = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(nnx.state(module, nnx.Param))[0]:
        value = np.asarray(value)
        h.update(str(path).encode()); h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def main():
    a = parser().parse_args()
    if a.out.exists():
        raise FileExistsError(f'Refusing to overwrite run: {a.out}')
    if sha256(a.wm) != a.wm_sha256:
        raise ValueError('World-model checkpoint hash mismatch')
    if a.family == 'flow' and a.wm_sha256 != FLOW_SHA256:
        raise ValueError('Flow checkpoint is not the frozen predeclared seed-0 model')
    if sha256(a.reset_bank) != a.reset_bank_sha256:
        raise ValueError('Reset-bank hash mismatch')
    probe_admission = validate_jepa_bundle(a.wm) if a.family == 'jepa' else None
    a.out.mkdir(parents=True)
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    # The historical Flow module sets this on import; set equally for both arms.
    jax.config.update('jax_default_matmul_precision', 'tensorfloat32')
    numerical_config = {key: getattr(jax.config, key) for key in EXPECTED_JAX_CONFIG}
    if numerical_config != EXPECTED_JAX_CONFIG:
        raise ValueError('Policy runtime precision or PRNG differs from the reviewed common configuration')
    sys.path.insert(0, str(Path(__file__).with_name('inventory_source')))
    from agent import AC, PHID, entropy, gae, hl_expect, hl_loss, logprob, n_params, save_ac, to_u8
    with np.load(a.reset_bank, allow_pickle=False) as bank:
        if str(bank['training_data_sha256']) != DATA_SHA256:
            raise ValueError('Bank did not originate in the shared frozen 500k archive')
        if str(bank['bank_format']) != 'genuine_episode_prefix8_v1':
            raise ValueError('Expected the approved genuine eight-frame-prefix initialization')
        reset_contexts, reset_actions = np.asarray(bank['contexts']), np.asarray(bank['actions'])
    if a.smoke_updates < 0:
        raise ValueError('smoke-updates must be nonnegative')
    # Immutable historical wmgym_5m_nogru recipe.
    a.envs, a.collect, a.chunk, a.groups, a.epochs = 64, 100, 25, 4, 4
    a.lr, a.gamma, a.lam, a.clip, a.vf, a.ent = 2.5e-4, .99, .8, .2, .5, .01
    a.budget = a.smoke_updates * 6400 if a.smoke_updates else 5_000_000
    a.eval_every = 8
    E, T, LC = a.envs, a.collect, a.chunk
    NCH, NG = T // LC, a.groups
    CPG = NCH // NG
    zc = jnp.zeros((E, 0))
    ac = AC(nnx.Rngs(a.seed + 2), recurrent=False)
    assert n_params(ac) == 6_092_177
    initial_actor_sha256 = parameter_hash(ac, nnx, jax, np)
    opt = nnx.Optimizer(ac, optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(a.lr, weight_decay=0.01)),
                        wrt=nnx.Param)
    if a.family == 'flow':
        from studies.revision.inventory_flow import make_env
        wm, vreset, vstep = make_env(a.wm, nfe=4, max_steps=1024,
                                    reset_contexts_u8=reset_contexts, reset_actions=reset_actions)
    else:
        from studies.revision.inventory_jepa import make_env
        wm, vreset, vstep = make_env(a.wm, max_steps=1024,
                                    reset_contexts_u8=reset_contexts, reset_actions=reset_actions)
    adapter_hash_before = wm.parameter_sha256() if hasattr(wm, 'parameter_sha256') else None
    manifest = {'protocol': 'native-inventory-v1', 'family': a.family, 'model_seed': 0,
                'policy_seed': a.seed, 'actor': {'params': n_params(ac), 'recurrent': False,
                'cond_dim': 0, 'actions': 17, 'initial_parameter_sha256': initial_actor_sha256},
                'config': vars(a), 'world_model_sha256': a.wm_sha256,
                'data_sha256': DATA_SHA256, 'reset_bank_sha256': a.reset_bank_sha256,
                'sources': source_hashes(), 'devices': [str(d) for d in jax.devices()],
                'backend': jax.default_backend(), 'device_platforms': sorted({d.platform for d in jax.devices()}),
                'packages': {k: importlib.metadata.version(k) for k in ('jax', 'jaxlib', 'flax', 'optax', 'numpy')},
                'matmul_precision': str(jax.config.jax_default_matmul_precision),
                'jax_config': numerical_config,
                'reset_pairing': 'Same candidate bank indices at each absolute training step. Accepted resets differ with learned done.',
                'reset_sampler': 'jax.random.randint(keys[0], (64,), 0, bank_size), called each step for every slot',
                'training_time_limit': 1024, 'native_evaluation_time_limit': 10000,
                'intermediate_evaluation': 'Omitted; historical eval key consumed every 8 iterations and final.',
                'selection': 'Fixed final checkpoint only, all six runs retained; no metric-based selection',
                'reward': 'Raw single learned reward head; sampled learned termination; no modifications',
                'native_history': 32 if a.family == 'flow' else 8,
                'initial_context': 'Eight genuine prefix frames and seven recorded outgoing actions; no reset crossing',
                'actor_reset_previous_action': 0,
                'smoke': bool(a.smoke_updates)}
    manifest['adapter_parameter_sha256_before'] = adapter_hash_before
    manifest['jepa_probe_admission'] = probe_admission
    write_json(a.out / 'config.json', manifest)
    print('[init] ' + json.dumps({k: manifest[k] for k in ('family', 'model_seed', 'policy_seed', 'actor', 'devices')}), flush=True)
    @nnx.jit
    def collect(ac, es, obs_u8, pol, rng):
        def step(carry, key):
            es, obs, (g, prev_a), rng = carry
            ka, ks, kr = jax.random.split(key, 3)
            logits, vlog, g2 = ac.step(obs, prev_a, g, zc)
            act = jax.random.categorical(ka, logits).astype(jnp.int32)
            nobs, nes, rew, done, _ = vstep(jax.random.split(ks, E), es, act)
            robs, res = vreset(jax.random.split(kr, E))
            nes = jax.tree.map(lambda r, x: jnp.where(done.reshape((E,) + (1,) * (x.ndim - 1)), r, x), res, nes)
            nobs = jnp.where(done[:, None, None, None], robs, nobs)
            out = (obs, act, prev_a, rew, done.astype(jnp.float32), hl_expect(vlog), logprob(logits, act))
            g2 = jnp.where(done[:, None], 0.0, g2)
            return (nes, to_u8(nobs), (g2, jnp.where(done, 0, act)), rng), out

        (es, obs_u8, pol, rng), outs = jax.lax.scan(step, (es, obs_u8, pol, rng), jax.random.split(rng, T))
        return es, obs_u8, pol, outs

    @nnx.jit
    def prep(ac, frames_full, acts, pas_full, rews, dones, g0):
        acts_full = jnp.concatenate([acts, jnp.zeros((1, E), jnp.int32)], 0)
        dones_full = jnp.concatenate([dones, jnp.zeros((1, E))], 0)

        def hstep(g, inp):
            obs, a_t, pa_t, d_t = inp
            logits, vlog, g2 = ac.step(obs, pa_t, g, zc)
            out = (logprob(logits, a_t), hl_expect(vlog), g)
            return jnp.where(d_t[:, None] > 0.5, 0.0, g2), out

        _, (lp, val, gs) = jax.lax.scan(hstep, g0, (frames_full, acts_full, pas_full, dones_full))
        adv, ret = gae(val[:T], rews, dones, val[T], a.gamma, a.lam)
        return lp[:T], adv, ret, gs[:T]

    @nnx.jit
    def ppo(ac, opt, frames, acts, pas, dones, gs, oldlp, adv, ret, gi):
        def rs(x):
            x = x.reshape((NG, CPG, LC) + x.shape[1:])[gi]
            return x.swapaxes(0, 1).reshape((LC, CPG * E) + x.shape[3:])

        advn = (adv - adv.mean()) * jax.lax.rsqrt(adv.var() + 1e-5)
        fr, ac_t, pa, d = rs(frames), rs(acts), rs(pas), rs(dones)
        olp, an, rt = rs(oldlp), rs(advn), rs(ret)
        g0 = jax.lax.stop_gradient(gs[::LC].reshape((NG, CPG * E, PHID))[gi])
        zcg = jnp.zeros((CPG * E, 0))

        def loss_fn(ac):
            def sstep(g, inp):
                obs, pa_t, d_t = inp
                logits, vlog, g2 = ac.step(obs, pa_t, g, zcg)
                return jnp.where(d_t[:, None] > 0.5, 0.0, g2), (logits, vlog)

            _, (logits, vlog) = jax.lax.scan(sstep, g0, (fr, pa, d))
            ratio = jnp.exp(logprob(logits, ac_t) - olp)
            pg = -jnp.minimum(ratio * an, jnp.clip(ratio, 1 - a.clip, 1 + a.clip) * an).mean()
            vf = hl_loss(vlog, rt).mean()
            ent = entropy(logits).mean()
            return pg + a.vf * vf - a.ent * ent, (pg, vf, ent)

        (loss, aux), gr = nnx.value_and_grad(loss_fn, has_aux=True)(ac)
        opt.update(ac, gr)
        return jnp.stack([loss, aux[0], aux[1], aux[2]])

    rng = jax.random.PRNGKey(a.seed + 7)
    rng, r0 = jax.random.split(rng)
    robs, es = vreset(jax.random.split(r0, E))
    obs_u8 = to_u8(robs)
    pol = (jnp.zeros((E, PHID)), jnp.zeros((E,), jnp.int32))
    env_steps, it, start = 0, 0, time.monotonic()
    collection_keys = []
    skipped_eval_keys = []
    durations = []
    while env_steps + T * E <= a.budget:
        if time.monotonic() - start >= a.max_seconds:
            break
        it += 1
        t0 = time.monotonic()
        rng, kc = jax.random.split(rng)
        collection_keys.append(np.asarray(kc))
        g_start = pol[0]
        es, obs_u8, pol, (frames, acts, pas, rews, dones, vals, lps) = collect(ac, es, obs_u8, pol, kc)
        env_steps += T * E
        oldlp, adv, ret_t, gs = prep(ac, jnp.concatenate([frames, obs_u8[None]], 0), acts,
                                     jnp.concatenate([pas, pol[1][None]], 0), rews, dones, g_start)
        last = np.zeros(4)
        for _ in range(a.epochs):
            for gi in range(NG):
                last = np.asarray(ppo(ac, opt, frames, acts, pas, dones, gs, oldlp, adv, ret_t, jnp.asarray(gi)))
        if not np.isfinite(last).all() or not np.isfinite(np.asarray(rews)).all():
            raise FloatingPointError('Nonfinite PPO loss or raw learned reward; run retained as failed')
        durations.append(time.monotonic() - t0)
        row = {'iter': it, 'env_steps': env_steps, 'sec': durations[-1],
               'learned_reward_mean': float(jnp.mean(rews)), 'done_mean': float(jnp.mean(dones)),
               'loss': float(last[0]), 'pg': float(last[1]), 'vf': float(last[2]), 'ent': float(last[3])}
        print('[iter] ' + json.dumps(row), flush=True)
        with (a.out / 'training.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        final = env_steps + T * E > a.budget
        if it % a.eval_every == 0 or final:
            # Original evaluate() split the training RNG once; preserve that exact stride.
            rng, ke = jax.random.split(rng)
            skipped_eval_keys.append(np.asarray(ke))
    reached_budget = env_steps == a.budget // (T * E) * (T * E)
    budget_complete = reached_budget and not a.smoke_updates and env_steps == 4_998_400
    name = 'model.pkl' if reached_budget else 'model_incomplete.pkl'
    meta = {'cond': 'none', 'arm': 'inventory_wm_ppo', 'family': a.family,
            'model_seed': 0, 'policy_seed': a.seed, 'iter': it, 'env_steps': env_steps,
            'budget_complete': budget_complete, 'smoke': bool(a.smoke_updates),
            'world_model_sha256': a.wm_sha256, 'reset_bank_sha256': a.reset_bank_sha256,
            'model_sha256': a.wm_sha256,
            'initial_actor_sha256': initial_actor_sha256}
    save_ac(ac, a.out / name, meta)
    key_arrays = dict(initial_reset_key=np.asarray(r0), collection_keys=np.asarray(collection_keys),
                      omitted_evaluation_keys=np.asarray(skipped_eval_keys))
    np.savez(a.out / 'training_keys.npz', **key_arrays)
    key_hash = hashlib.sha256()
    for key, value in sorted(key_arrays.items()):
        key_hash.update(f'{key}:{value.dtype}:{value.shape}:'.encode())
        key_hash.update(value.tobytes())
    if sha256(a.wm) != a.wm_sha256:
        raise RuntimeError('Frozen checkpoint bytes changed during policy training')
    frozen_state_after = getattr(wm, 'parameter_sha256', None)
    if callable(frozen_state_after):
        frozen_state_after = frozen_state_after()
    if frozen_state_after != adapter_hash_before:
        raise RuntimeError('Frozen in-memory adapter state changed during policy training')
    result = {**meta, 'checkpoint': str(a.out / name), 'checkpoint_sha256': sha256(a.out / name),
              'elapsed_seconds': time.monotonic() - start, 'iteration_seconds': durations,
              'steady_iteration_seconds': float(np.median(durations[1:])) if len(durations) > 1 else None,
              'source_hashes_unchanged': source_hashes() == manifest['sources'],
              'source_hashes': manifest['sources'],
              'backend': manifest['backend'], 'device_platforms': manifest['device_platforms'],
              'jax_config': manifest['jax_config'],
              'world_model_bytes_unchanged': True, 'adapter_state_unchanged': True,
              'adapter_parameter_sha256_after': frozen_state_after,
              'training_keys_sha256': sha256(a.out / 'training_keys.npz'),
              'training_key_arrays_sha256': key_hash.hexdigest()}
    if not result['source_hashes_unchanged']:
        raise RuntimeError('Source bytes changed during the policy run')
    write_json(a.out / 'final.json', result)
    print('[done] ' + json.dumps({k: v for k, v in result.items() if k != 'iteration_seconds'}), flush=True)
    if not reached_budget:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
