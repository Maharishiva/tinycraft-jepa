import argparse
import contextlib
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from unittest import mock

SOURCE = Path('/private/tmp/tinycraft-jepa-iclr-20260919/studies/revision/run_termination_queue.py')
spec = importlib.util.spec_from_file_location('termination_queue', SOURCE)
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)
ORIGINAL_ROOT = Q.ROOT
CHECKS = []


def fixture(base):
    root = Path(base).resolve() / 'checkout'
    for name in Q.SOURCE_NAMES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ORIGINAL_ROOT / name, target)
    models = Path(base).resolve() / 'models'
    for seed in range(3):
        target = models / ('tile_factored' if seed == 0 else f'tile_factored_s{seed}') / 'model_probe.npz'
        target.parent.mkdir(parents=True)
        target.write_bytes(f'fixture-only-model-{seed}'.encode())
    data = Path(base).resolve() / 'train.npz'
    data.write_bytes(b'fixture-only-data')
    args = argparse.Namespace(model_root=models, data=data, gpu='1',
        deadline_utc=datetime.fromtimestamp(time.time() + 10000, timezone.utc).isoformat(),
        minimum_job_seconds=300.0)
    return root, args


with tempfile.TemporaryDirectory(prefix='termination-queue-dryrun-') as tmp:
    root, args = fixture(tmp)
    before = list((ORIGINAL_ROOT / 'runs/revision').glob('termination*'))
    result = subprocess.run([sys.executable, str(SOURCE), '--model-root', str(args.model_root),
        '--data', str(args.data), '--deadline-utc', args.deadline_utc, '--dry-run'],
        check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    assert plan['dry_run'] and plan['n_planned'] == 24
    assert list((ORIGINAL_ROOT / 'runs/revision').glob('termination*')) == before
    assert len(plan['jobs']) == 24
    assert [(j['torso'], j['model_seed'], j['policy_seed']) for j in plan['jobs']] == [
        (t, m, p) for t, n in [('mlp', 5), ('native_grid', 3)] for m in range(3) for p in range(n)]
    assert all(j['status'] == 'not_started' and j['gpu'] == '1' for j in plan['jobs'])
    assert plan['runtime_admission']['minimum_campaign_seconds'] == 7200
    assert plan['source_sha256'][str(ORIGINAL_ROOT / 'research/termination_sensitivity_protocol.md')] == \
        '001b66925f6d17cb3ee3b313fd27986219662de79b93f8a90aa2f1d3e0cff3c8'
    CHECKS.append('CLI dry run: exact 24-job ordering, budgets, protocol hash; no output creation')
    Q.ROOT = root
    campaign, manifest, deadline = Q.build_plan(args, root)
    for gpu in ['0,1', '-1', 'GPU-abc']:
        bad = argparse.Namespace(**vars(args)); bad.gpu = gpu
        try: Q.build_plan(bad, root)
        except ValueError: pass
        else: raise AssertionError('multi/invalid GPU admitted')
    for value in ['2026-01-01T00:00:00+00:00', '2027-01-01T00:00:00', '2027-01-01T01:00:00+01:00']:
        bad = argparse.Namespace(**vars(args)); bad.deadline_utc = value
        try: Q.build_plan(bad, root)
        except ValueError: pass
        else: raise AssertionError('invalid deadline admitted')
    for value in [-1.0, 0.0, float('nan'), float('inf')]:
        bad = argparse.Namespace(**vars(args)); bad.minimum_job_seconds = value
        try: Q.build_plan(bad, root)
        except ValueError: pass
        else: raise AssertionError('invalid floor admitted')
    campaign.mkdir(parents=True)
    try: Q.build_plan(args, root)
    except ValueError: pass
    else: raise AssertionError('existing root admitted')
    campaign.rmdir()
    (root / 'studies/revision/matched_ppo.py').write_text('changed')
    try: Q.build_plan(args, root)
    except ValueError: pass
    else: raise AssertionError('parent drift admitted')
    CHECKS.append('Refused existing roots, invalid GPU/deadline/runtime floor, and frozen-parent drift')

for scenario in ['complete', 'failed', 'incomplete', 'timeout', 'admission', 'ledger_nonfinite', 'terminal_mass']:
    with tempfile.TemporaryDirectory(prefix='termination-queue-test-') as tmp:
        root, args = fixture(tmp)
        Q.ROOT = root
        Q.ACTIVE_CAMPAIGN = None
        if scenario in ('timeout', 'admission'):
            args.minimum_job_seconds = 0.001 if scenario == 'timeout' else 300.0
            args.deadline_utc = datetime.fromtimestamp(time.time() + (0.5 if scenario == 'timeout' else 1.0),
                timezone.utc).isoformat()
        campaign, manifest, deadline = Q.build_plan(args, root)
        calls = []

        class FakeProcess:
            def __init__(self, command, **kwargs):
                calls.append(command)
                self.returncode = None
                self.pid = 987654321
                self.job = manifest['jobs'][len(calls) - 1]

            def wait(self, timeout=None):
                if self.returncode is not None:
                    return self.returncode
                self.returncode = 3 if scenario == 'failed' else 0
                if scenario == 'failed':
                    return self.returncode
                job = self.job
                out = Path(job['out'])
                out.mkdir(parents=True)
                required = ['studies/revision/matched_ppo_termination.py', 'studies/revision/matched_ppo.py',
                    'studies/stoch/rl.py', 'studies/stoch/env.py', 'train.py', 'probe.py', 'studies/redcube/env.py']
                if job['torso'] == 'native_grid':
                    required += ['studies/revision/policy.py']
                config = {'args': {key.replace('-', '_'): value for key, value in Q.FIXED_ARGS.items()},
                    'parent_harness_sha256': Q.PARENT_SHA256, 'sensitivity_harness_sha256': Q.HARNESS_SHA256,
                    'model_sha256': manifest['content_sha256'][job['model']],
                    'data_sha256': manifest['content_sha256'][job['data']],
                    'source_sha256': {str(root / name): manifest['source_sha256'][str(root / name)] for name in required}}
                for key in ['model_seed', 'policy_seed', 'torso', 'model', 'data', 'out']:
                    config['args'][key] = job[key]
                (out / 'model.pkl').write_bytes(b'fixture-only-policy')
                for name in ['episodes.npz', 'starts.npz']:
                    (out / name).write_bytes(b'fixture-only-artifact')
                final = {key: job[key] for key in ['arm', 'torso', 'model_seed', 'policy_seed']}
                final.update(budget_complete=scenario != 'incomplete',
                    updates_completed=150 if scenario != 'incomplete' else 1,
                    updates_requested=150, transitions=1228800, eval_seed=20260919, n_ep=256,
                    model_sha256=config['model_sha256'], policy_sha256=Q.sha256(out / 'model.pkl'))
                (out / 'final.json').write_text(json.dumps(final))
                (out / 'config.json').write_text(json.dumps(config))
                rows = [{'update': i, 'loss': 0.5, 'postterminal_alive_mass_fraction': 0.0,
                         'postterminal_reward_per_rollout': 0.0} for i in range(1, 151)]
                if scenario == 'ledger_nonfinite': rows[-1]['loss'] = float('nan')
                if scenario == 'terminal_mass': rows[-1]['postterminal_alive_mass_fraction'] = 0.1
                (out / 'ledger.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
                return self.returncode

            def poll(self):
                return self.returncode

        if scenario == 'timeout':
            manifest['jobs'][0]['command'] = [sys.executable, '-c',
                'import time; print("started", flush=True); time.sleep(60)']
        patches = [mock.patch.object(Q, 'build_plan', return_value=(campaign, manifest, deadline)),
            mock.patch.object(Q, 'assert_idle_gpu'),
            mock.patch.object(sys, 'argv', ['queue', '--model-root', str(args.model_root),
                '--data', str(args.data), '--deadline-utc', args.deadline_utc])]
        with contextlib.ExitStack() as stack:
            for patch in patches: stack.enter_context(patch)
            if scenario != 'timeout': stack.enter_context(mock.patch.object(Q.subprocess, 'Popen', FakeProcess))
            with contextlib.redirect_stdout(io.StringIO()):
                try: Q.main()
                except (RuntimeError, TimeoutError):
                    assert scenario not in ('complete', 'admission')
                else:
                    assert scenario in ('complete', 'admission')
        saved = json.loads((campaign / 'campaign.json').read_text())
        status = json.loads((campaign / 'campaign_status.json').read_text())
        assert len(saved['jobs']) == 24 and status['n_planned'] == 24
        if scenario == 'complete':
            assert saved['status'] == 'complete' and status['n_complete'] == 24 and len(calls) == 24
            assert all(len(j['artifact_sha256']) == 6 for j in saved['jobs'])
        elif scenario == 'admission':
            assert saved['status'] == 'not_admitted_deadline' and len(calls) == 0
        elif scenario == 'timeout':
            assert saved['status'] == 'deadline_incomplete' and saved['jobs'][0]['status'] == 'deadline_timeout'
            assert Path(saved['jobs'][0]['log']).exists()
            assert all(j['status'] == 'not_started' for j in saved['jobs'][1:])
        else:
            assert saved['status'] == 'failed_incomplete' and len(calls) == 1
            assert saved['jobs'][0]['status'] == 'failed_incomplete'
            assert all(j['status'] == 'not_started' for j in saved['jobs'][1:])
        CHECKS.append(f'{scenario}: expected status, full denominator, preserved logs, no retry')

print('Launcher SHA256:', Q.sha256(SOURCE))
print('Tests use only temporary synthetic files and mock jobs except one real Python sleep process to exercise the timeout. No GPU or scientific job was launched.')
for item in CHECKS: print('PASS', item)
