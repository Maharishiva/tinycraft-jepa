"""Fixed six-run Flow/JEPA queue with source admission and exclusive GPU ownership.

No native evaluations occur during policy training. Every complete fixed final
checkpoint receives the same independent 256-episode native evaluation.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from studies.revision.inventory_common import FLOW_SHA256, sha256, source_hashes, write_json, validate_jepa_bundle
from studies.revision.inventory_numerical_review import EXPECTED_JAX_CONFIG, validate_review


def parse_deadline(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('deadline must include UTC timezone')
    return parsed.timestamp()


def jobs_for(args):
    jobs = []
    for seed in (0, 1, 2):
        for family in ('flow', 'jepa'):
            model = getattr(args, family + '_wm')
            digest = FLOW_SHA256 if family == 'flow' else args.jepa_sha256
            directory = args.out / f'{family}_p{seed}'
            command = [sys.executable, '-u', '-m', 'studies.revision.inventory_ppo',
                       '--family', family, '--wm', str(model), '--wm-sha256', digest,
                       '--reset-bank', str(args.reset_bank), '--reset-bank-sha256', args.reset_bank_sha256,
                       '--seed', str(seed), '--out', str(directory),
                       '--max-seconds', str(getattr(args, 'max_hours', 5.0) * 3600)]
            evaluation = [sys.executable, '-u', '-m', 'studies.revision.inventory_eval',
                          '--checkpoint', str(directory / 'model.pkl'), '--out', str(directory / 'native'),
                          '--eval-seed', '123']
            jobs.append({'family': family, 'model_seed': 0, 'policy_seed': seed,
                         'name': f'{family}_p{seed}', 'out': str(directory),
                         'world_model_sha256': digest, 'train_command': command, 'eval_command': evaluation})
    return jobs


def assert_gpu_idle(gpu):
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                    '--format=csv,noheader'], text=True)
    occupants = [line for line in text.splitlines() if line.split(',')[0].strip() == uuid]
    if occupants:
        raise RuntimeError(f'GPU {gpu} is not idle: {occupants}')
    return uuid


def run_process(command, log_path, env, deadline):
    """Each stage is its own process group; never kill processes outside that group."""
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError('Campaign deadline reached before next stage')
    with log_path.open('x') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=remaining)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            except ProcessLookupError:
                pass
            raise
        finally:
            # Even a normally exited leader may leave a descendant; clean only its owned group.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def validate_admission(path, args):
    admission = json.loads(path.read_text())
    if not admission.get('source_review_approved'):
        raise ValueError('A recorded independent source review is required')
    if admission.get('source_hashes') != source_hashes():
        raise ValueError('Source bytes differ from the approved source review')
    for family in ('flow', 'jepa'):
        smoke_path = Path(admission['smoke_results'][family])
        smoke = json.loads(smoke_path.read_text())
        expected = FLOW_SHA256 if family == 'flow' else args.jepa_sha256
        if not (smoke['smoke'] and smoke['family'] == family and smoke['env_steps'] >= 12800
                and smoke['world_model_sha256'] == expected
                and smoke['reset_bank_sha256'] == args.reset_bank_sha256
                and smoke['world_model_bytes_unchanged'] and smoke['adapter_state_unchanged']
                and smoke['source_hashes_unchanged'] and smoke['source_hashes'] == source_hashes()
                and smoke['jax_config'] == EXPECTED_JAX_CONFIG
                and smoke['backend'] == 'gpu' and smoke['device_platforms'] == ['gpu']):
            raise ValueError(f'{family} GPU smoke does not satisfy the admission contract')
    if 'flow_numerical_review' in admission:
        admission['verified_flow_numerical_review'] = validate_review(
            Path(admission['flow_numerical_review']), reset_bank_sha256=args.reset_bank_sha256)
    else:
        parity_path = Path(admission['flow_parity_result'])
        parity = json.loads(parity_path.read_text())
        if not (parity['passed'] and parity['backend'] == 'gpu' and parity['frozen_state_unchanged']
                and parity['world_model_sha256'] == FLOW_SHA256
                and parity['reset_bank_sha256'] == args.reset_bank_sha256
                and parity['source_hashes'] == source_hashes()
                and parity['parity_source_sha256'] == sha256(Path(__file__).with_name('inventory_flow_parity.py'))):
            raise ValueError('Flow GPU native-core parity is not certified for these inputs and source bytes')
    native_paths = [Path(p) for p in admission['native_smoke_results']]
    if len(native_paths) != 2:
        raise ValueError('Two same-policy/key native smokes with different execution chunks are required')
    native_checks = []
    for path in native_paths:
        summary = json.loads(path.read_text())
        provenance = json.loads(path.with_name('provenance.json').read_text())
        if not (summary['status'] == 'smoke_complete' and summary['reportable'] is False
                and summary['episodes'] >= 4 and summary['horizon'] >= 16
                and provenance['backend'] == 'gpu'
                and provenance['evaluator_source_sha256'] == sha256(Path(__file__).with_name('inventory_eval.py'))):
            raise ValueError('Native smoke has not passed the current evaluator checks')
        native_checks.append((provenance, path.with_name('episodes.jsonl').read_text()))
    left, right = native_checks
    if not (left[0]['policy_checkpoint_sha256'] == right[0]['policy_checkpoint_sha256']
            and left[0]['key_array_sha256'] == right[0]['key_array_sha256']
            and left[0]['execution_chunk'] != right[0]['execution_chunk'] and left[1] == right[1]):
        raise ValueError('Native evaluator is not invariant to chunk size for the same policy and keys')
    estimate = float(admission['estimated_campaign_seconds'])
    if not math.isfinite(estimate) or not 0 < estimate <= args.max_hours * 3600:
        raise ValueError('Measured campaign estimate exceeds the admitted wall-clock budget')
    return admission


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flow-wm', type=Path, required=True)
    p.add_argument('--jepa-wm', type=Path, required=True)
    p.add_argument('--jepa-sha256', required=True)
    p.add_argument('--reset-bank', type=Path, required=True)
    p.add_argument('--reset-bank-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--max-hours', type=float, default=5.0)
    p.add_argument('--deadline-utc', required=True)
    p.add_argument('--admission', type=Path)
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    if not 0 < a.max_hours <= 5:
        raise ValueError('Campaign wall-clock limit must be in (0,5] hours')
    if a.out.exists():
        raise FileExistsError(f'Campaign path already exists: {a.out}')
    for path, expected in [(a.flow_wm, FLOW_SHA256), (a.jepa_wm, a.jepa_sha256),
                           (a.reset_bank, a.reset_bank_sha256)]:
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f'Input identity mismatch: {path}: {actual}')
    jobs = jobs_for(a)
    probe_admission = validate_jepa_bundle(a.jepa_wm)
    external_deadline = parse_deadline(a.deadline_utc)
    deadline = min(external_deadline, time.time() + a.max_hours * 3600)
    if deadline <= time.time():
        raise ValueError('Deadline is in the past')
    if not a.plan_only and a.admission is None:
        raise ValueError('--admission is required before launching any GPU process')
    admission = validate_admission(a.admission, a) if a.admission else None
    if admission and time.time() + admission['estimated_campaign_seconds'] > deadline:
        raise ValueError('Measured campaign estimate cannot finish before the admitted deadline')
    a.out.mkdir(parents=True)
    (a.out / 'logs').mkdir()
    campaign = {'protocol': 'native-inventory-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'model_seeds': [0], 'policy_seeds': [0, 1, 2], 'families': ['flow', 'jepa'],
                'actual_training_transitions': 4_998_400, 'nominal_training_transitions': 5_000_000,
                'evaluation_episodes': 256, 'evaluation_seed': 123, 'native_time_limit': 10000,
                'imagined_time_limit': 1024, 'reset_bank_sha256': a.reset_bank_sha256,
                'sources': source_hashes(), 'deadline_unix': deadline, 'max_hours': a.max_hours,
                'gpu': a.gpu, 'jobs': jobs, 'admission': admission,
                'jepa_probe_admission': probe_admission,
                'selection': 'All six fixed final policies retained, no checkpoint or outcome selection',
                'reset_pairing': 'Candidate contexts paired at absolute training positions; accepted resets may differ'}
    write_json(a.out / 'campaign.json', campaign)
    if a.plan_only:
        print(json.dumps(campaign, indent=2), flush=True)
        return
    def terminated(signum, frame):
        raise KeyboardInterrupt(f'Queue received signal {signum}; terminating its child process group')
    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu),
               XLA_PYTHON_CLIENT_MEM_FRACTION='.85', PYTHONUNBUFFERED='1')
    lock_path = Path('/tmp') / f'tinycraft-inventory-gpu-{a.gpu}.lock'
    statuses = []
    paired_training = {}
    common_eval_keys = None
    # Cooperative advisory lock plus actual occupancy check. The queue owns this GPU until exit.
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_uuid = assert_gpu_idle(a.gpu)
        write_json(a.out / 'gpu_ownership.json', {'pid': os.getpid(), 'gpu_uuid': gpu_uuid,
                   'lock': str(lock_path), 'start_unix': time.time(), 'deadline_unix': deadline})
        for job in jobs:
            if (a.out / 'STOP').exists():
                raise RuntimeError('STOP requested between stages; existing artifacts retained')
            if source_hashes() != campaign['sources']:
                raise RuntimeError('Campaign source bytes changed after admission')
            status = {'name': job['name'], 'start_unix': time.time(), 'status': 'training'}
            statuses.append(status)
            write_json(a.out / 'campaign_status.json', statuses)
            print('[start] ' + job['name'], flush=True)
            try:
                assert_gpu_idle(a.gpu)
                rc = run_process(job['train_command'], a.out / 'logs' / (job['name'] + '_train.log'), env, deadline)
                if source_hashes() != campaign['sources']:
                    raise RuntimeError('Source bytes changed during the policy-training child')
                status['train_returncode'] = rc
                final_path = Path(job['out']) / 'final.json'
                final = json.loads(final_path.read_text()) if final_path.exists() else {}
                if rc or not final.get('budget_complete') or final.get('env_steps') != 4_998_400:
                    raise RuntimeError(f"{job['name']} failed or did not complete its fixed training budget")
                pair_signature = (final['initial_actor_sha256'], final['training_key_arrays_sha256'])
                seed = job['policy_seed']
                if seed in paired_training and paired_training[seed] != pair_signature:
                    raise RuntimeError('Paired policy initialization or training candidate RNG schedule differs')
                paired_training[seed] = pair_signature
                status.update(status='native_evaluation', final_policy_sha256=final['checkpoint_sha256'])
                write_json(a.out / 'campaign_status.json', statuses)
                assert_gpu_idle(a.gpu)
                rc = run_process(job['eval_command'], a.out / 'logs' / (job['name'] + '_eval.log'), env, deadline)
                if source_hashes() != campaign['sources']:
                    raise RuntimeError('Source bytes changed during the native-evaluation child')
                status['eval_returncode'] = rc
                if rc:
                    raise RuntimeError(f"{job['name']} native evaluation failed")
                native_dir = Path(job['out']) / 'native'
                summary = json.loads((native_dir / 'summary.json').read_text())
                provenance = json.loads((native_dir / 'provenance.json').read_text())
                if not (summary['status'] == 'complete' and summary['reportable']
                        and summary['episodes'] == 256 and summary['horizon'] == 10000
                        and summary['eval_seed'] == 123
                        and summary['policy_checkpoint_sha256'] == final['checkpoint_sha256']):
                    raise RuntimeError('Native evaluator did not certify the prescribed fixed-final contract')
                if common_eval_keys is not None and provenance['key_array_sha256'] != common_eval_keys:
                    raise RuntimeError('Native episode keys differ across policy evaluations')
                common_eval_keys = provenance['key_array_sha256']
                status['native_summary_sha256'] = sha256(native_dir / 'summary.json')
                status.update(status='complete', finish_unix=time.time())
                write_json(a.out / 'campaign_status.json', statuses)
                print('[complete] ' + job['name'], flush=True)
            except BaseException as error:
                status.update(status='failed_or_incomplete', finish_unix=time.time(), error=repr(error))
                write_json(a.out / 'campaign_status.json', statuses)
                raise
    if source_hashes() != campaign['sources']:
        raise RuntimeError('Source bytes changed before final campaign completion')
    write_json(a.out / 'complete.json', {'n_complete': len(statuses), 'finish_unix': time.time(),
               'sources_unchanged': source_hashes() == campaign['sources']})


if __name__ == '__main__':
    main()
