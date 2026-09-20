"""Independent CPU-only queue validation; no nvidia-smi, remote calls or fitting."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / 'studies/revision/run_last_only_queue.py'
REQUIRED = ('studies/revision/run_last_only_queue.py', 'studies/revision/last_only_objective.py',
            'studies/revision/last_only_probe.py', 'studies/revision/target_gradient.py', 'train.py',
            'probe.py', 'studies/redcube/env.py', 'studies/stoch/train_arms.sh', 'pyproject.toml')
PIN_SOURCES = ('studies/revision/full_history_audit.py', 'train.py',
               'research/full_history_audit_protocol.md')


def load_queue():
    spec = importlib.util.spec_from_file_location('independent_last_queue', QUEUE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.q = load_queue()
        self.temp = tempfile.TemporaryDirectory(prefix='last-only-independent-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.q.ROOT = self.root
        for name in set(REQUIRED + PIN_SOURCES + ('research/last_only_protocol.md',)):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('frozen fixture ' + name)
        self.data = self.root / 'data'
        data_hashes = {}
        for world in self.q.WORLDS:
            for split in ('train', 'eval'):
                name = f'{world}/{split}.npz'
                path = self.data / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
                data_hashes[name] = self.q.sha256(path)
        self.pin_path = self.root / 'pin/window_pin.json'
        self.pin_path.parent.mkdir()
        pin = {'protocol': 'frozen_full_history_audit_v1_window_pin', 'status': 'complete',
               'created_utc': '2026-09-19T00:00:00+00:00', 'windows_per_world': 8192,
               'sample_seed': 20260919,
               'sampler': 'numpy.Generator(PCG64(seed)).choice(valid_windows,8192,replace=False)',
               'sources_sha256': {p: self.q.sha256(self.root / p) for p in PIN_SOURCES}, 'data': {}}
        for world in self.q.WORLDS:
            artifact = self.pin_path.parent / f'{world}_windows.npz'
            artifact.write_bytes(world.encode())
            pin['data'][world] = {'sha256': data_hashes[f'{world}/eval.npz'],
                                  'artifact': artifact.name, 'artifact_sha256': self.q.sha256(artifact)}
        self.pin_path.write_text(json.dumps(pin))
        self.launch_path = self.root / 'launch.json'
        self.launch = {'protocol': 'last_only_objective_v1', 'planned_fits': 6,
                       'python_executable': os.path.abspath(sys.executable),
                       'worlds': list(self.q.WORLDS), 'model_seeds': [0, 1, 2],
                       'modes': ['last_only'], 'steps': 8500, 'deadline_utc': self.q.DEADLINE,
                       'source_sha256': {p: self.q.sha256(self.root / p) for p in REQUIRED},
                       'data_sha256': data_hashes, 'window_pin_sha256': self.q.sha256(self.pin_path),
                       'protocol_sha256': self.q.sha256(self.root / 'research/last_only_protocol.md')}
        self.launch_path.write_text(json.dumps(self.launch))
        self.out = self.root / 'campaign'
        self.calls = []
        self.behavior = None
        self.fail_at = None
        self.timeout_at = None

    def fake_popen(self, command, stdout, **kwargs):
        self.calls.append(command)
        self.assertTrue(kwargs['start_new_session'])
        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '1')
        index = len(self.calls)
        is_smoke = command[1] == '-c'
        is_train = is_smoke or command[2].endswith('last_only_objective')
        get = lambda key: Path(command[command.index(key) + 1])
        out = Path(command[-1]) if is_smoke else get('--out')
        out.mkdir(parents=True, exist_ok=False)
        if is_train:
            data = Path(command[-2]) if is_smoke else get('--data')
            steps = 500 if is_smoke else 8500
            checkpoint = out / 'model.npz'
            checkpoint.write_bytes(f'model {index}'.encode())
            meta = {'status': 'complete', 'completed_steps': steps,
                    'data_sha256': self.q.sha256(data), 'elapsed_s': 12,
                    'protocol': 'last_only_objective_v1', 'mode': 'last_only',
                    'fixed_checkpoint_steps': [0, 500, 1000, 2000, 4000, 4250, 8500],
                    'final_checkpoint_sha256': self.q.sha256(checkpoint),
                    'checkpoints': [{'path': 'model.npz', 'sha256': self.q.sha256(checkpoint)}]}
            (out / 'manifest.json').write_text(json.dumps(meta))
            if is_smoke:
                stdout.write('[train] 250/500 pred 1 sigreg 2 10.0s\n[train] 500/500 pred 1 sigreg 2 11.0s\n')
                stdout.flush()
        else:
            meta = {'status': 'complete', 'protocol': 'last_only_readouts_v1',
                    'sha256': {name: self.q.sha256(get('--' + name)) for name in ('model', 'train', 'eval')},
                    'mlp': {'steps': 4000}}
            (out / 'protocol.json').write_text(json.dumps(meta))
        if self.behavior:
            self.behavior(index)
        process = mock.Mock(pid=99999999)
        process.wait.return_value = 7 if index == self.fail_at else 0
        if index == self.timeout_at:
            process.wait.side_effect = subprocess.TimeoutExpired(command, 1)
        process.poll.return_value = 0
        return process

    def invoke(self, remaining=10000):
        now = self.q.datetime.fromisoformat(self.q.DEADLINE).timestamp() - remaining
        argv = ['queue', '--data-root', str(self.data), '--out', str(self.out),
                '--launch-manifest', str(self.launch_path), '--window-pin', str(self.pin_path), '--gpu', '1']
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(self.q.time, 'time', return_value=now), \
                mock.patch.object(self.q, 'idle_gpu'), mock.patch.object(self.q.fcntl, 'flock'), \
                mock.patch.object(self.q.subprocess, 'check_output', return_value='fake GPU identity'), \
                mock.patch.object(self.q.subprocess, 'Popen', side_effect=self.fake_popen), \
                mock.patch.object(self.q, 'terminate_child', side_effect=lambda: setattr(self.q, 'CHILD', None)), \
                contextlib.redirect_stdout(io.StringIO()):
            self.q.main()

    def test_exact_full_matrix_and_cli(self):
        self.invoke()
        self.assertEqual(len(self.calls), 14)
        train = [c for c in self.calls if len(c) > 2 and c[2] == 'studies.revision.last_only_objective']
        self.assertEqual([(c[c.index('--config') + 1], int(c[c.index('--seed') + 1])) for c in train],
                         [(w, s) for w in self.q.WORLDS for s in (0, 1, 2)])
        for command in train:
            self.assertEqual(command[command.index('--mode') + 1], 'last_only')
            self.assertEqual(set(command[3::2]), {'--config', '--data', '--seed', '--mode', '--out'})
        readout = [c for c in self.calls if len(c) > 2 and c[2] == 'studies.revision.last_only_probe']
        self.assertEqual(len(readout), 7)
        self.assertTrue(all(c[c.index('--model') + 1].endswith('/model.npz') for c in readout))
        self.assertEqual(json.loads((self.out / 'campaign.json').read_text())['status'], 'complete')

    def test_no_overwrite_even_empty(self):
        self.out.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'no overwrite'):
            self.invoke()
        self.assertEqual(self.calls, [])

    def test_reject_changed_archive_before_launch(self):
        (self.data / self.q.WORLDS[0] / 'train.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'archive differs'):
            self.invoke()
        self.assertEqual(self.calls, [])

    def test_reject_source_mutation_after_child(self):
        self.behavior = lambda _: (self.root / 'train.py').write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'frozen source changed'):
            self.invoke()
        self.assertEqual(len(self.calls), 1)

    def test_reject_data_mutation_even_with_preserved_stat(self):
        def mutate(_):
            path = self.data / self.q.WORLDS[0] / 'eval.npz'
            stat = path.stat()
            path.write_bytes(b'x' * stat.st_size)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.behavior = mutate
        with self.assertRaisesRegex(RuntimeError, 'archive changed during child'):
            self.invoke()
        self.assertEqual(len(self.calls), 1)

    def test_runtime_rejection_runs_no_registered_fit(self):
        self.invoke(remaining=1000)
        self.assertEqual(len(self.calls), 2)
        report = json.loads((self.out / 'campaign.json').read_text())
        self.assertEqual(report['status'], 'not_admitted_runtime')
        self.assertIn('runtime only', report['admission']['basis'])

    def test_child_failure_preserves_artifacts_without_retry(self):
        self.fail_at = 3
        with self.assertRaisesRegex(RuntimeError, 'no retries or seed replacement'):
            self.invoke()
        self.assertEqual(len(self.calls), 3)
        report = json.loads((self.out / 'campaign.json').read_text())
        self.assertEqual(report['jobs'][-1]['status'], 'failed')
        self.assertEqual(report['jobs'][-1]['returncode'], 7)
        self.assertTrue((self.out / self.q.WORLDS[0] / 'm0/last_only/model.npz').exists())

    def test_deadline_timeout_records_incomplete_and_stops(self):
        self.timeout_at = 1
        with self.assertRaisesRegex(RuntimeError, 'deadline stopped'):
            self.invoke()
        self.assertEqual(len(self.calls), 1)
        report = json.loads((self.out / 'campaign.json').read_text())
        self.assertEqual(report['status'], 'deadline_incomplete')
        self.assertEqual(report['jobs'][-1]['status'], 'deadline_timeout')
        self.assertTrue((self.out / 'smoke/last_only/model.npz').exists())

    def test_expired_deadline_rejects_before_any_child(self):
        with self.assertRaisesRegex(RuntimeError, 'deadline has already passed'):
            self.invoke(remaining=-1)
        self.assertEqual(self.calls, [])

    def test_reject_empty_pin_source_map(self):
        pin = json.loads(self.pin_path.read_text())
        pin['sources_sha256'] = {}
        self.pin_path.write_text(json.dumps(pin))
        self.launch['window_pin_sha256'] = self.q.sha256(self.pin_path)
        self.launch_path.write_text(json.dumps(self.launch))
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertEqual(self.calls, [])

    def test_projection_and_bad_timing(self):
        log = '[train] 250/500 pred 1 sigreg 2 10.0s\n[train] 500/500 pred 1 sigreg 2 11.0s\n'
        self.assertEqual(self.q.train_projection(log, {'elapsed_s': 12}), 44.0)
        for malformed in ('', log + log, log.replace('11.0s', '10.0s')):
            with self.assertRaises(RuntimeError):
                self.q.train_projection(malformed, {'elapsed_s': 12})

    def test_cleanup_exited_leader_with_term_ignoring_descendant(self):
        heartbeat = self.root / 'heartbeat'
        child_code = ("import pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                      "p=pathlib.Path(sys.argv[1]);\nwhile True:\n p.write_text(str(time.time())); time.sleep(.03)")
        leader_code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); time.sleep(.15)"
        leader = subprocess.Popen([sys.executable, '-c', leader_code, child_code, str(heartbeat)], start_new_session=True)
        try:
            leader.wait(timeout=3)
            self.q.CHILD = leader
            self.q.terminate_child()
            before = heartbeat.read_text()
            time.sleep(.15)
            self.assertEqual(before, heartbeat.read_text())
            self.assertIsNone(self.q.CHILD)
        finally:
            try:
                os.killpg(leader.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            leader.wait()


if __name__ == '__main__':
    unittest.main(verbosity=2)
