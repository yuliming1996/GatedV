"""Bounded orchestration checks; no scientific training or downloads."""
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reproduce', ROOT / 'tools/reproduce.py')
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


class ReproductionTests(unittest.TestCase):
    def arguments(self, root, study='smoke', *extra):
        return reproduce.normalize(reproduce.parser().parse_args([
            '--study', study, '--output', str(root / 'output'),
            '--data-root', str(root / 'data'), *extra]))

    def fake_evaluation(self, root):
        args = self.arguments(root)
        cohort = reproduce.make_cohort(args.study, args.device)
        args.output.mkdir()
        reproduce.write_json(args.output / 'cohort.json', cohort)
        seal = root / 'seal.json'
        reproduce.write_json(seal, {'synthetic': True})
        directory = args.output / 'evaluation'
        directory.mkdir()
        binding = {'cohort_sha256': reproduce.digest(args.output / 'cohort.json'),
                   'seal_sha256': reproduce.digest(seal), 'source_sha256': reproduce.source_hashes(),
                   'environment': {'device': 'cpu'}}
        jobs, results, artifacts = [], {}, {}
        for item in cohort['runs']:
            cfg = item['config']; name = Path(item['path']).name
            run = args.output / item['path']; run.mkdir(parents=True)
            result = {'identity': {'environment': binding['environment']}, 'artifacts': {'final.pt': 'checkpoint-sha'}}
            reproduce.write_json(run / 'result.json', result)
            (directory / (name + '.npz')).write_bytes(b'placeholder-payload')
            summary = {'status': 'complete', 'arm': cfg['arm'], 'seed': cfg['seed'], 'binding': binding,
                       'checkpoint_sha256': 'checkpoint-sha', 'training_result_sha256': reproduce.digest(run / 'result.json'),
                       'arrays_sha256': reproduce.digest(directory / (name + '.npz')), 'nll': 3., 'ppl': math.exp(3.)}
            reproduce.write_json(directory / (name + '.json'), summary)
            jobs.append({'arm': cfg['arm'], 'seed': cfg['seed'], 'label': name})
            results[name] = result
            for suffix in ('.json', '.npz'):
                artifacts[name + suffix] = reproduce.digest(directory / (name + suffix))
        reproduce.write_json(directory / 'done.json', {'status': 'complete', 'binding': binding,
                             'jobs': jobs, 'artifacts_sha256': artifacts})
        return directory, args, cohort, seal, results

    def test_formal_cohorts_preserve_every_scientific_field_and_order(self):
        for study, count, steps in [('small', 10, 32768), ('medium', 6, 327680)]:
            original = reproduce.read_json(ROOT / 'configs' / (study + '_cohort.json'))
            actual = reproduce.make_cohort(study, 'cpu')
            self.assertEqual(len(actual['runs']), count)
            for prior, row in zip(original['runs'], actual['runs']):
                self.assertEqual(row['config'], dict(prior['config'], device='cpu'))
                self.assertEqual(row['config']['steps'], steps)
                self.assertFalse(Path(row['path']).is_absolute())
                self.assertNotIn('..', Path(row['path']).parts)

    def test_formal_cohort_rejects_missing_or_wrong_paired_seed(self):
        original = reproduce.read_json(ROOT / 'configs/small_cohort.json')
        for variant in ('missing', 'wrong_seed'):
            changed = json.loads(json.dumps(original))
            if variant == 'missing':
                changed['runs'].pop()
            else:
                changed['runs'][0]['config']['seed'] = 999
            with mock.patch.object(reproduce, 'read_json', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'every published paired seed'):
                    reproduce.make_cohort('small', 'cpu')

    def test_formal_cohort_rejects_shortened_budget_and_changed_defaults(self):
        original = reproduce.read_json(ROOT / 'configs/medium_cohort.json')
        for field, value in [('steps', 4), ('lr', 0.001), ('batch', 4), ('context', 128),
                             ('eval_every', 1024), ('width', 128)]:
            changed = json.loads(json.dumps(original))
            changed['runs'][0]['config'][field] = value
            with mock.patch.object(reproduce, 'read_json', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'full-budget confirmation contract'):
                    reproduce.make_cohort('medium', 'cpu')

    def test_dry_run_without_site_packages_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'absent_output'
            data = Path(directory) / 'absent_data'
            done = subprocess.run([sys.executable, '-B', '-S', str(ROOT / 'tools/reproduce.py'),
                                   '--study', 'small', '--dry-run', '--output', str(output),
                                   '--data-root', str(data)], capture_output=True, text=True, check=True)
            plan = json.loads(done.stdout)
            self.assertEqual(len(plan['runs']), 10)
            self.assertFalse(output.exists())
            self.assertFalse(data.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_help_does_not_require_training_dependencies(self):
        subprocess.run([sys.executable, '-B', '-S', str(ROOT / 'tools/reproduce.py'), '--help'],
                       capture_output=True, text=True, check=True)

    def test_bad_resume_does_not_create_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'existing reproduction.json'):
                reproduce.main(['--study', 'smoke', '--resume', '--output', str(root / 'missing')])
            self.assertEqual(list(root.iterdir()), [])

    def test_existing_data_is_verified_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory), 'small', '--download')
            args.data_root.mkdir()
            command = reproduce.preparation_command(args)
            self.assertIn('--verify-only', command)
            self.assertNotIn('--download', command)

    def test_cuda_alias_uses_explicit_device_index(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory), 'medium', '--device', 'cuda')
            self.assertEqual(args.device, 'cuda:0')

    def test_predeclared_identity_rejects_scientific_change(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            cohort = reproduce.make_cohort(args.study, args.device)
            with reproduce.coordinator_lock(args.output):
                reproduce.declare(args, cohort)
                args.resume = True
                changed = reproduce.make_cohort(args.study, args.device)
                changed['runs'][0]['config']['steps'] += 1
                with self.assertRaisesRegex(ValueError, 'changed'):
                    reproduce.declare(args, changed)

    def test_coordinator_lock_prevents_duplicate_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with reproduce.coordinator_lock(output):
                with self.assertRaisesRegex(RuntimeError, 'Another coordinator'):
                    with reproduce.coordinator_lock(output):
                        self.fail('Second coordinator acquired lock')

    def test_same_cohort_can_move_without_absolute_path_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            cohort = reproduce.make_cohort(args.study, args.device)
            with reproduce.coordinator_lock(args.output):
                reproduce.declare(args, cohort)
            original = args.output
            args.output = Path(directory) / 'moved'
            original.rename(args.output)
            args.resume = True
            with reproduce.coordinator_lock(args.output):
                reproduce.declare(args, cohort)
            saved = (args.output / 'reproduction.json').read_text()
            self.assertNotIn(str(Path(directory)), saved)

    def test_stop_forwards_only_to_owned_live_child(self):
        runner = reproduce.Runner()
        runner.child = mock.Mock()
        runner.child.poll.return_value = None
        runner.child.pid = 12345
        with mock.patch.object(reproduce.os, 'killpg') as terminate:
            runner.request_stop()
            terminate.assert_called_once_with(12345, signal.SIGTERM)
        self.assertTrue(runner.stopping)
        self.assertEqual(runner.run(['never-launch']), 75)

    def test_stop_terminates_owned_child_and_descendant(self):
        # Real processes, all synthetic and CPU-idle; covers the prepare -> curl
        # descendant shape without downloading data or importing PyTorch.
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'processes.json'
            descendant = 'import time; time.sleep(60)'
            child = ('import json,os,subprocess,sys,time; '
                     f'p=subprocess.Popen([sys.executable,"-c",{descendant!r}]); '
                     f'open({str(marker)!r},"w").write(json.dumps([os.getpid(),p.pid])); '
                     'p.wait()')
            coordinator = (
                'import importlib.util,signal,sys; '
                f's=importlib.util.spec_from_file_location("reproduce",{str(ROOT / "tools/reproduce.py")!r}); '
                'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);r=m.Runner(); '
                'signal.signal(signal.SIGTERM,r.request_stop); '
                f'sys.exit(r.run([sys.executable,"-B","-c",{child!r}]))')
            process = subprocess.Popen([sys.executable, '-B', '-c', coordinator])
            owned = []
            try:
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(marker.exists(), 'Synthetic child failed to start')
                owned = json.loads(marker.read_text())
                self.assertNotEqual(os.getpgid(owned[0]), os.getpgrp())
                self.assertEqual(os.getpgid(owned[0]), os.getpgid(owned[1]))
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=5), 75)
                # An adopted zombie is already terminated; Linux init may reap
                # it later. Require no running/sleeping descendant remains.
                def running(pid):
                    state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)],
                                           capture_output=True, text=True).stdout.strip()
                    return bool(state) and not state.startswith('Z')
                deadline = time.monotonic() + 3
                while any(running(pid) for pid in owned) and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertFalse(any(running(pid) for pid in owned))
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=5)

    def test_paused_endpoint_never_opens_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            cohort = reproduce.make_cohort(args.study, args.device)
            runner = mock.Mock(stopping=False)
            runner.run.side_effect = [0, 75]  # Device passes; first endpoint checkpoints and pauses.
            with reproduce.coordinator_lock(args.output), mock.patch.object(reproduce, 'smoke_data'), \
                    mock.patch.object(reproduce, 'verify_endpoints') as gate, \
                    mock.patch.object(reproduce, 'evaluate_cohort') as evaluation:
                self.assertEqual(reproduce.execute(args, cohort, runner), 75)
            self.assertEqual(runner.run.call_count, 2)
            gate.assert_not_called()
            evaluation.assert_not_called()

    def test_child_failure_has_no_automatic_retry(self):
        runner = reproduce.Runner()
        child = mock.Mock()
        child.wait.return_value = 9
        with mock.patch.object(reproduce.subprocess, 'Popen', return_value=child) as launch:
            with self.assertRaisesRegex(RuntimeError, 'No automatic retry'):
                runner.run(['python', '-B', 'example.py'])
        launch.assert_called_once()

    def test_incomplete_budget_fails_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            cohort = reproduce.make_cohort(args.study, args.device)
            row = cohort['runs'][0]
            output = args.output / row['path']
            output.mkdir(parents=True)
            reproduce.write_json(output / 'result.json', {
                'status': 'complete', 'identity': {'config': row['config']},
                'steps': row['config']['steps'] - 1, 'tokens': 1})
            with self.assertRaisesRegex(ValueError, 'Incomplete training budget'):
                reproduce.verify_endpoints(args, cohort)

    def test_completed_evaluation_reuse_checks_payload_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self.fake_evaluation(Path(directory))
            reproduce.verify_evaluation(*values)
            (values[0] / 'baseline_s17.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Evaluation artifact changed'):
                reproduce.verify_evaluation(*values)

    def test_completed_evaluation_requires_entire_declared_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self.fake_evaluation(Path(directory))
            path = values[0] / 'done.json'
            done = reproduce.read_json(path)
            done['jobs'].pop()
            reproduce.write_json(path, done)
            with self.assertRaisesRegex(ValueError, 'complete declared cohort'):
                reproduce.verify_evaluation(*values)


if __name__ == '__main__':
    unittest.main()
