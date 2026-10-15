"""Prepare data, train a complete paired cohort, then evaluate and summarize it.

small/medium repeat the published confirmation recipes at their selected learning
rates. smoke is a tiny synthetic CPU integration check, not a scientific result.
Only the Python standard library is imported for --help and --dry-run.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PAUSED = 75


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (
        _ for _ in ()).throw(ValueError('Nonfinite JSON value: ' + value)))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def source_hashes():
    return {path.name: digest(path) for path in sorted((ROOT / 'code').glob('*.py'))}


def formal_contract(study):
    """Published confirmation contract, checked before downloads or output writes."""
    shared = dict(vocab=8192, context=256, qk_norm=False, projection_hidden=0,
                  batch=8, lr_schedule='cosine', log_every=1, threads=2, device='cuda:0')
    if study == 'small':
        return dict(shared, width=128, layers=6, heads=4, hidden=384, steps=32768,
                    lr=0.0048, warmup=512, eval_tokens=131072, eval_every=2048,
                    save_every=256), set(range(20261201, 20261206))
    if study == 'medium':
        return dict(shared, width=384, layers=12, heads=6, hidden=1152, steps=327680,
                    lr=0.0006, warmup=5120, eval_tokens=524288, eval_every=20480,
                    save_every=1024), {20270211, 20270212, 20270213}
    raise ValueError('Unknown formal study')


def make_cohort(study, device):
    if study == 'smoke':
        base = dict(vocab=32, width=24, layers=1, heads=3, hidden=48,
                    context=8, qk_norm=False, projection_hidden=0, batch=2,
                    steps=4, seed=17, lr=0.001, warmup=1, lr_schedule='cosine',
                    eval_tokens=32, eval_every=2, save_every=2, log_every=1,
                    threads=1, device=device)
        runs = [{'config': dict(base, arm=arm)} for arm in ('baseline', 'v_silu')]
    else:
        original = read_json(ROOT / 'configs' / (study + '_cohort.json'))
        require(original['schema'] == 'portable_cohort_v1', 'Unsupported cohort schema')
        runs = original['runs']
        template, seeds = formal_contract(study)
        pairs = {(arm, seed) for arm in ('baseline', 'v_silu') for seed in seeds}
        require(len(runs) == len(pairs) and
                {(row['config']['arm'], row['config']['seed']) for row in runs} == pairs,
                'Formal cohort must contain every published paired seed exactly once')
        for row in runs:
            cfg = row['config']
            require(cfg == dict(template, arm=cfg['arm'], seed=cfg['seed']),
                    'Formal recipe differs from the published full-budget confirmation contract')
    cohort = {'schema': 'portable_cohort_v1', 'runs': []}
    seen = set()
    for item in runs:
        cfg = dict(item['config'], device=device)
        name = f"{cfg['arm']}_s{cfg['seed']}"
        require(name not in seen and cfg['arm'] in ('baseline', 'v_silu'), 'Unexpected cohort')
        seen.add(name)
        cohort['runs'].append({'path': 'runs/' + name, 'config': cfg})
    return cohort


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--study', choices=('small', 'medium', 'smoke'), required=True)
    result.add_argument('--download', action='store_true', help='Download missing pinned public data shards')
    result.add_argument('--data-root', type=Path, help='Prepared data root; default data/paper_STUDY')
    result.add_argument('--raw', type=Path, help='Public parquet cache; default data/raw')
    result.add_argument('--output', type=Path, help='New cohort output; default outputs/reproduction_STUDY')
    result.add_argument('--device', help='Default cuda:0 for full recipes; cpu for smoke')
    result.add_argument('--resume', action='store_true', help='Explicitly resume the same cohort/source/data/environment')
    result.add_argument('--dry-run', action='store_true', help='Print the full plan without imports, downloads or writes')
    return result


def normalize(args):
    args.device = args.device or ('cpu' if args.study == 'smoke' else 'cuda:0')
    if args.device == 'cuda':
        args.device = 'cuda:0'
    require(args.device == 'cpu' or args.device == 'cuda' or
            (args.device.startswith('cuda:') and args.device[5:].isdigit()), 'Use a CPU or CUDA device')
    args.data_root = (args.data_root or ROOT / 'data' / ('paper_' + args.study)).resolve()
    args.raw = (args.raw or ROOT / 'data/raw').resolve()
    args.output = (args.output or ROOT / 'outputs' / ('reproduction_' + args.study)).resolve()
    require(args.output != args.data_root and args.output not in args.data_root.parents
            and args.data_root not in args.output.parents, 'Output and prepared data roots must not overlap')
    require(args.study != 'smoke' or not args.download, 'Smoke uses synthetic data and never downloads')
    return args


def python_command(script, *arguments):
    return [sys.executable, '-B', str(ROOT / script), *map(str, arguments)]


def preparation_command(args):
    command = python_command('tools/prepare_data.py', '--study', args.study,
                             '--output', args.data_root, '--raw', args.raw)
    if args.data_root.exists():
        command.append('--verify-only')
    elif args.download:
        command.append('--download')
    return command


def device_command(args):
    # Run in an isolated child so --help/--dry-run remain standard-library only.
    code = ('import sys; sys.path.insert(0, sys.argv[1]); '
            'from common import configure; configure(sys.argv[2]); '
            'print("Numerical policy/device preflight passed: " + sys.argv[2], flush=True)')
    return [sys.executable, '-B', '-c', code, str(ROOT / 'code'), args.device]


def training_command(args, item):
    name = Path(item['path']).name
    run = args.output / item['path']
    command = python_command('code/train.py', '--config', args.output / 'recipes' / (name + '.json'),
                             '--data', args.data_root / (args.study + '_corpus'), '--output', run)
    if (run / 'manifest.json').exists():
        require(args.resume, 'Existing endpoint requires --resume')
        command.append('--resume')
    return command


def dry_plan(args, cohort):
    # Reading configs and existence checks are intentional; no mkdir or runtime import.
    rows = [dict(arm=item['config']['arm'], seed=item['config']['seed'],
                 updates=item['config']['steps'], peak_lr=item['config']['lr'],
                 training_targets=item['config']['steps'] * item['config']['batch'] * item['config']['context'])
            for item in cohort['runs']]
    return {'study': args.study, 'synthetic': args.study == 'smoke', 'device': args.device,
            'output': str(args.output), 'data_root': str(args.data_root),
            'data_preparation': 'Generate fixed synthetic arrays' if args.study == 'smoke' else preparation_command(args),
            'runs': rows, 'training_commands': [training_command(args, item) for item in cohort['runs']],
            'after_all_endpoints': ['Verify all final endpoints', 'Evaluate the complete paired holdout cohort',
                                    'Recompute paired statistics' if args.study != 'smoke' else 'Describe synthetic smoke results'],
            'scope': 'Confirmation recipes only; no learning-rate search, GLU experiment or timing benchmark.'}


class Runner:
    """Stop only the current child's owned session, including download descendants."""
    def __init__(self):
        self.child = None
        self.stopping = False

    def request_stop(self, *_):
        self.stopping = True
        if self.child is not None and self.child.poll() is None:
            try:
                # start_new_session makes the child's PID its new process-group
                # identity. While the direct child is alive/unreaped this PID
                # cannot be reused by an unrelated process group.
                os.killpg(self.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def run(self, command):
        if self.stopping:
            return PAUSED
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        self.child = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
        # Cover a stop arriving between the initial check and child assignment.
        if self.stopping:
            self.request_stop()
        code = self.child.wait()
        self.child = None
        if self.stopping or code == PAUSED:
            self.stopping = True
            return PAUSED
        if code != 0:
            raise RuntimeError(f'Child exited {code}; inspect its output before explicit --resume. No automatic retry.')
        return 0


@contextmanager
def coordinator_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.coordinator.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another coordinator holds this output lock') from exc
        yield


def identity(args, cohort):
    files = [Path('tools/reproduce.py')]
    if args.study != 'smoke':
        files += [Path('tools/prepare_data.py'), Path('reproduction/data_plan.json'),
                  Path('configs') / (args.study + '_cohort.json'),
                  Path('configs') / (args.study + '_holdout.json')]
    return {'schema': 'reproduction_v1', 'study': args.study, 'synthetic': args.study == 'smoke',
            'cohort': cohort, 'source_sha256': source_hashes(),
            'input_sha256': {path.as_posix(): digest(ROOT / path) for path in files}}


def declare(args, cohort):
    expected = identity(args, cohort)
    manifest = args.output / 'reproduction.json'
    if manifest.exists():
        require(args.resume, 'Existing cohort requires explicit --resume')
        require(read_json(manifest)['identity'] == expected, 'Cohort/source/configuration changed; use a new output')
        require(read_json(args.output / 'cohort.json') == cohort, 'Declared cohort changed')
        for item in cohort['runs']:
            require(read_json(args.output / 'recipes' / (Path(item['path']).name + '.json')) == item['config'],
                    'Declared recipe changed')
    else:
        require(not args.resume, '--resume requires an existing reproduction.json')
        require(not any(p.name != '.coordinator.lock' for p in args.output.iterdir()),
                'New cohort requires an empty output directory')
        (args.output / 'recipes').mkdir()
        for item in cohort['runs']:
            write_json(args.output / 'recipes' / (Path(item['path']).name + '.json'), item['config'])
        write_json(args.output / 'cohort.json', cohort)
        write_json(manifest, {'identity': expected, 'created_unix': time.time(),
                             'scope': 'Synthetic integration check; not research' if args.study == 'smoke'
                             else 'Full confirmation cohort at the published selected learning rate; not an LR search'})
    return expected


def smoke_data(args):
    import numpy as np
    root = args.data_root
    if root.exists():
        record = read_json(root / 'synthetic.json')
        require(record['schema'] == 'gatedv_synthetic_smoke_v1', 'Not this runner\'s synthetic dataset')
        for name, expected in record['files_sha256'].items():
            require(digest(root / name) == expected, 'Synthetic data changed: ' + name)
        return
    temporary = root.with_name(root.name + '.partial')
    require(not temporary.exists(), f'Partial synthetic data exists; inspect {temporary}')
    corpus, holdout = temporary / 'smoke_corpus', temporary / 'smoke_holdout'
    corpus.mkdir(parents=True); holdout.mkdir()
    rng = np.random.Generator(np.random.PCG64(2026101501))
    for split, folder, documents in [('train', corpus, 8), ('dev', corpus, 4), ('holdout', holdout, 4)]:
        tokens = rng.integers(4, 32, size=(documents, 25), dtype=np.uint16)
        tokens[:, -1] = 3
        bounds = np.array([(25*i, 25*(i+1)) for i in range(documents)], dtype=np.int64)
        np.save(folder / f'tokens_{split}.npy', tokens.flatten(), allow_pickle=False)
        np.save(folder / f'docs_{split}.npy', bounds, allow_pickle=False)
        if split == 'holdout':
            np.save(folder / 'eval_offsets.npy', bounds[:, 0], allow_pickle=False)
            np.save(folder / 'eval_document_ids.npy', np.arange(documents, dtype=np.int64), allow_pickle=False)
    seal = {'context': 8, 'documents': 4, 'eval_tokens': 32, 'eos': 3,
            'files_sha256': {p.name: digest(p) for p in sorted(holdout.glob('*.npy'))},
            'scope': 'Synthetic integration check; not a paper dataset'}
    write_json(temporary / 'holdout_seal.json', seal)
    write_json(temporary / 'synthetic.json', {'schema': 'gatedv_synthetic_smoke_v1',
               'files_sha256': {p.relative_to(temporary).as_posix(): digest(p)
                                for p in sorted(temporary.rglob('*')) if p.is_file()}})
    temporary.rename(root)


def verify_endpoints(args, cohort):
    """Standard-library checks in addition to the trainer/evaluator's stronger gate."""
    results = {}
    for item in cohort['runs']:
        run = args.output / item['path']; cfg = item['config']
        value = read_json(run / 'result.json')
        require(value['status'] == 'complete' and value['identity']['config'] == cfg,
                'Incomplete or changed endpoint')
        require(value['steps'] == cfg['steps'] and value['tokens'] == cfg['steps'] * cfg['batch'] * cfg['context'],
                'Incomplete training budget')
        require(value['identity']['source_sha256'] == source_hashes(), 'Endpoint source changed')
        require(set(value['artifacts']) == {'final.pt', 'dev_final.json', 'dev_final.npz'}, 'Incomplete final artifacts')
        for name, sha in value['artifacts'].items():
            require(digest(run / name) == sha, 'Endpoint artifact changed: ' + name)
        results[Path(item['path']).name] = value
    return results


def verify_evaluation(directory, args, cohort, seal, results):
    done = read_json(directory / 'done.json')
    binding = done['binding']
    require(done['status'] == 'complete' and binding['cohort_sha256'] == digest(args.output / 'cohort.json')
            and binding['seal_sha256'] == digest(seal) and binding['source_sha256'] == source_hashes(),
            'Evaluation completion/binding differs')
    expected_jobs = [{'arm': item['config']['arm'], 'seed': item['config']['seed'],
                      'label': Path(item['path']).name} for item in cohort['runs']]
    require(done['jobs'] == expected_jobs, 'Evaluation does not cover exactly the complete declared cohort')
    expected_files = {row['label'] + suffix for row in expected_jobs for suffix in ('.json', '.npz')}
    require(set(done['artifacts_sha256']) == expected_files, 'Evaluation artifact set differs')
    for name, sha in done['artifacts_sha256'].items():
        require(digest(directory / name) == sha, 'Evaluation artifact changed: ' + name)
    for row in expected_jobs:
        name = row['label']; summary = read_json(directory / (name + '.json')); result = results[name]
        require(summary['status'] == 'complete' and summary['arm'] == row['arm'] and summary['seed'] == row['seed']
                and summary['binding'] == binding and binding['environment'] == result['identity']['environment'],
                'Evaluation identity/environment differs')
        require(summary['checkpoint_sha256'] == result['artifacts']['final.pt']
                and summary['training_result_sha256'] == digest(args.output / 'runs' / name / 'result.json')
                and summary['arrays_sha256'] == done['artifacts_sha256'][name + '.npz'], 'Evaluation endpoint binding differs')
        require(math.isfinite(summary['nll']) and math.isclose(summary['ppl'], math.exp(summary['nll']), rel_tol=1e-12),
                'Evaluation contains inconsistent metrics')
    return done


def evaluate_cohort(args, cohort, seal, results, runner):
    root = args.output / 'evaluation'
    root.mkdir(exist_ok=True)
    attempts = sorted(root.glob('attempt_*'))
    complete = [path for path in attempts if (path / 'done.json').exists()]
    if complete:
        require(len(complete) == 1, 'Multiple complete evaluations require manual inspection')
        verify_evaluation(complete[0], args, cohort, seal, results)
        return complete[0]
    # Retain interrupted/failed evaluations. Never silently overwrite their arrays.
    attempt = root / f'attempt_{len(attempts) + 1:03d}'
    require(not attempt.exists(), 'Evaluation attempt naming conflict')
    command = python_command('code/evaluate.py', '--cohort', args.output / 'cohort.json',
                             '--holdout', args.data_root / (args.study + '_holdout'), '--seal', seal,
                             '--output', attempt, '--device', args.device)
    if runner.run(command) == PAUSED:
        return None
    verify_evaluation(attempt, args, cohort, seal, results)
    return attempt


def smoke_summary(evaluation):
    import numpy as np
    rows = []
    done = read_json(evaluation / 'done.json')
    for job in done['jobs']:
        summary = read_json(evaluation / (job['label'] + '.json'))
        with np.load(evaluation / (job['label'] + '.npz'), allow_pickle=False) as arrays:
            nll = math.fsum(map(float, arrays['loss_sums'])) / int(arrays['counts'].sum())
        require(math.isclose(nll, summary['nll'], abs_tol=1e-12, rel_tol=0), 'Smoke loss-array reduction differs')
        rows.append({'arm': job['arm'], 'seed': job['seed'], 'nll': nll, 'ppl': math.exp(nll)})
    return {'study': 'smoke', 'synthetic': True, 'scope': 'Technical integration check only; no scientific interpretation',
            'evaluation_done_sha256': digest(evaluation / 'done.json'), 'rows': rows}


def verified_completion(args, cohort, results, seal):
    path = args.output / 'complete.json'
    if not path.exists():
        return None
    complete = read_json(path)
    require(complete['status'] == ('COMPLETE_SYNTHETIC_SMOKE' if args.study == 'smoke' else 'COMPLETE_CONFIRMATION_REPRODUCTION'),
            'Invalid completion status')
    for name, sha in complete['artifacts_sha256'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid completion artifact path')
        require(digest(args.output / relative) == sha, 'Completion artifact changed: ' + name)
    evaluation = args.output / complete['evaluation']
    verify_evaluation(evaluation, args, cohort, seal, results)
    require(read_json(args.output / complete['comparison'])['evaluation_done_sha256'] == digest(evaluation / 'done.json'),
            'Comparison references another evaluation')
    return complete


def execute(args, cohort, runner):
    declare(args, cohort)
    # Fail on an unavailable CUDA runtime before downloading or reconstructing data.
    if runner.run(device_command(args)) == PAUSED:
        return PAUSED
    if args.study == 'smoke':
        smoke_data(args)
    elif runner.run(preparation_command(args)) == PAUSED:
        return PAUSED
    seal = args.data_root / 'holdout_seal.json' if args.study == 'smoke' else ROOT / 'configs' / (args.study + '_holdout.json')
    for number, item in enumerate(cohort['runs'], start=1):
        if runner.stopping:
            return PAUSED
        print(f"Endpoint {number}/{len(cohort['runs'])}: {Path(item['path']).name}", flush=True)
        write_json(args.output / 'status.json', {'phase': 'training', 'endpoint': item['path'], 'number': number})
        if runner.run(training_command(args, item)) == PAUSED:
            return PAUSED
    results = verify_endpoints(args, cohort)
    if runner.stopping:
        return PAUSED
    complete = verified_completion(args, cohort, results, seal)
    if complete:
        write_json(args.output / 'status.json', {'phase': 'complete', 'completed_endpoints': len(results)})
        print('Already complete; endpoint, evaluation and comparison hashes verified.', flush=True)
        return 0
    write_json(args.output / 'status.json', {'phase': 'holdout_evaluation', 'completed_endpoints': len(results)})
    evaluation = evaluate_cohort(args, cohort, seal, results, runner)
    if evaluation is None or runner.stopping:
        return PAUSED
    analyses = args.output / 'analysis'
    analyses.mkdir(exist_ok=True)
    comparison = analyses / f'comparison_{len(list(analyses.glob("comparison_*.json"))) + 1:03d}.json'
    require(not comparison.exists(), 'Analysis attempt naming conflict')
    if args.study == 'smoke':
        write_json(comparison, smoke_summary(evaluation))
    else:
        command = python_command('code/analyze.py', '--evaluation', evaluation, '--study', args.study,
                                 '--domains', args.data_root / (args.study + '_holdout') / 'documents_holdout.jsonl',
                                 '--output', comparison)
        if runner.run(command) == PAUSED:
            return PAUSED
    artifacts = ['reproduction.json', 'cohort.json', comparison.relative_to(args.output).as_posix(),
                 (evaluation / 'done.json').relative_to(args.output).as_posix()]
    artifacts += [item['path'] + '/result.json' for item in cohort['runs']]
    complete = {'status': 'COMPLETE_SYNTHETIC_SMOKE' if args.study == 'smoke' else 'COMPLETE_CONFIRMATION_REPRODUCTION',
                'study': args.study, 'synthetic': args.study == 'smoke', 'completed_endpoints': len(results),
                'evaluation': evaluation.relative_to(args.output).as_posix(),
                'comparison': comparison.relative_to(args.output).as_posix(),
                'artifacts_sha256': {name: digest(args.output / name) for name in artifacts},
                'limits': ['Different hardware/software can change training outcomes; published improvements are not guaranteed.',
                           'This command does not reproduce learning-rate searches, GLU controls or timing measurements.']}
    write_json(args.output / 'complete.json', complete)
    write_json(args.output / 'status.json', {'phase': 'complete', 'completed_endpoints': len(results)})
    print(f'Complete. Comparison: {comparison}', flush=True)
    return 0


def main(argv=None):
    args = normalize(parser().parse_args(argv))
    cohort = make_cohort(args.study, args.device)
    if args.dry_run:
        print(json.dumps(dry_plan(args, cohort), indent=2))
        return 0
    # Refuse accidental output creation when a mistyped path is passed with resume.
    require(not args.resume or (args.output / 'reproduction.json').is_file(), '--resume requires an existing reproduction.json')
    runner = Runner()
    signal.signal(signal.SIGTERM, runner.request_stop)
    signal.signal(signal.SIGINT, runner.request_stop)
    with coordinator_lock(args.output):
        try:
            code = execute(args, cohort, runner)
            if code == PAUSED:
                write_json(args.output / 'status.json', {'phase': 'paused', 'exit_code': PAUSED,
                           'instruction': 'Inspect the last phase; repeat the same command with --resume when ready.'})
            return code
        except Exception as exc:
            # Keep each failed attempt; an explicit later resume never erases it.
            write_json(args.output / f'failure_{time.time_ns()}.json', {'type': type(exc).__name__, 'message': str(exc)})
            raise


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1)
