"""Train one declared endpoint on frozen pretokenized train/dev arrays.

The portable checkpoint schema is intentionally distinct from historical runs.
No holdout path or holdout payload is accepted by this program.
"""
import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import random
import signal
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F
from common import (MODEL_KEYS, ROOT, configure, cpu_tree, digest, environment,
                    load_recipe, parameter_digest, read_json, sync, write_json)
from data import Corpus, evaluate, learning_rate, save_eval
from model import ARMS, Config, LanguageModel

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


@contextmanager
def run_lock(output):
    with (output / '.run.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def calibration(path, cfg):
    if path is None:
        if cfg['arm'] in ('v_affine_match', 'v_silu_standardized'):
            raise ValueError('Mechanism controls require --calibration from train-only initialization')
        return None
    value = read_json(path)
    if 'by_seed' in value:
        value = value['by_seed'][str(cfg['seed'])]['constants']
    if not isinstance(value, list):
        raise ValueError('Calibration must be a constants list or a by_seed manifest')
    return value


def save_checkpoint(path, model, optimizer, step, identity, elapsed, samples):
    device = next(model.parameters()).device
    sync(device)
    state = {'schema': 'portable_v1', 'model': cpu_tree(model.state_dict()),
             'optimizer': cpu_tree(optimizer.state_dict()), 'step': step, 'identity': identity,
             'training_seconds': elapsed, 'step_seconds': samples,
             'torch_rng': torch.get_rng_state(),
             'cuda_rng': [x.cpu() for x in torch.cuda.get_rng_state_all()] if device.type == 'cuda' else [],
             'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    return digest(path)


def train(args):
    cfg = load_recipe(args.config)
    for key in ('arm', 'seed', 'lr', 'device'):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if not np.isfinite(cfg['lr']) or cfg['lr'] <= 0:
        raise ValueError('Learning rate must be finite and positive')
    model_cfg = Config(**{k: cfg[k] for k in MODEL_KEYS})
    device = configure(cfg['device'])
    torch.set_num_threads(cfg['threads'])
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])
    constants = calibration(args.calibration, cfg)
    folder = args.data.resolve()
    files = {name: digest(folder / name) for name in
             ('tokens_train.npy', 'docs_train.npy', 'tokens_dev.npy', 'docs_dev.npy')}
    for name in ('manifest.json', 'tokenizer.json'):
        if (folder / name).is_file():
            files[name] = digest(folder / name)
    corpus, dev = Corpus(folder, 'train', cfg['context']), Corpus(folder, 'dev', cfg['context'])
    for split in (corpus, dev):
        if split.tokens.ndim != 1 or split.tokens.dtype.kind not in 'iu' or split.docs.dtype.kind not in 'iu':
            raise ValueError('Token and boundary arrays must be integer arrays')
        if int(split.tokens.min()) < 0 or int(split.tokens.max()) >= cfg['vocab']:
            raise ValueError('Token id outside configured vocabulary')
    schedule = corpus.schedule(cfg['seed'], cfg['steps'], cfg['batch'])
    eval_schedule = dev.evaluation_schedule(cfg['eval_tokens'])
    model = LanguageModel(model_cfg).initialize(cfg['seed'])
    initial_sha = parameter_digest(model)
    if args.expected_initial_sha256 and initial_sha != args.expected_initial_sha256:
        raise ValueError('Initialization differs from expected fingerprint')
    if constants is not None:
        model.apply_calibration(constants)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = model.optimizer(cfg['lr'])
    identity = {'schema': 'portable_v1', 'config': cfg, 'data_sha256': files,
                'initial_parameter_sha256': initial_sha, 'calibration': constants,
                'environment': environment(device),
                'source_sha256': {p.name: digest(p) for p in sorted((ROOT / 'code').glob('*.py'))}}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        return train_locked(args, output, identity, model, optimizer, corpus, dev, schedule, eval_schedule)


def train_locked(args, output, identity, model, optimizer, corpus, dev, schedule, eval_schedule):
    cfg = identity['config']; device = next(model.parameters()).device
    manifest = output / 'manifest.json'
    existing = manifest.exists()
    if existing:
        if not args.resume:
            raise ValueError('Existing run requires explicit --resume')
        if read_json(manifest)['identity'] != identity:
            raise ValueError('Resume configuration/data/source/environment/initialization differs')
        if (output / 'result.json').exists():
            result = read_json(output / 'result.json')
            if result['status'] != 'complete' or result['identity'] != identity or any(
                    digest(output / name) != sha for name, sha in result['artifacts'].items()):
                raise ValueError('Completed run artifacts differ')
            print('Endpoint already complete; verified artifacts.'); return 0
    else:
        if args.resume or any(p.name != '.run.lock' for p in output.iterdir()):
            raise ValueError('New run needs an empty output directory; resume needs a manifest')
        write_json(manifest, {'identity': identity, 'model': model.metadata(), 'created_unix': time.time()})
        np.save(output / 'train_offsets.npy', schedule, allow_pickle=False)
        np.save(output / 'dev_offsets.npy', eval_schedule[0], allow_pickle=False)
        np.save(output / 'dev_doc_ids.npy', eval_schedule[1], allow_pickle=False)
    for name, expected in [('train_offsets.npy', schedule), ('dev_offsets.npy', eval_schedule[0]),
                           ('dev_doc_ids.npy', eval_schedule[1])]:
        if not np.array_equal(np.load(output / name, allow_pickle=False), expected):
            raise ValueError('Frozen schedule changed: ' + name)
    step, elapsed, samples = 0, 0., []
    checkpoint = output / ('final.pt' if (output / 'final.pt').exists() else 'latest.pt')
    if existing:
        # Only load checkpoints you created/trust: optimizer/RNG payload uses pickle.
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['schema'] != 'portable_v1' or state['identity'] != identity or not 0 <= state['step'] <= cfg['steps']:
            raise ValueError('Checkpoint identity/step differs')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        if identity['calibration'] is not None:
            model.verify_calibration(identity['calibration'])
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda':
            if len(state['cuda_rng']) != torch.cuda.device_count():
                raise ValueError('CUDA RNG device count differs')
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        step, elapsed, samples = state['step'], state['training_seconds'], state['step_seconds']
        if len(samples) != step or not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Checkpoint timing/update ledger differs')
        if step and (not optimizer.state or any(float(v['step']) != step for v in optimizer.state.values())):
            raise ValueError('AdamW update counters differ from checkpoint step')
    else:
        save_checkpoint(checkpoint, model, optimizer, 0, identity, elapsed, samples)

    def evaluation(label):
        summary = output / (label + '.json')
        if summary.exists():
            value = read_json(summary)
            if digest(output / (label + '.npz')) != value['arrays_sha256']:
                raise ValueError('Evaluation array hash changed')
            return value
        return save_eval(output, label, evaluate(model, dev, eval_schedule, cfg['batch']))

    if step == 0:
        evaluation('initial_dev')
    elif not (output / 'initial_dev.json').exists():
        raise ValueError('Resumed endpoint lacks its initialization evaluation')
    if step and step < cfg['steps'] and cfg['eval_every'] and step % cfg['eval_every'] == 0:
        evaluation(f'dev_step_{step}')
    while step < cfg['steps'] and not STOP:
        model.train()
        x, y = corpus.batch(schedule[step], device)
        rate = learning_rate(step, cfg['steps'], cfg['lr'], cfg['warmup'], cfg['lr_schedule'])
        for group in optimizer.param_groups:
            group['lr'] = rate
        sync(device); begun = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); sync(device)
        duration = time.perf_counter() - begun
        elapsed += duration; samples.append(duration); step += 1
        if step % cfg['log_every'] == 0 or step == cfg['steps']:
            row = {'step': step, 'tokens': step * cfg['batch'] * cfg['context'],
                   'train_nll': float(loss.detach().cpu()), 'grad_norm': float(grad_norm.cpu()), 'lr': rate}
            with (output / 'progress.jsonl').open('a') as stream:
                import json
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            print(row, flush=True)
        if step % cfg['save_every'] == 0:
            save_checkpoint(output / 'latest.pt', model, optimizer, step, identity, elapsed, samples)
        if not STOP and step < cfg['steps'] and cfg['eval_every'] and step % cfg['eval_every'] == 0:
            evaluation(f'dev_step_{step}')
    filename = 'final.pt' if step == cfg['steps'] else 'latest.pt'
    save_checkpoint(output / filename, model, optimizer, step, identity, elapsed, samples)
    if step < cfg['steps']:
        write_json(output / 'paused.json', {'status': 'paused', 'step': step, 'checkpoint': filename})
        return 75
    metrics = evaluation('dev_final')
    for name, expected in identity['data_sha256'].items():
        if digest(corpus.root / name) != expected:
            raise ValueError('Frozen training/development data changed during the run: ' + name)
    result = {'status': 'complete', 'identity': identity, 'steps': step,
              'tokens': step * cfg['batch'] * cfg['context'], 'model': model.metadata(),
              'dev': metrics, 'training_seconds': elapsed,
              'median_step_ms': float(np.median(samples[10:] or samples)) * 1000,
              'artifacts': {name: digest(output / name) for name in ('final.pt', 'dev_final.json', 'dev_final.npz')}}
    write_json(output / 'result.json', result)
    (output / 'paused.json').unlink(missing_ok=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Full-budget JSON recipe; see configs/')
    parser.add_argument('--data', type=Path, required=True, help='Frozen tokens_train/dev.npy and docs_train/dev.npy')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--device', help='cuda:0 for experiments; cpu for verification')
    parser.add_argument('--calibration', type=Path, help='Frozen train-only per-layer moment coefficients')
    parser.add_argument('--expected-initial-sha256')
    parser.add_argument('--resume', action='store_true', help='Resume same recipe/data/source/environment only')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    return train(args)


if __name__ == '__main__':
    raise SystemExit(main())
