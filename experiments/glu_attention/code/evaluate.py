"""Evaluate a declared cohort only after every final training endpoint validates.

This portable gate verifies local artifacts. It does not replace the independent
historical data/training audits preserved in the evidence bundle.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from common import MODEL_KEYS, ROOT, configure, digest, environment, read_json, write_json
from data import evaluate, save_eval
from model import Config, LanguageModel


def require(ok, message):
    if not ok:
        raise ValueError(message)


def executable_sources():
    """The same complete portable source map frozen by the trainer."""
    return {path.name: digest(path) for path in sorted((ROOT / 'code').glob('*.py'))}


def verify_endpoint(run, expected):
    """Uses training files only; never opens any holdout file."""
    require(not (run / 'failure.json').exists(), 'Training failure requires diagnosis')
    result = read_json(run / 'result.json')
    require(result['status'] == 'complete', 'Incomplete training endpoint')
    require(result['identity']['source_sha256'] == executable_sources(),
            'Executable source differs from the completed training endpoint')
    cfg = result['identity']['config']
    require(cfg == expected and result['steps'] == cfg['steps'], 'Declared endpoint configuration differs')
    require(result['tokens'] == cfg['steps'] * cfg['batch'] * cfg['context'], 'Endpoint token budget differs')
    require(set(result['artifacts']) == {'final.pt', 'dev_final.npz', 'dev_final.json'},
            'Incomplete endpoint artifact set')
    for name, sha in result['artifacts'].items():
        require(name in ('final.pt', 'dev_final.npz', 'dev_final.json'), 'Unexpected endpoint artifact')
        require(digest(run / name) == sha, 'Endpoint artifact changed: ' + name)
    checkpoint = run / 'final.pt'
    require(result['artifacts']['final.pt'] == digest(checkpoint), 'Checkpoint binding differs')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(state['schema'] == 'glu_controls_v1' and state['identity'] == result['identity'], 'Checkpoint identity differs')
    require(state['step'] == cfg['steps'] and len(state['step_seconds']) == cfg['steps'], 'Incomplete update ledger')
    parameter_ids = {p for group in state['optimizer']['param_groups'] for p in group['params']}
    require(set(state['optimizer']['state']) == parameter_ids and parameter_ids, 'Incomplete optimizer parameter state')
    require(state['optimizer']['state'] and all(float(v['step']) == cfg['steps']
            for v in state['optimizer']['state'].values()), 'Incomplete AdamW counters')
    require(all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in state['model'].values()),
            'Checkpoint contains nonfinite or non-FP32 weights')
    # Validate the full executable model and frozen mechanism buffers before
    # open_holdout can be called, not merely the shapes/finiteness of tensors.
    model = LanguageModel(Config(**{k: cfg[k] for k in MODEL_KEYS})).float()
    constants = state['identity']['calibration']
    require(constants is not None or cfg['arm'] not in ('v_affine_match', 'v_silu_standardized'),
            'Mechanism endpoint lacks frozen calibration')
    if constants is not None:
        model.apply_calibration(constants)
    model.load_state_dict(state['model'], strict=True)
    if constants is not None:
        model.verify_calibration(constants)
    return result


def verify_cohort(path):
    cohort = read_json(path)
    require(cohort.get('schema') == 'portable_cohort_v1' and cohort['runs'], 'Invalid declared cohort')
    rows = []; seen = set(); data_identity = None; model_shape = None; training_recipe = None
    shared_model_keys = tuple(k for k in MODEL_KEYS if k not in ('arm', 'glu_width_policy'))
    shared_training_keys = ('batch', 'steps', 'warmup', 'lr_schedule', 'eval_tokens', 'eval_every')
    for item in cohort['runs']:
        run = (path.parent / item['path']).resolve()
        cfg = item['config']; key = (cfg['arm'], cfg['seed'])
        require(key not in seen, 'Duplicate arm/seed in cohort'); seen.add(key)
        result = verify_endpoint(run, cfg)
        if data_identity is None:
            data_identity = result['identity']['data_sha256']
            model_shape = {k: cfg[k] for k in shared_model_keys}
            training_recipe = {k: cfg[k] for k in shared_training_keys}
        require(result['identity']['data_sha256'] == data_identity, 'Cohort used different training/development data')
        require({k: cfg[k] for k in shared_model_keys} == model_shape, 'Cohort model shape differs')
        require({k: cfg[k] for k in shared_training_keys} == training_recipe, 'Cohort training exposure/schedule differs')
        rows.append((run, cfg, result))
    return rows


def verify_holdout_files(folder, seal_path, expected_seal_sha):
    require(digest(seal_path) == expected_seal_sha, 'Frozen holdout descriptor changed')
    seal = read_json(seal_path)
    required = {'tokens_holdout.npy', 'docs_holdout.npy', 'eval_offsets.npy', 'eval_document_ids.npy'}
    require(required.issubset(seal['files_sha256']), 'Missing frozen holdout hashes')
    for name, sha in seal['files_sha256'].items():
        require(Path(name).name == name and digest(folder / name) == sha, 'Holdout hash differs: ' + name)
    require(digest(seal_path) == expected_seal_sha, 'Frozen holdout descriptor changed while validating')
    return seal


def open_holdout(folder, seal_path, context, vocab, expected_seal_sha):
    # This function is invoked only after every declared endpoint passes.
    seal = verify_holdout_files(folder, seal_path, expected_seal_sha)
    require(seal['context'] == context, 'Holdout context differs')
    tokens = np.load(folder / 'tokens_holdout.npy', mmap_mode='r', allow_pickle=False)
    docs = np.load(folder / 'docs_holdout.npy', allow_pickle=False)
    offsets = np.load(folder / 'eval_offsets.npy', allow_pickle=False)
    ids = np.load(folder / 'eval_document_ids.npy', allow_pickle=False)
    require(tokens.ndim == 1 and tokens.dtype == np.uint16 and docs.dtype == np.int64
            and docs.shape == (seal['documents'], 2), 'Holdout array schema differs')
    require(offsets.dtype == ids.dtype == np.int64 and np.array_equal(offsets, docs[:, 0])
            and np.array_equal(ids, np.arange(len(docs))), 'Holdout must use every document first window')
    require(docs[0, 0] == 0 and docs[-1, 1] == len(tokens)
            and np.array_equal(docs[1:, 0], docs[:-1, 1])
            and ((docs[:, 1] - docs[:, 0]) > context).all(), 'Holdout boundaries differ')
    require((tokens < vocab).all() and (tokens[docs[:, 1] - 1] == seal.get('eos', 3)).all(), 'Holdout vocabulary/EOS differs')
    require(len(docs) * context == seal['eval_tokens'], 'Holdout target budget differs')

    class HoldoutCorpus:
        def __init__(self):
            self.context = context
        def batch(self, starts, device):
            positions = starts[:, None] + np.arange(context + 1)[None, :]
            sample = torch.from_numpy(np.asarray(tokens[positions], dtype=np.int64)).to(device)
            return sample[:, :-1], sample[:, 1:]
    return HoldoutCorpus(), (offsets, ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True, help='Predeclared complete cohort; paths relative to its JSON')
    parser.add_argument('--holdout', type=Path, required=True, help='Explicit sealed holdout array directory')
    parser.add_argument('--seal', type=Path, required=True, help='Frozen file hashes and first-window cohort definition')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cohort_sha = digest(args.cohort)
    rows = verify_cohort(args.cohort.resolve())
    require(digest(args.cohort) == cohort_sha, 'Cohort changed during the endpoint gate')
    device = configure(args.device); torch.set_num_threads(rows[0][1]['threads'])
    seal_sha = digest(args.seal)
    corpus, schedule = open_holdout(args.holdout, args.seal, rows[0][1]['context'], rows[0][1]['vocab'], seal_sha)
    binding = {'cohort_sha256': cohort_sha, 'seal_sha256': seal_sha,
               'source_sha256': executable_sources(), 'environment': environment(device)}
    args.output.mkdir(parents=True, exist_ok=True)
    done_path = args.output / 'done.json'
    require(not any(args.output.iterdir()), 'Evaluation needs a new empty output directory')
    artifacts = {}; jobs = []
    for run, cfg, result in rows:
        checkpoint = run / 'final.pt'; sha = digest(checkpoint)
        require(sha == result['artifacts']['final.pt'], 'Checkpoint changed after gate')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = LanguageModel(Config(**{k: cfg[k] for k in MODEL_KEYS})).to(device)
        constants = state['identity']['calibration']
        if constants is not None:
            model.apply_calibration(constants)
        model.load_state_dict(state['model'])
        if constants is not None:
            model.verify_calibration(constants)
        del state
        label = f"{cfg['arm']}_s{cfg['seed']}"
        summary = save_eval(args.output, label, evaluate(model, corpus, schedule, cfg['batch']))
        require(digest(checkpoint) == sha, 'Checkpoint changed during read-only evaluation')
        summary.update(status='complete', arm=cfg['arm'], seed=cfg['seed'], ppl=math.exp(summary['nll']),
                       checkpoint_sha256=sha, training_result_sha256=digest(run / 'result.json'), binding=binding)
        write_json(args.output / (label + '.json'), summary)
        for suffix in ('.json', '.npz'):
            artifacts[label + suffix] = digest(args.output / (label + suffix))
        jobs.append({'arm': cfg['arm'], 'seed': cfg['seed'], 'label': label})
        del model
    require(digest(args.cohort) == cohort_sha, 'Cohort changed during evaluation')
    require(executable_sources() == binding['source_sha256'], 'Executable source changed during evaluation')
    verify_holdout_files(args.holdout, args.seal, seal_sha)
    write_json(done_path, {'status': 'complete', 'binding': binding, 'jobs': jobs, 'artifacts_sha256': artifacts})
    print(f'Evaluated {len(rows)} completed endpoints on {len(schedule[0])} frozen document windows.')


if __name__ == '__main__':
    main()
