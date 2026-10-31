"""Evaluate both value operators on each final checkpoint of a portable 0.0003 reproduction."""
import argparse
import math
from pathlib import Path
from statistics import mean, stdev
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import numpy as np
import torch
from common import MODEL_KEYS, configure, digest, parameter_digest, read_json, write_json
from data import evaluate, save_eval
from evaluate import open_holdout, verify_cohort
from model import Config, LanguageModel


def evaluate_modes(weights, cfg, device, corpus, schedule, output, reference):
    rows = []
    for mode, infer_arm in [('linear', 'baseline'), ('silu', 'v_silu')]:
        config = {k: cfg[k] for k in MODEL_KEYS}
        config['arm'] = infer_arm
        model = LanguageModel(Config(**config))
        model.load_state_dict(weights, strict=True)
        fingerprint = parameter_digest(model)
        if rows and rows[0]['parameter_sha256'] != fingerprint:
            raise ValueError('Inference modes must use identical trained weights')
        model.requires_grad_(False).eval().to(device)
        result = evaluate(model, corpus, schedule, cfg['batch'])
        diagonal = cfg['arm'] == infer_arm
        with np.load(reference, allow_pickle=False) as ref:
            for field in ('counts', 'doc_ids', 'offsets'):
                np.testing.assert_array_equal(result[field], ref[field])
            if diagonal:
                np.testing.assert_array_equal(result['loss_sums'], ref['loss_sums'])
        label = f"{cfg['arm']}_s{cfg['seed']}_infer_{mode}"
        row = save_eval(output, label, result)
        row.update(training_arm=cfg['arm'], seed=cfg['seed'], inference=mode,
                   parameter_sha256=fingerprint, ppl=math.exp(row['nll']), checked_diagonal=diagonal)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reproduction', type=Path, default=ROOT / 'outputs/reproduction_medium-lr3e4')
    parser.add_argument('--holdout', type=Path, default=ROOT / 'data/paper_medium/medium_holdout')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    cohort = args.reproduction / 'cohort.json'
    template = ROOT / 'configs/medium-lr3e4_cohort.json'
    if read_json(cohort) != read_json(template):
        raise ValueError('Expected the six published 0.0003 CUDA recipes')
    done = read_json(args.reproduction / 'complete.json')
    if (done['status'], done['study']) != ('COMPLETE_CONFIRMATION_REPRODUCTION', 'medium-lr3e4'):
        raise ValueError('Run the medium-lr3e4 reproduction to completion first')
    rows = verify_cohort(cohort.resolve())
    original = args.reproduction / done['evaluation']
    if digest(original / 'done.json') != done['artifacts_sha256'][done['evaluation'] + '/done.json']:
        raise ValueError('Original evaluation has changed')
    archive = read_json(original / 'done.json')
    device = configure(rows[0][1]['device'])
    torch.set_num_threads(rows[0][1]['threads'])
    seal = ROOT / 'configs/medium_holdout.json'
    corpus, schedule = open_holdout(args.holdout, seal, 256, 8192, archive['binding']['seal_sha256'])
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for folder, cfg, training in rows:
        checkpoint = folder / 'final.pt'
        native = original / (folder.name + '.npz')
        meta = original / (folder.name + '.json')
        sha = training['artifacts']['final.pt']
        if (digest(meta) != archive['artifacts_sha256'][meta.name] or read_json(meta)['checkpoint_sha256'] != sha
                or digest(native) != archive['artifacts_sha256'][native.name] or digest(checkpoint) != sha):
            raise ValueError(f'Weights or native reference changed for {folder.name}')
        weights = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
        for row in evaluate_modes(weights, cfg, device, corpus, schedule, args.output, native):
            results.append(dict(row, checkpoint_sha256=sha))
    write_json(args.output / 'complete.json', {'cohort_sha256': digest(cohort), 'results': results})
    for arm in ('baseline', 'v_silu'):
        for mode in ('linear', 'silu'):
            ppls = [r['ppl'] for r in results if r['training_arm'] == arm and r['inference'] == mode]
            print(f'{arm} training / {mode} inference: PPL {mean(ppls):,.3f} ± {stdev(ppls):.3f}')


if __name__ == '__main__':
    main()
