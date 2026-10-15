"""Inspect shapes/parameter budgets without training, data access, or CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
import torch
from common import MODEL_KEYS, load_recipe
from model import LanguageModel, Config


def inspect():
    rows = []
    for arm in ('baseline', 'v_silu', 'v_glu_full', 'v_glu_reduced'):
        path = ROOT / 'configs' / 'lr_controls_20261013' / f'medium_{arm}_lr6e-4.json'
        recipe = load_recipe(path)
        cfg = Config(**{k: recipe[k] for k in MODEL_KEYS})
        # Shapes and numel only; allocates no model weights on CPU or GPU.
        with torch.device('meta'):
            model = LanguageModel(cfg)
        meta = model.metadata()
        rows.append({'arm': arm, 'parameters': meta['parameters'],
                     **meta['value_projection'], 'heads': cfg.heads,
                     'lr': recipe['lr'], 'steps': recipe['steps'],
                     'target_exposures': recipe['steps'] * recipe['batch'] * cfg.context,
                     'recipe_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    base = rows[0]['parameters']
    for row in rows:
        if row['parameters'] - base != row['parameter_delta_vs_baseline']:
            raise ValueError('Actual parameter count disagrees with projection formula')
        row['parameter_change_pct'] = 100 * (row['parameters'] - base) / base
    return {'status': 'PARAMETER_INSPECTION_ONLY', 'scope': 'Meta-tensor shapes only; no data, training, or CUDA execution.',
            'rows': rows,
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((ROOT / 'code').glob('*.py'))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new inspection JSON; refuses overwrite')
    args = parser.parse_args()
    result = inspect()
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
