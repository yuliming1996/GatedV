"""Recompute the single-seed development grid and the shared-rate rows of Table 7."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

GRID = Path(__file__).resolve().parent / 'value_gate_grid'
ARMS = ('baseline', 'v_silu', 'v_glu_full', 'v_glu_reduced', 'v_tanh_gate')
SHARED = (0.0006, 0.0003, 0.0001)


def main():
    with (GRID / 'endpoints.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = {(arm, lr) for arm in ARMS for lr in SHARED}
    expected.update((arm, 0.00001) for arm in ARMS[:-1])
    observed = {(row['arm'], float(row['peak_lr'])) for row in rows}
    if len(rows) != 19 or observed != expected:
        raise ValueError('Expected sixteen original grid cells and three completed tanh cells')
    schedule = None
    ppls = {}
    for row in rows:
        arm, lr = row['arm'], float(row['peak_lr'])
        cfg = json.loads((GRID.parents[1] / row['config']).read_text())
        if ((cfg['arm'], cfg['lr'], cfg['seed'], cfg['steps']) != (arm, lr, 20270201, 327680)
                or int(row['seed']) != cfg['seed'] or int(row['steps']) != cfg['steps']
                or int(row['training_targets']) != cfg['steps'] * cfg['context'] * cfg['batch']):
            raise ValueError(f'Recipe differs for {row["run_id"]}')
        path = GRID / (row['run_id'] + '.npz')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['arrays_sha256']:
            raise ValueError(f'Array digest differs for {path.name}')
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['counts'], np.full(2048, 256))
            current = np.stack([z['offsets'], z['doc_ids']])
            if schedule is None:
                schedule = current
            np.testing.assert_array_equal(current, schedule)
            loss = z['loss_sums']
            if loss.shape != (2048,) or not np.isfinite(loss).all() or (loss < 0).any():
                raise ValueError(f'Invalid development losses in {path.name}')
            nll = math.fsum(loss) / int(z['counts'].sum())
        ppl = math.exp(nll)
        np.testing.assert_allclose([nll, ppl], [float(row['dev_nll']), float(row['dev_ppl'])],
                                   rtol=1e-12, atol=0)
        ppls[arm, lr] = ppl
    print('Development PPL (seed 20270201)')
    print('arm               ' + '  '.join(f'{lr:>8g}' for lr in SHARED))
    for arm in ARMS:
        print(f'{arm:18}' + '  '.join(f'{ppls[arm, lr]:8.3f}' for lr in SHARED))
    print('Checked all 19 completed endpoints, including four additional runs at 0.00001.')


if __name__ == '__main__':
    main()
