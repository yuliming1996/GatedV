"""Recompute Appendix B from the twelve archived per-window evaluations."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np

RESULTS = Path(__file__).resolve().parent
MATRIX = RESULTS / 'cross_eval'
NATIVE = RESULTS / 'medium_lr3e4'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    complete = json.loads((MATRIX / 'complete.json').read_text())
    summary = json.loads((MATRIX / 'summary.json').read_text())
    reference = json.loads((NATIVE / 'manifest.json').read_text())
    seeds = [20270211, 20270212, 20270213]
    cells = {'BB': ('baseline', 'linear'), 'BS': ('baseline', 'silu'),
             'GB': ('v_silu', 'linear'), 'GS': ('v_silu', 'silu')}
    if (complete['status'] != 'COMPLETE_CROSS_EVALUATION'
            or complete['protocol_sha256'] != digest(MATRIX / 'protocol.json')
            or summary['complete_sha256'] != digest(MATRIX / 'complete.json')
            or complete['archive_sha256'] != reference['evaluation_done_sha256']):
        raise ValueError('Cross-evaluation records do not match the original evaluations')
    rows = {(r['training_arm'], r['inference'], r['seed']): r for r in complete['results']}
    native = {(r['arm'], r['seed']): r for r in reference['runs']}
    if len(rows) != 12 or len(complete['results']) != 12:
        raise ValueError('Expected twelve distinct evaluations')
    weights = {}
    for key, (arm, mode) in cells.items():
        ppls = []
        for seed in seeds:
            row = rows[arm, mode, seed]
            label = f'{arm}_s{seed}'
            archive = NATIVE / f'{label}.npz'
            path = MATRIX / f'{label}_infer_{mode}.npz'
            diagonal = key in ('BB', 'GS')
            if (digest(path) != row['arrays_sha256'] or digest(archive) != native[arm, seed]['arrays_sha256']
                    or row['checkpoint_sha256'] != native[arm, seed]['checkpoint_sha256']
                    or row['parameter_sha256'] != weights.setdefault((arm, seed), row['parameter_sha256'])
                    or row['checked_diagonal'] != diagonal):
                raise ValueError(f'Evaluation binding differs for {path.name}')
            with np.load(path, allow_pickle=False) as z, np.load(archive, allow_pickle=False) as ref:
                for field in ('counts', 'doc_ids', 'offsets'):
                    np.testing.assert_array_equal(z[field], ref[field])
                np.testing.assert_array_equal(z['doc_ids'], np.arange(2048))
                np.testing.assert_array_equal(z['counts'], np.full(2048, 256))
                loss = z['loss_sums']
                if loss.shape != (2048,) or not np.isfinite(loss).all() or (loss < 0).any():
                    raise ValueError(f'Invalid window losses in {path.name}')
                if diagonal:
                    np.testing.assert_array_equal(loss, ref['loss_sums'])
                nll = math.fsum(loss) / int(z['counts'].sum())
            ppl = math.exp(nll)
            np.testing.assert_allclose([nll, ppl], [row['nll'], row['ppl']], rtol=1e-12, atol=0)
            ppls.append(ppl)
        expected = summary['cell_statistics'][key]
        np.testing.assert_allclose([mean(ppls), stdev(ppls)],
                                   [expected['mean_ppl'], expected['sample_sd_ppl']], rtol=1e-12, atol=0)
        print(f'{arm:8} training / {mode:6} inference: PPL {mean(ppls):,.3f} ± {stdev(ppls):.3f}')


if __name__ == '__main__':
    main()
