"""Recompute the main medium-model held-out comparison from the six saved arrays."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

RESULTS = Path(__file__).resolve().parent
sys.path.insert(0, str(RESULTS.parent / 'code'))
from statistics_core import medium_cluster_interval


def main():
    directory = RESULTS / 'medium_lr3e4'
    manifest = json.loads((directory / 'manifest.json').read_text())
    seeds = [20270211, 20270212, 20270213]
    arms = ('baseline', 'v_silu')
    expected_pairs = {(arm, seed) for arm in arms for seed in seeds}
    rows = manifest['runs']
    if (manifest['peak_lr'] != 0.0003 or manifest['seeds'] != seeds or len(rows) != 6
            or {(r['arm'], r['seed']) for r in rows} != expected_pairs):
        raise ValueError('Expected both models for all three confirmation seeds at 0.0003')

    losses, perplexities = {}, {}
    offsets = None
    doc_ids = np.arange(2048)
    counts = np.full(2048, 256)
    for row in rows:
        key = row['arm'], row['seed']
        path = directory / f'{key[0]}_s{key[1]}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['arrays_sha256']:
            raise ValueError(f'Array digest differs for {path.name}')
        with np.load(path, allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays['doc_ids'], doc_ids)
            np.testing.assert_array_equal(arrays['counts'], counts)
            if offsets is None:
                offsets = arrays['offsets']
            np.testing.assert_array_equal(arrays['offsets'], offsets)
            loss = arrays['loss_sums'].astype(np.float64)
        if loss.shape != counts.shape or not np.isfinite(loss).all() or (loss < 0).any():
            raise ValueError(f'Invalid loss sums in {path.name}')
        losses[key] = loss
        nll = math.fsum(loss) / int(counts.sum())
        perplexities[key] = math.exp(nll)
        np.testing.assert_allclose([nll, perplexities[key]], [row['nll'], row['ppl']],
                                   rtol=0, atol=1e-12)

    ppl = {arm: [perplexities[arm, seed] for seed in seeds] for arm in arms}
    base, gated = (np.stack([losses[arm, seed] for seed in seeds]) for arm in arms)
    change = 100 * (statistics.mean(ppl['v_silu']) / statistics.mean(ppl['baseline']) - 1)
    paired_window_differences = (gated - base).mean(axis=0)
    mean_difference = math.fsum(paired_window_differences) / int(counts.sum())
    lower = int(np.count_nonzero(paired_window_differences < 0))
    interval = medium_cluster_interval(base, gated, counts, doc_ids)['nll_difference_ci95']
    expected = manifest['published']
    np.testing.assert_allclose([change, mean_difference, *interval],
                               [expected['ppl_change_pct'], expected['mean_paired_nll'],
                                *expected['document_bootstrap_ci95']], rtol=0, atol=1e-12)
    if lower != expected['windows_with_lower_seed_mean_nll']:
        raise ValueError('Window improvement count differs from the published result')

    for arm in arms:
        print(f'{arm} held-out PPL: {statistics.mean(ppl[arm]):.3f} ± {statistics.stdev(ppl[arm]):.3f}')
    wins = sum(g < b for g, b in zip(ppl['v_silu'], ppl['baseline']))
    print(f'PPL change: {change:.2f}% ({wins}/{len(seeds)} seed pairs improved)')
    print(f'Mean paired NLL difference: {mean_difference:.6f}')
    print(f'95% document-bootstrap interval: [{interval[0]:.6f}, {interval[1]:.6f}]')
    print(f'Windows with lower mean NLL across seeds: {lower}/{len(counts)} '
          f'({100 * lower / len(counts):.1f}%)')


if __name__ == '__main__':
    main()
