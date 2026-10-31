"""Paired NLL/PPL statistics from committed portable evaluation loss arrays."""
import argparse
import math
from pathlib import Path
import statistics
import numpy as np
from common import digest, read_json, write_json
from statistics_core import medium_cluster_interval, small_cluster_interval, seed_uncertainty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--study', choices=('pilot', 'small', 'medium'), required=True, help='pilot: one descriptive pair, no uncertainty interval; otherwise five/three seed kernels')
    parser.add_argument('--domains', type=Path, help='Optional JSONL with doc_id and domain_group; explicitly opened after evaluation')
    parser.add_argument('--candidate', default='v_silu')
    parser.add_argument('--reference', default='baseline', help='Use v_silu to compare GLU directly against GatedV')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.reference == args.candidate:
        raise ValueError('Reference and candidate must differ')
    done = read_json(args.evaluation / 'done.json')
    if done['status'] != 'complete':
        raise ValueError('Evaluation cohort incomplete')
    for name, sha in done['artifacts_sha256'].items():
        if Path(name).name != name or digest(args.evaluation / name) != sha:
            raise ValueError('Evaluation artifact changed: ' + name)
    losses = {}; summaries = {}; reference = None
    for row in done['jobs']:
        if row['arm'] not in (args.reference, args.candidate):
            continue
        key = (row['arm'], row['seed'])
        if key in losses:
            raise ValueError('Duplicate arm/seed')
        meta = read_json(args.evaluation / (row['label'] + '.json'))
        path = args.evaluation / (row['label'] + '.npz')
        if meta['status'] != 'complete' or meta['arrays_sha256'] != digest(path):
            raise ValueError('Loss payload differs')
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {'loss_sums', 'counts', 'doc_ids', 'offsets'}:
                raise ValueError('Unexpected loss array schema')
            cohort = {k: arrays[k].copy() for k in ('counts', 'doc_ids', 'offsets')}
            values = arrays['loss_sums'].copy()
        if reference is None:
            reference = cohort
        if any(not np.array_equal(cohort[k], reference[k]) for k in cohort):
            raise ValueError('Unpaired evaluation windows')
        if (not np.isfinite(values).all() or (values < 0).any() or (cohort['counts'] <= 0).any()
                or values.shape != cohort['counts'].shape):
            raise ValueError('Invalid per-window loss sums/counts')
        losses[key] = values
        nll = math.fsum(map(float, values)) / int(cohort['counts'].sum())
        if not math.isclose(nll, meta['nll'], rel_tol=0, abs_tol=1e-12):
            raise ValueError('Summary NLL differs from raw loss sums')
        summaries[key] = {'arm': key[0], 'seed': key[1], 'nll': nll, 'ppl': math.exp(nll)}
    seeds = sorted(seed for arm, seed in losses if arm == args.reference)
    expected_n = {'pilot': 1, 'small': 5, 'medium': 3}[args.study]
    if len(seeds) != expected_n or set(losses) != {(a, s) for a in (args.reference, args.candidate) for s in seeds}:
        raise ValueError(f'{args.study} requires exactly {expected_n} complete paired seeds')
    base = np.array([losses[args.reference, s] for s in seeds])
    candidate = np.array([losses[args.candidate, s] for s in seeds])
    differences = [summaries[args.candidate, s]['nll'] - summaries[args.reference, s]['nll'] for s in seeds]
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences) if len(seeds) > 1 else None
    se = sd / math.sqrt(len(seeds)) if sd is not None else None
    uncertainty = None if args.study == 'pilot' else (seed_uncertainty(differences) if args.study == 'small' else
                   {'mean': mean, 'sample_sd': sd, 'standard_error': se, 'df': 2,
                    't_critical': 4.302652729911275,
                    't_ci95': [mean - 4.302652729911275 * se, mean + 4.302652729911275 * se]})
    bootstrap = small_cluster_interval if args.study == 'small' else medium_cluster_interval
    document = None if args.study == 'pilot' else bootstrap(base, candidate, reference['counts'], reference['doc_ids'].tolist())
    domains = None
    if args.domains:
        rows = [read_json_line(line) for line in args.domains.read_text().splitlines() if line.strip()]
        mapping = {r['doc_id']: r['domain_group'] for r in rows}
        if len(mapping) != len(rows):
            raise ValueError('Duplicate domain metadata doc_id')
        labels = [mapping[int(i)] for i in reference['doc_ids']]
        domains = None if args.study == 'pilot' else bootstrap(base, candidate, reference['counts'], labels)
    mean_ppl = {arm: statistics.mean(summaries[arm, s]['ppl'] for s in seeds) for arm in (args.reference, args.candidate)}
    result = {'study': args.study, 'reference': args.reference, 'candidate': args.candidate, 'seeds': seeds,
              'evaluation_done_sha256': digest(args.evaluation / 'done.json'),
              'rows': list(summaries.values()), 'paired_nll_differences': differences,
              'paired_ppl_change_pct': [100 * math.expm1(d) for d in differences],
              'mean_ppl': mean_ppl,
              'mean_ppl_change_pct': 100 * (mean_ppl[args.candidate] / mean_ppl[args.reference] - 1),
              'seed_uncertainty': uncertainty, 'document_cluster': document, 'domain_cluster': domains,
              'limits': ['Bootstrap is conditional on observed training seeds.',
                         'Exponentiated pooled NLL interval is not a mean-PPL-ratio interval.',
                         'One document per domain yields identical document/domain information.',
                         'Small paired-seed t intervals depend on distributional assumptions.']}
    if args.study == 'pilot':
        result['limits'] = ['One paired seed: descriptive only; no seed-level or bootstrap confidence interval is reported.']
    if args.output.exists():
        raise ValueError('Refusing to overwrite analysis output')
    args.output.parent.mkdir(parents=True, exist_ok=True); write_json(args.output, result)
    print(f"Mean PPL change: {result['mean_ppl_change_pct']:.4f}%; paired NLL: {mean:.6f}")


def read_json_line(line):
    import json
    return json.loads(line)


if __name__ == '__main__':
    main()
