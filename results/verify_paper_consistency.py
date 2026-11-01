"""Check displayed paper/README results against saved data and training recipes.

No training or new evaluation. Run the cohort-specific verifiers as well for
their full array, bootstrap and provenance checks.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {}


def read(relative):
    data = (ROOT / relative).read_bytes()
    SOURCES[relative] = hashlib.sha256(data).hexdigest()
    return data


def load(relative):
    return json.loads(read(relative))


def csv_rows(relative):
    return list(csv.DictReader(io.StringIO(read(relative).decode())))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stats(rows, field='nll'):
    return {arm: [math.exp(float(r[field])) for r in rows if r['arm'] == arm]
            for arm in ('baseline', 'v_silu')}


def tex_ppl(values):
    return f'${mean(values):.3f}\\pm{stdev(values):.3f}$'


def pct(a, b):
    return 100 * (mean(a) / mean(b) - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    paper = read('paper/main.tex').decode()
    readme = read('README.md').decode()

    def table(label):
        tail = paper.split('\\label{' + label + '}', 1)[1]
        return tail.split('\\end{tabular}', 1)[0]

    def check_row(label, cells):
        row = ' & '.join(cells) + r'\\'
        require(row in table(label), f'Paper table {label}: expected {row}')

    small = stats(load('results/studies/D/results.json')['data']['rows'])
    medium_manifest = load('results/medium_lr3e4/manifest.json')
    medium_rows = []
    for row in medium_manifest['runs']:
        name = f"results/medium_lr3e4/{row['arm']}_s{row['seed']}.npz"
        data = read(name)
        require(SOURCES[name] == row['arrays_sha256'], f'Array hash: {name}')
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            nll = math.fsum(z['loss_sums']) / int(z['counts'].sum())
        require(math.isclose(nll, row['nll'], abs_tol=1e-12), f'NLL: {name}')
        medium_rows.append(dict(row, nll=nll))
    medium = stats(medium_rows)
    main_results = {}
    settings = [('small', '2.33M', 0.0048, 32768, small, list(range(20261201, 20261206))),
                ('medium-lr3e4', '26.16M', 0.0003, 327680, medium, list(range(20270211, 20270214)))]
    for study, label, lr, updates, values, seeds in settings:
        b, g = values['baseline'], values['v_silu']
        change = pct(g, b)
        check_row('tab:confirmation', [label, tex_ppl(b), tex_ppl(g),
                                      f'${change:.2f}\\%$', f'{len(seeds)}/{len(seeds)}'])
        md = (f'| {label} | {lr:g} | {len(seeds)} | {mean(b):.3f} ± {stdev(b):.3f} | '
              f'{mean(g):.3f} ± {stdev(g):.3f} | {change:.2f}% |').replace('-', '−')
        require(md in readme, f'README result row: {label}')
        runs = load(f'configs/{study}_cohort.json')['runs']
        require(len(runs) == 2 * len(seeds) and
                {(r['config']['arm'], r['config']['seed']) for r in runs} ==
                {(a, s) for a in ('baseline', 'v_silu') for s in seeds}, f'Cohort: {study}')
        for run in runs:
            cfg = run['config']
            require((cfg['lr'], cfg['steps'], cfg['batch'], cfg['context']) ==
                    (lr, updates, 8, 256), f'Training budget/LR: {study}')
        main_results[study] = {'peak_lr': lr, 'paired_seeds': len(seeds),
                              'mean_ppl': {a: mean(v) for a, v in values.items()},
                              'sample_sd_ppl': {a: stdev(v) for a, v in values.items()},
                              'ppl_change_pct': change, 'training_targets_per_run': updates * 2048}

    check_row('tab:backbone', ['Updates per run', '32,768', '327,680'])
    check_row('tab:backbone', ['Target exposures per run', '67,108,864', '671,088,640'])
    check_row('tab:backbone', ['Confirmation peak learning rate', '0.0048', '0.0003'])
    original_runs = load('configs/medium_cohort.json')['runs']
    require(len(original_runs) == 6 and all(r['config']['lr'] == 0.0006 for r in original_runs),
            'Original medium recipe must remain at 0.0006')
    for item in load('code/source_manifest.json')['files']:
        require(hashlib.sha256(read(item['path'])).hexdigest() == item['packaged_sha256'],
                f'Packaged runtime/recipe: {item["path"]}')

    grid = csv_rows('results/value_gate_grid/endpoints.csv')
    lookup = {(r['arm'], float(r['peak_lr'])): r for r in grid}
    labels = {'baseline': 'Baseline', 'v_silu': 'GatedV', 'v_glu_full': 'Full-width GLU',
              'v_glu_reduced': 'Reduced-width GLU', 'v_tanh_gate': r'$V\odot\tanh(V)$'}
    for arm, label in labels.items():
        rates = (0.0006, 0.0003, 0.0001)
        ppls = [f'{math.exp(float(lookup[arm, lr]["dev_nll"])):.3f}' for lr in rates]
        params = int(lookup[arm, rates[0]]['parameters']) / 1e6
        check_row('tab:value_gates', [label, f'{params:.3f}', *ppls])
        if arm in ('baseline', 'v_silu'):
            check_row('tab:medium_lr', [label, *[f'{math.exp(float(lookup[arm, lr]["dev_nll"])):.3f}'
                                                for lr in (0.00001, 0.0001, 0.0003, 0.0006)]])

    endpoints = csv_rows('results/all_endpoints.csv')

    def values(stages, phase, arm, lr):
        rows = sorted((r for r in endpoints if r['stage'] in stages and r['phase'] == phase
                       and r['arm'] == arm and float(r['lr']) == lr), key=lambda r: int(r['seed']))
        require(len(rows) == 3, f'Expected three seeds: {stages}, {arm}, {lr}')
        return [math.exp(float(r['nll'])) for r in rows]

    comparisons = [('C', 'reused_dev_long', arm, 0.0048, label) for arm, label in
                   [('q_silu', 'Q-only'), ('v_silu', 'GatedV'), ('qkv_silu', 'All Q/K/V')]]
    comparisons += [('EF', 'reused_dev_adaptive', 'v_silu', lr, f'{lr:g}')
                    for lr in (0.0012, 0.0024, 0.0048, 0.0096)]
    for stages, phase, arm, lr, label in comparisons:
        b, g = values(stages, phase, 'baseline', lr), values(stages, phase, arm, lr)
        wins = sum(y < x for x, y in zip(b, g))
        check_row('tab:ablations', [label, tex_ppl(b), tex_ppl(g),
                                   f'${pct(g, b):+.2f}\\%$', f'{wins}/3'])
    for arm, label in [('v_affine_match', 'Affine match'), ('v_silu_standardized', 'Moment-matched SiLU')]:
        for lr in (0.0024, 0.0048):
            c = values('EF', 'reused_dev_adaptive', arm, lr)
            b = values('EF', 'reused_dev_adaptive', 'baseline', lr)
            g = values('EF', 'reused_dev_adaptive', 'v_silu', lr)
            check_row('tab:moment', [label, f'{lr:g}', tex_ppl(c),
                                    f'${pct(c, b):+.2f}\\%$', f'${pct(c, g):+.2f}\\%$'])
    cost = load('results/studies/M/results.json')['data']['cost']
    for mode, label in [('forward', 'Full-context forward time ratio'), ('train_step', 'Training-step time ratio')]:
        check_row('tab:cost', [label, '1.000000', f"{cost[mode]['mean_of_seed_medians']:.6f}"])
    check_row('tab:cost', ['CUDA peak allocated (GiB)',
                          *[f"{cost['memory'][a]['cuda_peak_allocated']['median'] / 2**30:.4f}"
                            for a in ('baseline', 'v_silu')]])

    curves = csv_rows('results/learning_curves_lr3e4.csv')
    development = {}
    for phase in ('small_confirmation', 'medium_confirmation'):
        rows = [r for r in curves if r['phase'] == phase]
        require(all(int(r['training_targets']) == int(r['step']) * 2048 for r in rows), 'Curve budget')
        for arm in ('baseline', 'v_silu'):
            seeds = {r['seed'] for r in rows if r['arm'] == arm}
            for seed in seeds:
                rr = [r for r in rows if r['arm'] == arm and r['seed'] == seed]
                total = int(rr[0]['total_steps'])
                require(sorted(int(r['step']) for r in rr) == list(range(0, total + 1, total // 16)),
                        'Expected 17 curve checkpoints per run')
        final = stats([r for r in rows if r['step'] == r['total_steps']], 'dev_nll')
        development[phase] = {a: mean(v) for a, v in final.items()}
        for value in development[phase].values():
            require(f'{value:.3f}' in paper, 'Development endpoint missing from paper')

    old = stats(load('results/studies/M/results.json')['data']['rows'])
    for lr, values in ((0.0006, old), (0.0003, medium)):
        check_row('tab:medium_lr_confirmation', ['Held-out', f'{lr:g}',
                  tex_ppl(values['baseline']), tex_ppl(values['v_silu']),
                  f"${pct(values['v_silu'], values['baseline']):.2f}\\%$"])
    cells = load('results/cross_eval/summary.json')['cell_statistics']
    for label, keys in [('Linear (baseline)', ('BB', 'BS')), ('SiLU (GatedV)', ('GB', 'GS'))]:
        fields = [f"${cells[k]['mean_ppl']:,.3f}\\pm{cells[k]['sample_sd_ppl']:.3f}$".replace(',', '{,}')
                  for k in keys]
        check_row('tab:cross_eval', [label, *fields])
    for match in re.finditer(r'\]\(([^)#]+)(?:#[^)]*)?\)', readme):
        target = match.group(1)
        if '://' not in target:
            require((ROOT / target).exists(), f'README link: {target}')
    report = {'status': 'PASS_PAPER_REPOSITORY_CONSISTENCY',
              'scope': 'Displayed current results, confirmation recipes and source hashes; no training rerun.',
              'main_results': main_results, 'development_final_mean_ppl': development,
              'sources_sha256': dict(sorted(SOURCES.items()))}
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(json.dumps({'status': report['status'], 'main_results': main_results}, indent=2))


if __name__ == '__main__':
    main()
