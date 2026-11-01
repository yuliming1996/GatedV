"""Check the three single-seed G2 development endpoints and their provenance."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results/g2_headwise'
load = lambda path: json.loads(path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest = load(RESULTS/'manifest.json')
    for path, digest in manifest['files_sha256'].items():
        if sha(RESULTS/path) != digest:
            raise ValueError(f'Changed G2 evidence: {path}')
    plan, flight = load(RESULTS/'protocol.json'), load(RESULTS/'preflight.json')
    if sha(RESULTS/'protocol.json') != flight['protocol_sha256']:
        raise ValueError('G2 protocol binding differs')
    audit = load(RESULTS/'training_audit.json')
    if audit['status'] != 'PASS_THREE_G2_DEVELOPMENT_ENDPOINTS':
        raise ValueError('G2 training audit did not pass')
    rows = manifest['rows']
    if len(rows) != 3 or {r['peak_lr'] for r in rows} != {0.0006,0.0003,0.0001}:
        raise ValueError('Expected the three declared G2 learning rates')
    with (ROOT/'results/value_gate_grid/endpoints.csv').open() as f:
        grid = {(r['arm'],float(r['peak_lr'])):r for r in csv.DictReader(f)}
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/'experiments/g2_headwise/code'/name) != digest:
            raise ValueError(f'Changed training source: {name}')
    for row in rows:
        job = next(j for j in plan['jobs'] if j['id'] == row['run_id'])
        path = ROOT/'experiments/g2_headwise'/job['config']
        if sha(path) != job['sha256']:
            raise ValueError('G2 recipe digest differs')
        cfg = load(path)
        if (cfg['arm'],cfg['seed'],cfg['lr'],cfg['steps'],cfg['batch'],cfg['context']) != (
                'v_g2_headwise',20270201,row['peak_lr'],327680,8,256):
            raise ValueError('G2 training recipe differs')
        folder = RESULTS/row['run_id']
        result = load(folder/'result.json')
        identity = result['identity']
        if (identity != load(folder/'manifest.json')['identity'] or identity['config'] != cfg
                or identity['schema'] != 'headwise_g2_v1'
                or identity['source_sha256'] != plan['source_sha256']
                or identity['data_sha256'] != plan['data']['sha256']
                or identity['environment'] != flight['environment']
                or identity['initial_parameter_sha256'] != flight['initial_sha256']):
            raise ValueError('G2 run identity differs')
        if (result['status'],result['steps'],result['tokens'],result['model']['parameters']) != (
                'complete',327680,671088640,26186112):
            raise ValueError('G2 endpoint is incomplete')
        if (sha(folder/'dev_final.npz') != row['arrays_sha256']
                or sha(folder/'result.json') != row['result_sha256']):
            raise ValueError('G2 endpoint digest differs')
        for name in ['dev_final.json','dev_final.npz']:
            if sha(folder/name) != result['artifacts'][name]:
                raise ValueError('G2 result artifact binding differs')
        with np.load(folder/'dev_final.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['counts'],np.full(2048,256))
            losses = z['loss_sums']
            if losses.shape != (2048,) or not np.isfinite(losses).all() or (losses < 0).any():
                raise ValueError('Invalid G2 losses')
            nll = math.fsum(losses)/524288
            ppl = math.exp(nll)
            np.testing.assert_allclose([nll,ppl], [row['dev_nll'],row['dev_ppl']],rtol=0,atol=1e-12)
            np.testing.assert_allclose(nll,result['dev']['nll'],rtol=0,atol=1e-12)
            for arm,field in [('baseline','baseline_ppl'),('v_silu','gatedv_ppl')]:
                ref = grid[arm,row['peak_lr']]
                refpath = ROOT/'results/value_gate_grid'/(ref['run_id']+'.npz')
                if sha(refpath) != ref['arrays_sha256']:
                    raise ValueError('Comparison array digest differs')
                with np.load(refpath,allow_pickle=False) as a:
                    for key in ['offsets','doc_ids','counts']:
                        np.testing.assert_array_equal(a[key],z[key])
                    refppl = math.exp(math.fsum(a['loss_sums'])/int(a['counts'].sum()))
                np.testing.assert_allclose(refppl,row[field],rtol=0,atol=1e-12)
        np.testing.assert_allclose([100*(ppl/row['baseline_ppl']-1),100*(row['gatedv_ppl']/ppl-1)],
                                   [row['g2_vs_baseline_pct'],row['gatedv_vs_g2_pct']],rtol=0,atol=1e-12)
    if (manifest['training_updates'],manifest['training_targets']) != (983040,2013265920):
        raise ValueError('G2 aggregate budget differs')
    if min(rows,key=lambda r:(r['dev_nll'],r['peak_lr']))['peak_lr'] != manifest['selected_peak_lr']:
        raise ValueError('G2 development selection differs')
    return rows


if __name__ == '__main__':
    rows = verify()
    print('PASS: three G2 development endpoints, matching windows, source and recipe bindings.')
    print('Peak LR    Baseline    GatedV   Headwise G2')
    for r in rows:
        print(f"{r['peak_lr']:7g}  {r['baseline_ppl']:10.3f} {r['gatedv_ppl']:9.3f} {r['dev_ppl']:13.3f}")
