#!/usr/bin/env python3
"""Verify bundled numerical results without training, checkpoints, corpus, or GPU.
Requires Python 3.10+ and NumPy. This checks arithmetic and file integrity, not
original training execution or the semantic absence of data contamination.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math, statistics
import numpy as np
ROOT=Path(__file__).resolve().parent

def load(p):return json.loads((ROOT/p).read_text())
def study(s,name='results.json'):return load(f'studies/{s}/{name}')['data']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,label):
 if not math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-12):raise AssertionError(f'{label}: {a} != {b}')
def vector(a,b,label):
 assert len(a)==len(b),label
 for x,y in zip(a,b):close(x,y,label)
def summary(rows):
 return {arm:{'mean_nll':statistics.mean(r['nll'] for r in rows if r['arm']==arm),'sd_nll':statistics.stdev(r['nll'] for r in rows if r['arm']==arm),'mean_ppl':statistics.mean(r['ppl'] for r in rows if r['arm']==arm),'sd_ppl':statistics.stdev(r['ppl'] for r in rows if r['arm']==arm)} for arm in ['baseline','v_silu']}
def check_confirmation(s):
 d=study(s);rows=d['rows'];ss=summary(rows)
 for arm in ss:
  for k,x in ss[arm].items():close(x,d['summaries'][arm].get(k,d['summaries'][arm].get(k.replace('sd_','sample_sd_'))),s+arm+k)
 seeds=sorted({r['seed'] for r in rows});lookup={(r['arm'],r['seed']):r for r in rows}
 delta=[lookup['v_silu',seed]['nll']-lookup['baseline',seed]['nll'] for seed in seeds]
 pct=[100*(lookup['v_silu',seed]['ppl']/lookup['baseline',seed]['ppl']-1) for seed in seeds]
 ratio=100*(ss['v_silu']['mean_ppl']/ss['baseline']['mean_ppl']-1)
 c=d['comparison'];vector(delta,c.get('paired_nll_differences',c.get('nll_differences')),s+'delta')
 vector(pct,c.get('paired_ppl_change_pct',c.get('ppl_changes_pct_by_seed')),s+'paired_pct');close(ratio,c['mean_ppl_change_pct'],s+'mean PPL ratio')
 u=c['seed_uncertainty'];t=u.get('t_critical',u.get('t_critical_two_sided_95'));m=statistics.mean(delta);sd=statistics.stdev(delta);se=sd/math.sqrt(len(seeds))
 ci=[m-t*se,m+t*se];vector(ci,u.get('t_ci95',u.get('mean_nll_difference_t_ci95')),s+'seed t CI')
 close(sd,u['sample_sd'],s+'paired sample SD');close(se,u['standard_error'],s+'SE');close(m,u['mean'],s+'paired mean')
 assert c.get('improved_seed_count',5 if c.get('five_of_five_improve') else -1)==sum(x<0 for x in delta)
 return {'seeds':seeds,'summaries':ss,'paired_nll_mean':m,'paired_nll_sample_sd':sd,'paired_nll_t_ci95':ci,'mean_ppl_change_pct':ratio}
def check_learning_curves():
 provenance=load('learning_curves_provenance.json')
 assert sha(ROOT/'learning_curves.csv')==provenance['csv_sha256']
 with (ROOT/'learning_curves.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==provenance['rows']==374
 points={x['source_summary']:x for x in provenance['point_provenance']}
 runs={}
 for row in rows:
  point=points[row['source_summary']]
  assert row['summary_sha256']==point['summary_sha256'] and row['arrays_sha256']==point['arrays_sha256']
  close(row['dev_nll'],point['raw_nll_fsum'],'curve saved NLL')
  close(math.exp(float(row['dev_nll'])),row['ppl'],'curve exp(NLL)')
  assert int(row['training_targets'])==int(row['step'])*2048
  key=tuple(row[k] for k in ['stage','phase','arm','seed','lr'])
  runs.setdefault(key,[]).append(int(row['step']))
 assert len(runs)==22 and all(len(v)==17 and len(set(v))==17 and min(v)==0 for v in runs.values())
 return {'rows':len(rows),'runs':len(runs),'scope':'Saved summary arithmetic and provenance bindings; omitted intermediate source arrays were not re-read.'}
def check_partial_glu():
 d=load('glu_lr_controls_20261015.json');rows=d['rows']
 assert d['status']=='PARTIAL_DEV_ONLY_5_OF_16' and len(rows)==5
 assert not d['holdout_evaluated'] and not d['full_learning_rate_selection_complete'] and not d['multi_seed_confirmation']
 audits={x['source_sha256'] for x in d['source_audits']}
 expected_parameters={'baseline':26158464,'v_silu':26158464,'v_glu_full':27927936,'v_glu_reduced':26186112}
 for r in rows:
  assert r['source_audit_sha256'] in audits and r['source_endpoint_audit_status']=='PASS_ENDPOINT'
  assert r['parameters']==expected_parameters[r['arm']] and r['seed']==20270201
  assert r['steps']==327680 and r['targets']==r['steps']*2048==671088640
  close(math.exp(r['dev_nll']),r['dev_ppl'],'GLU snapshot exp(NLL)')
  evaluations=r['development_evaluations'];assert len(evaluations)==17
  assert [x['label'] for x in evaluations]==['initial_dev']+[f'dev_step_{s}' for s in range(20480,327680,20480)]+['dev_final']
  for x in evaluations:
   close(math.exp(x['nll']),x['ppl'],'GLU saved curve exp(NLL)')
   assert x['tokens']==524288 and x['documents']==2048
   assert r['artifacts_sha256'][x['label']+'.npz']==x['arrays_sha256']
   assert r['artifacts_sha256'][x['label']+'.json']==x['summary_sha256']
  close(evaluations[-1]['nll'],r['dev_nll'],'GLU final NLL binding')
  for digest in r['artifacts_sha256'].values():assert len(digest)==64 and set(digest)<=set('0123456789abcdef')
 with (ROOT/'glu_lr_controls_20261015.csv').open() as f:csv_rows=list(csv.DictReader(f))
 assert len(csv_rows)==5
 for a,b in zip(rows,csv_rows):
  for k,v in b.items():
   if isinstance(a[k],(float,int)):close(a[k],v,'GLU CSV '+k)
   else:assert a[k]==v
 six={r['arm']:r for r in rows if r['peak_lr']==.0006};assert len(six)==4
 c=d['comparisons'];base=six['baseline']['dev_ppl'];gated=six['v_silu']['dev_ppl']
 for arm,r in six.items():close(100*(1-r['dev_ppl']/base),c['six_e4_ppl_reduction_vs_baseline_percent'][arm],'GLU fixed LR comparison')
 for arm in ['v_glu_full','v_glu_reduced']:close(100*(1-gated/six[arm]['dev_ppl']),c['six_e4_gatedv_ppl_reduction_vs_glu_percent'][arm],'GatedV versus GLU')
 low=next(r for r in rows if r['peak_lr']==.0003);assert low['arm']=='baseline'
 close(100*(1-low['dev_ppl']/base),c['baseline_ppl_reduction_3e4_vs_6e4_percent'],'baseline LR comparison')
 return {'status':d['status'],'endpoint_rows':len(rows),'development_summary_rows':sum(len(r['development_evaluations']) for r in rows),'comparisons':c,'scope':'Arithmetic of extracted audit summaries only; original GLU checkpoint, logs, and per-document arrays are not bundled or re-audited.'}
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive-root',type=Path,help='Optional parent directory containing the original nonlinear_qkv_20260925 archive');parser.add_argument('--output',type=Path);args=parser.parse_args()
 packaged=load('packaging_provenance.json')
 for entry in packaged['files']:assert sha(ROOT/entry['packaged_relative'])==entry['packaged_sha256'],entry['packaged_relative']
 ledger=load('source_ledger.json');byte_copies=0
 for e in ledger['entries']:
  p=ROOT/e['bundle_relative'];assert sha(p)==e['bundle_sha256'],f'Bundle hash: {p}'
  if e['representation']=='byte_identical_numeric_array_copy':assert e['bundle_sha256']==e['source_sha256'];byte_copies+=1
  if args.archive_root:assert sha(args.archive_root/e['source_relative'])==e['source_sha256'],e['source_relative']
 with (ROOT/'all_endpoints.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==133 and len({(r['stage'],r['job_id']) for r in rows})==133
 for r in rows:
  for k in ['seed','steps','target_exposures']:r[k]=int(r[k])
  for k in ['lr','nll','ppl']:r[k]=float(r[k])
  assert r['target_exposures']==r['steps']*2048
  close(math.exp(r['nll']),r['ppl'],r['job_id']+' exp(NLL)')
  # Bind each portable row to its retained original numeric source fields.
  d=load(r['source_json'])['data']
  if r['stage']=='A':source=next(x for x in d['rows'] if x['id']==r['job_id']);nll=source['dev_nll']
  elif r['stage']=='B':source=next(x for x in d['per_seed'] if x['job_id']==r['job_id']);nll=source['holdout_nll']
  elif r['stage']=='C':source=next(x for x in d['screen_rows_independently_recomputed']+d['long_rows_independently_recomputed'] if x['job_id']==r['job_id']);nll=source['dev_nll']
  else:source=next(x for x in d['rows']+d.get('selection_endpoints',[]) if x['job_id']==r['job_id']);nll=source['nll']
  close(r['nll'],nll,r['job_id']+' source NLL')
 budget=load('budget.json');assert sum(r['steps'] for r in rows)==budget['total_updates']==6063104
 assert sum(r['target_exposures'] for r in rows)==budget['total_target_exposures']==12417236992
 for b in budget['stages']:
  rr=[r for r in rows if r['stage']==b['stage']];assert len(rr)==b['runs'];assert sum(r['steps'] for r in rr)==b['updates']
 with (ROOT/'all_summary_cells.csv').open() as f:cells=list(csv.DictReader(f))
 for c in cells:
  rr=[r for r in rows if (r['stage'] in ['E','F'] if c['stage']=='EF' else r['stage']==c['stage']) and r['phase']==c['phase'] and r['arm']==c['arm'] and r['lr']==float(c['lr'])]
  assert len(rr)==int(c['seeds'])
  for k in ['nll','ppl']:
   close(statistics.mean(r[k] for r in rr),c['mean_'+k],'cell mean')
   if len(rr)>1:close(statistics.stdev(r[k] for r in rr),c['sample_sd_'+k],'cell sample SD')
 verified={'small':check_confirmation('D'),'medium':check_confirmation('M')}
 d=study('M');loss={};counts=None;offsets=None
 for r in d['rows']:
  p=ROOT/'medium_holdout'/(r['job_id']+'.npz');assert sha(p)==r['eval_arrays_sha256']
  with np.load(p,allow_pickle=False) as z:
   assert set(z.files)=={'loss_sums','counts','doc_ids','offsets'}
   assert z['loss_sums'].shape==(2048,) and np.isfinite(z['loss_sums']).all()
   assert np.array_equal(z['counts'],np.full(2048,256))
   assert np.array_equal(z['doc_ids'],np.arange(2048))
   assert counts is None or np.array_equal(counts,z['counts'])
   assert offsets is None or np.array_equal(offsets,z['offsets'])
   counts=z['counts'].copy();offsets=z['offsets'].copy();v=z['loss_sums'].copy();loss[r['arm'],r['seed']]=v
   close(math.fsum(float(x) for x in v)/int(counts.sum()),r['nll'],r['job_id']+' raw NLL')
 seeds=verified['medium']['seeds'];delta=np.stack([loss['v_silu',s]-loss['baseline',s] for s in seeds]).mean(axis=0)
 rng=np.random.Generator(np.random.PCG64(2026092901));boot=[]
 for _ in range(20):
  ids=rng.integers(0,2048,size=(100,2048));boot.extend((delta[ids].sum(axis=1)/counts[ids].sum(axis=1)).tolist())
 ci=np.quantile(boot,[.025,.975],method='linear').tolist();vector(ci,d['comparison']['domain_cluster']['nll_difference_ci95'],'medium bootstrap')
 vector(ci,d['comparison']['document_cluster']['nll_difference_ci95'],'medium same-partition bootstrap')
 verified['medium']['conditional_document_domain_nll_ci95']=ci
 slots=[load('medium_cost/'+f'slot_{i:03d}.json') for i in range(42)];times={};mem={arm:[] for arm in ['baseline','v_silu']}
 keys=['process_rss','cuda_allocated','cuda_reserved','cuda_peak_allocated','cuda_peak_reserved']
 for x in slots:
  ident=x['identity'];assert x['status']=='complete';assert ident['checkpoint_sha256']==x['checkpoint_sha256_after']
  for mode in ['forward','train_step']:
   vv=x[mode]['seconds'];assert len(vv)==10 and all(math.isfinite(v) and v>0 for v in vv)
   k=(ident['arm'],ident['seed'],ident['round'],mode);assert k not in times;times[k]=statistics.median(vv)
  mem[ident['arm']].append({k:max(m[k] for m in x['memory_samples']) for k in keys})
 cost={}
 for mode in ['forward','train_step']:
  mids=[]
  for seed in seeds:
   ratios=[times['v_silu',seed,rr,mode]/times['baseline',seed,rr,mode] for rr in range(7)];mids.append(statistics.median(ratios))
   expected=next(v for v in d['cost'][mode]['per_seed'] if v['seed']==seed);vector(ratios,expected['paired_round_ratios'],'cost round ratios');close(mids[-1],expected['median_round_ratio'],'cost seed median')
  v=statistics.mean(mids);close(v,d['cost'][mode]['mean_of_seed_medians'],'cost aggregate');cost[mode]={'ratio':v,'overhead_percent':100*(v-1),'seed_median_ratios':mids}
 for arm in mem:
  for k in keys:
   close(statistics.median(x[k] for x in mem[arm]),d['cost']['memory'][arm][k]['median'],'memory median')
   close(max(x[k] for x in mem[arm]),d['cost']['memory'][arm][k]['max'],'memory max')
 # Recompute finite LR choice independently. Exact ties select the lower LR.
 selection={a:min((r for r in d['selection_endpoints'] if r['arm']==a),key=lambda r:(r['nll'],r['lr']))['lr'] for a in ['baseline','v_silu']}
 assert selection==d['selection']['selected_lr_by_arm']=={'baseline':.0006,'v_silu':.0006}
 report={'status':'PASS','scope':'Portable file integrity and saved-number reanalysis only; not a training rerun, new evaluation, or independent replication. Domain uniqueness relies on the retained data audit; raw text/domain ledgers are intentionally omitted. Small-study bootstrap is retained from audited originals, not recomputed here.','checked_source_records':len(ledger['entries']),'byte_identical_numeric_array_files':byte_copies,'endpoint_rows':len(rows),'summary_cells':len(cells),'scientific_budget':budget,'confirmations':verified,'medium_selection':selection,'medium_raw_cost':cost,'original_source_hashes_checked':bool(args.archive_root)}
 report['learning_curves']=check_learning_curves()
 report['partial_glu_lr_snapshot']=check_partial_glu()
 report['packaged_files_checked']=len(packaged['files'])
 if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
