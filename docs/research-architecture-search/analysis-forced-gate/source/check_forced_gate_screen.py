"""Independent predeclared decision from terminal raw training results."""
from pathlib import Path
import hashlib,json
ROOT=Path('/Users/mr.z/ode-workspace-prep/architecture-search/forced_gate_20261024_m1')
OUT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[ROOT/'output'/f'forced_gate_{s}_20261024_m1/results.json' for s in ('v1','product16')]
assert json.loads((ROOT/'output/forced_gate_queue_20261024_m1/status.json').read_text())['state']=='COMPLETE'
rows=[]
for p in paths:
 assert json.loads((p.parent/'status.json').read_text())['state']=='COMPLETE'
 rows+=json.loads(p.read_text())
assert len(rows)==15
lookup={(r['dataset'],r['label']):r for r in rows};tasks=('gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d');refs=('swiglu_budget','swiglu')
decisions={}
for arm in ('forced_k2','forced_exact'):
 qualifications=[];changes={}
 for task in tasks:
  r=lookup[task,arm];base=[lookup[task,b] for b in refs];static=lookup[task,'forced_static'];k2=lookup[task,'forced_k2']
  changes[task]=dict(oracle_percent_vs_each_swiglu={b:100*(r['validation_oracle_mse']/lookup[task,b]['validation_oracle_mse']-1) for b in refs},oracle_percent_vs_static=100*(r['validation_oracle_mse']/static['validation_oracle_mse']-1),oracle_percent_vs_K2=100*(r['validation_oracle_mse']/k2['validation_oracle_mse']-1),jacobian_percent_vs_better=100*(r['jacobian_error']/min(b['jacobian_error'] for b in base)-1),SW2_percent_vs_better=100*(r['validation_sw2']/min(b['validation_sw2'] for b in base)-1))
  if task==tasks[0]:continue
  improves=all(r['validation_oracle_mse']<=.9*b['validation_oracle_mse'] for b in base) and r['validation_oracle_mse']<=.95*static['validation_oracle_mse']
  primary=all(lookup[t,arm]['validation_oracle_mse']<=1.05*min(lookup[t,b]['validation_oracle_mse'] for b in refs) for t in tasks if t!=task)
  secondary=all(lookup[t,arm][m]<=1.1*min(lookup[t,b][m] for b in refs) for t in tasks[1:] for m in ('jacobian_error','validation_sw2'))
  if improves and primary and secondary:qualifications.append(task)
 decisions[arm]=dict(passes=bool(qualifications),qualifying_tasks=qualifications,higher_terms_passes=(arm=='forced_exact' and any(lookup[t,arm]['validation_oracle_mse']<=.95*lookup[t,'forced_k2']['validation_oracle_mse'] for t in qualifications)),changes=changes)
result=dict(state='NOMINATION' if any(r['passes'] for r in decisions.values()) else 'STOP_FAMILY',decisions=decisions,input_sha256={str(p):sha(p) for p in paths},script_sha256=sha(Path(__file__)),scope='Independent arithmetic from COMPLETE raw results; integrity requires separate native audits/replays.')
(OUT/'independent-screen-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
