"""Independent raw endpoint screen and historical Euler initialization check."""
from pathlib import Path
import hashlib,json
import torch
ROOT=Path('/Users/mr.z/ode-workspace-prep/architecture-search/state_clock_20261025_m1');OLD=ROOT.parent/'shared_features_20261024_m1';OUT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs={};rows=[];initial=[]
for suffix in ('v1','product16'):
 study=ROOT/'output'/f'state_clock_{suffix}_20261025_m1';assert json.loads((study/'status.json').read_text())['state']=='COMPLETE'
 path=study/'results.json';inputs[str(path)]=sha(path);rows+=json.loads(path.read_text())
 newcfgpath=study/'configs/state_euler.json';oldstudy=OLD/'output'/f'shared_features_{suffix}_20261024_m1';oldcfgpath=oldstudy/'configs/vector_parent.json'
 newcfg=json.loads(newcfgpath.read_text());oldcfg=json.loads(oldcfgpath.read_text());inputs[str(newcfgpath)]=sha(newcfgpath);inputs[str(oldcfgpath)]=sha(oldcfgpath)
 assert oldcfg['model_options']['ode_solve_init']=='constant_paired'
 for dataset in (('gaussian_aniso_2d','gmm_ring8_2d') if suffix=='v1' else ('rotated_product_bimodal_16d',)):
  def checkpoint(study,cfg,kind,step):return study/'runs'/cfg['experiment_id']/dataset/kind/'seed000/checkpoints'/f'step_{step:07d}.pt'
  oldpath=checkpoint(oldstudy,oldcfg,'ode_vector',0);newpath=checkpoint(study,newcfg,'ode_state_clock',0);inputs[str(oldpath)]=sha(oldpath);inputs[str(newpath)]=sha(newpath)
  a=torch.load(oldpath,weights_only=False,map_location='cpu');b=torch.load(newpath,weights_only=False,map_location='cpu')
  for key in ('model','runtime_buffers'):
   assert set(a[key])==set(b[key])
   for k in a[key]:torch.testing.assert_close(a[key][k],b[key][k],atol=0,rtol=0)
  for key in ('torch_rng','data_rng'):torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
  assert all(torch.count_nonzero(v)==0 for k,v in a['model'].items() if '.t_func.2.' in k)
  # Do not infer bit-identical trained trajectories from matching starts.
  ap=checkpoint(oldstudy,oldcfg,'ode_vector',5000);bp=checkpoint(study,newcfg,'ode_state_clock',5000);inputs[str(ap)]=sha(ap);inputs[str(bp)]=sha(bp)
  A=torch.load(ap,weights_only=False,map_location='cpu')['model'];B=torch.load(bp,weights_only=False,map_location='cpu')['model'];maximum=max(float((A[k]-B[k]).abs().max()) for k in A)
  initial.append(dict(dataset=dataset,old_init_option=oldcfg['model_options']['ode_solve_init'],old_order=oldcfg['model_options']['orders'],new_solver=newcfg['model_options']['state_clock_solver'],model_buffers_torch_and_data_RNG_bit_equal=True,all_final_controller_affines_zero_at_start=True,final_weight_max_difference=maximum,scope='Constant clocks were already used by the recent Stage17 parent. No bit-identical final Euler replay is claimed.'))
lookup={(r['dataset'],r['label']):r for r in rows};tasks=('gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d');baselines=('swiglu_budget','swiglu','swiglu_compute');controls=('state_euler','state_frozen');details={};leads=[]
for task in tasks:
 r=lookup[task,'state_midpoint'];mse=r['validation_oracle_mse'];best={k:min(lookup[task,b][k] for b in baselines) for k in ('validation_oracle_mse','jacobian_error','validation_sw2')}
 details[task]=dict(percent_vs_each_swiglu={b:100*(mse/lookup[task,b]['validation_oracle_mse']-1) for b in baselines},percent_vs_ODE_controls={c:100*(mse/lookup[task,c]['validation_oracle_mse']-1) for c in controls},ratio_to_best={k:r[k]/v for k,v in best.items()})
 if task in tasks[1:] and all(mse<=.9*lookup[task,b]['validation_oracle_mse'] for b in baselines) and all(mse<=.95*lookup[task,c]['validation_oracle_mse'] for c in controls):leads.append(task)
primary=all(details[t]['ratio_to_best']['validation_oracle_mse']<=1.05 for t in tasks);secondary=all(details[t]['ratio_to_best'][k]<=1.1 for t in tasks[1:] for k in ('jacobian_error','validation_sw2'));decision='NOMINATION' if leads and primary and secondary else 'STOP_SETTING'
audit=json.loads((OUT/'screen-decision.json').read_text());assert audit['state']==decision and audit['qualifying_nonlinear_leads']==leads
result=dict(state='PASS',registered_decision=decision,qualifying_nonlinear_tasks=leads,all_primary_guard=primary,secondary_guards=secondary,details=details,Euler_historical_initialization=initial,input_sha256=inputs,script_sha256=sha(Path(__file__)))
(OUT/'independent-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))
