"""Copied native evaluator; replay native first, then substitute one hidden solver."""
import argparse,json,math,time,sys
from pathlib import Path
import numpy as np
from rk4_trial import HERE,base,torch,build_model,StateRK4Layer
from state_trial import StateFlowLayer
sys.path.insert(0,str(HERE.parent/'order_response_20261115'))
from evaluate_order import parent as ev
ev.build_model=build_model

def assets(root,dataset,split,device):
 manifest=json.loads((Path(root)/'manifest.json').read_text())
 row=next(r for r in manifest['rows'] if r['dataset']==dataset and r['split']==split)
 file=Path(row['file']);assert base.digest(file)==row['sha256']
 with np.load(file) as z:arrays={k:z[k].copy() for k in z.files}
 assert {k:ev.array_hash(v) for k,v in arrays.items()}==row['arrays']
 fixed={k:torch.from_numpy(arrays['field_'+k]).to(device) for k in ('x','t','y','oracle')}
 fixed['directions']=torch.from_numpy(arrays['directions']).to(device)
 return arrays,fixed,row

def cases(values,law):
 if law.dim==2:return ev.mode_stats(values,law)
 latent=values.astype(float)@law.rotation
 obs=[];expected=[]
 for axis in (0,7,15):
  m=law.locations[axis];s=math.sqrt(law.variances[axis])
  phi=lambda z:(1+math.erf(z/math.sqrt(2)))/2
  expected.append(phi(-m/(2*s))-phi(-3*m/(2*s)))
  obs.append(float(np.mean(np.abs(latent[:,axis])<m/2)))
 return dict(gap_axes=[0,7,15],gap_mass=obs,gap_expected=expected,
             gap_error_max=float(np.max(np.abs(np.array(obs)-expected))))

def substitute_solver(model, solver):
 old = {name:value.detach().clone() for name,value in model.state_dict().items()}
 count = model.parameter_count()
 for _,layer in model.ode_layers():
  layer.__class__ = StateFlowLayer if solver=='midpoint' else StateRK4Layer
  layer.substeps = 4 if solver=='midpoint' else 2
  layer.orders = 2 if solver=='midpoint' else 4
 assert count==model.parameter_count()
 assert all(torch.equal(value,model.state_dict()[name]) for name,value in old.items())


def main():
 p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--data-root',required=True)
 p.add_argument('--assets',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
 p.add_argument('--split',choices=['selection','confirmation'],default='selection');p.add_argument('--steps',nargs='+',type=int,default=[15000,20000])
 p.add_argument('--witness',action='store_true');p.add_argument('--solver',required=True,choices=['midpoint','rk4'])
 a=p.parse_args();torch.set_num_threads(1);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 folder=Path(a.folder);cfg=json.loads((folder/'config.json').read_text());cell=cfg['cell']
 assert (folder/'ADAPTER_COMPLETE').exists()
 cache=base.verify_data(a.data_root,cell);manifest=cache/'manifest.json'
 law=base.train.load_distribution(json.loads(manifest.read_text())['datasets'][cell['dataset']])
 arrays,fixed,binding=assets(a.assets,cell['dataset'],a.split,a.device)
 base.dump(out/'binding.json',dict(source_folder=str(folder),source_config=cfg,assets=binding,split=a.split,
           heun_steps=128,banks=8,samples_per_bank=4096,inference_solver=a.solver,source_sha256=base.digest(__file__)))
 rows=[]
 for step in a.steps:
  model,identity=ev.restore(folder,cfg,manifest,step,a.device);snapshot=ev.tensor_snapshot(model)
  if step==20000:
   native=base.train.fixed_eval_data(cache/cell['dataset'],law,law.dim,0,torch.device(a.device),4096,256,'test')
   generated=base.train.generated_samples(model,native['sample_source'],32).cpu().numpy()
   with np.load(folder/'final_test_examples.npz') as z:assert np.array_equal(generated,z['generated'])
   base.dump(out/'native-anchor.json',dict(passed=True,exact=True,**identity))
  substitute_solver(model,a.solver)
  prediction,mse=ev.field(model,fixed);assert mse is not None
  np.save(out/f'field-{step}.npy',prediction,allow_pickle=False)
  eq=ev.equivariance(model,fixed['x'][:1024],fixed['t'][:1024]) if law.dim==2 else None
  count=1 if a.witness else 8;generated=[];scores=[];before=time.monotonic()
  for bank in range(count):
   source=torch.from_numpy(arrays[f'source_{bank}'][:64] if a.witness else arrays[f'source_{bank}']).to(a.device)
   target=torch.from_numpy(arrays[f'target_{bank}'][:64] if a.witness else arrays[f'target_{bank}']).to(a.device)
   value=base.train.generated_samples(model,source,128);assert bool(torch.isfinite(value).all())
   score=base.train.sliced_w2(value.double(),target.double(),fixed['directions'].double())
   generated.append(value.cpu().numpy());scores.append(score)
   np.save(out/f'{step}-b{bank}.npy',generated[-1],allow_pickle=False)
   base.dump(out/'progress.json',dict(step=step,banks_completed=bank+1,seconds=time.monotonic()-before))
  value=np.concatenate(generated);target=np.concatenate([arrays[f'target_{i}'][:64] if a.witness else arrays[f'target_{i}'] for i in range(count)])
  pool=base.train.sliced_w2(torch.from_numpy(value).to(a.device).double(),torch.from_numpy(target).to(a.device).double(),fixed['directions'].double())
  rows.append(dict(step=step,inference_solver=a.solver,field_mse=mse,sw2_mean=float(np.mean(scores)),sw2_sd=float(np.std(scores,ddof=1)) if count>1 else 0.,
                   pooled_sw2=pool,bank_sw2=scores,seconds=time.monotonic()-before,equivariance=eq,**cases(value,law),**identity))
  base.dump(out/'summary.json',rows);ev.same_tensors(model,snapshot)
 base.dump(out/'COMPLETE.json',dict(passed=True,witness=a.witness,split=a.split,steps=a.steps,parameters_unchanged=True))
 print(json.dumps(dict(passed=True,output=str(out))))

if __name__=='__main__':main()
