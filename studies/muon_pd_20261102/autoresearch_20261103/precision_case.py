"""Frozen precision and complete-vector-field Jacobian diagnostics, no training."""
import argparse,json,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--old-output',required=True);p.add_argument('--cycle2-output',required=True);p.add_argument('--dataset',required=True);p.add_argument('--save',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
source=Path(a.source);sys.path.insert(0,str(source/'autoresearch_20261103/cycle2'))
from trial2 import base,build_model
base.verify_sources();torch.set_num_threads(1)
ds=a.dataset
settings=[('pd_direct_raw','muon',.02 if ds=='gmm_ring8_2d' else .006,'professor_autoresearch_cycle2_20261103',Path(a.cycle2_output)),('scalar_k2','muon',.006 if ds=='gmm_ring8_2d' else .02,'muon_pd_20261102',Path(a.old_output)),('swiglu128','muon' if ds=='gmm_ring8_2d' else 'adamw',.02 if ds=='gmm_ring8_2d' else .0003,'muon_pd_20261102',Path(a.old_output))]
rows=[]
for family,method,lr,experiment,root in settings:
 cell=f'{ds}__{family}__{method}_{lr}';folder=root/experiment/ds/cell/'seed000';cfg=json.loads((folder/'config.json').read_text());file=folder/'checkpoints/step_0020000.pt'
 state=torch.load(file,map_location=a.device,weights_only=False);assert state['config_sha256']==base.digest(folder/'config.json')
 manifest=json.loads((Path(cfg['data_root'])/'manifest.json').read_text());spec=manifest['datasets'][ds];device=torch.device(a.device)
 fixed=base.train.fixed_eval_data(Path(cfg['data_root'])/ds,base.train.load_distribution(spec),spec['dim'],0,device,4096,256,'test')
 torch.manual_seed(0);model=build_model(spec['dim'],family).to(device);base.train.restore_model(model,state);model.eval()
 with torch.no_grad():native=torch.cat([model(x,t) for x,t in zip(fixed['x'].split(1024),fixed['t'].split(1024))])
 native_j=[]
 for x,t in zip(fixed['probe_x'][:16],fixed['probe_t'][:16]):
  j=torch.autograd.functional.jacobian(lambda h:model(h[None],t[None])[0],x)
  native_j.append(float(torch.linalg.svdvals(j.double())[0]))
 model=model.double()
 with torch.no_grad():precise=torch.cat([model(x.double(),t.double()) for x,t in zip(fixed['x'].split(1024),fixed['t'].split(1024))])
 precise_j=[]
 for x,t in zip(fixed['probe_x'][:16],fixed['probe_t'][:16]):
  j=torch.autograd.functional.jacobian(lambda h:model(h[None],t[None].double())[0],x.double());precise_j.append(float(torch.linalg.svdvals(j)[0]))
 diff=precise-native.double();oracle=fixed['oracle'].double()
 rows.append(dict(cell=cell,checkpoint_sha256=base.digest(file),float32_field_mse=float((native.double()-oracle).square().mean()),float64_field_mse=float((precise-oracle).square().mean()),prediction_max_error=float(diff.abs().max()),prediction_rms_error=float(diff.square().mean().sqrt()),native_complete_field_jacobian_norms=native_j,float64_complete_field_jacobian_norms=precise_j))
 print(json.dumps(rows[-1]),flush=True)
result=dict(dataset=ds,device=a.device,torch=torch.__version__,checkpoint_step=20000,calculation='same frozen parameters and FP32 validation inputs, full-model float32 versus float64; includes time-embedding precision, no retraining',jacobian_probes=16,rows=rows)
base.dump(a.save,result)
