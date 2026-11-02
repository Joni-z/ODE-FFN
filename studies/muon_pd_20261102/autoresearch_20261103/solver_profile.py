"""Frozen external-solver resolution and validation-time error profiles."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--old-output',required=True);p.add_argument('--cycle2-output',required=True);p.add_argument('--dataset',required=True);p.add_argument('--save',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
source=Path(a.source);sys.path.insert(0,str(source/'autoresearch_20261103/cycle2'))
from trial2 import base,build_model
from audit_completed import independent_sw2
base.verify_sources();torch.set_num_threads(1);ds=a.dataset
settings=[('pd_direct_raw','muon',.02 if ds=='gmm_ring8_2d' else .006,'professor_autoresearch_cycle2_20261103',Path(a.cycle2_output)),('scalar_k2','muon',.006 if ds=='gmm_ring8_2d' else .02,'muon_pd_20261102',Path(a.old_output)),('swiglu128','muon' if ds=='gmm_ring8_2d' else 'adamw',.02 if ds=='gmm_ring8_2d' else .0003,'muon_pd_20261102',Path(a.old_output))]
rows=[]
for family,method,lr,experiment,root in settings:
    cell=f'{ds}__{family}__{method}_{lr}';folder=root/experiment/ds/cell/'seed000';cfg=json.loads((folder/'config.json').read_text());file=folder/'checkpoints/step_0020000.pt'
    state=torch.load(file,map_location=a.device,weights_only=False);assert state['config_sha256']==base.digest(folder/'config.json')
    manifest=json.loads((Path(cfg['data_root'])/'manifest.json').read_text());spec=manifest['datasets'][ds];device=torch.device(a.device)
    fixed=base.train.fixed_eval_data(Path(cfg['data_root'])/ds,base.train.load_distribution(spec),spec['dim'],0,device,4096,256,'test')
    torch.manual_seed(0);model=build_model(spec['dim'],family).to(device);base.train.restore_model(model,state);model.eval()
    target=fixed['sample_target'].cpu().numpy();directions=fixed['directions'].cpu().numpy()
    sampling=[]
    with np.load(folder/'final_test_examples.npz') as old:saved=old['generated'].copy();assert np.array_equal(old['target'],target)
    for steps in [32,64,128]:
        if device.type=='cuda':torch.cuda.synchronize()
        started=time.perf_counter();generated=base.train.generated_samples(model,fixed['sample_source'],steps)
        if device.type=='cuda':torch.cuda.synchronize()
        seconds=time.perf_counter()-started;values=generated.cpu().numpy();distance=independent_sw2(values,target,directions)
        if steps==32:assert np.max(np.abs(values-saved))<2e-5
        sampling.append(dict(heun_steps=steps,nfe=2*steps,numpy_test_sw2=distance,sampling_seconds=seconds,baseline_sample_max_error=float(np.max(np.abs(values-saved))) if steps==32 else None))
    with torch.no_grad():pred=torch.cat([model(x,t) for x,t in zip(fixed['x'].split(1024),fixed['t'].split(1024))])
    errors=(pred.double()-fixed['oracle'].double()).square().mean(dim=1).cpu().numpy();times=fixed['t'].cpu().numpy().ravel();profile=[]
    for i in range(10):
        mask=(times>=i/10)&(times<(i+1)/10 if i<9 else times<=1)
        profile.append(dict(t_start=i/10,t_end=(i+1)/10,count=int(mask.sum()),field_mse=float(errors[mask].mean())))
    row=dict(cell=cell,checkpoint_sha256=base.digest(file),sampling=sampling,time_profile=profile,sampling_floor=independent_sw2(target,fixed['sample_target2'].cpu().numpy(),directions))
    rows.append(row);print(json.dumps(row),flush=True)
base.dump(a.save,dict(dataset=ds,torch=torch.__version__,device=a.device,rows=rows,interpretation='Frozen checkpoints; primary Heun32 retained; higher NFE is not a theoretical truth.'))
