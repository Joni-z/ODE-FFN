"""Frozen predictor parity decomposition on the original validation bank."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--old-output',required=True);p.add_argument('--cycle2-output',required=True);p.add_argument('--cycle3-output',required=True);p.add_argument('--dataset',required=True);p.add_argument('--save',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
source=Path(a.source);sys.path.insert(0,str(source/'autoresearch_20261103/cycle3'))
from trial3 import base,build_model as builder3
from trial2 import build_model as builder2
base.verify_sources();torch.set_num_threads(1);ds=a.dataset;ring=ds=='gmm_ring8_2d';device=torch.device(a.device)
old=Path(a.old_output);two=Path(a.cycle2_output);three=Path(a.cycle3_output)
settings=[('scalar_k2','muon',.006 if ring else .02,'muon_pd_20261102',old),('scalar_k1','muon',.006 if ring else .02,'professor_autoresearch_cycle3_20261103',three),('pd_direct_raw','muon',.02 if ring else .006,'professor_autoresearch_cycle2_20261103',two),('pd_direct_raw_k1','muon',.02 if ring else .006,'professor_autoresearch_cycle3_20261103',three),('swiglu64','muon' if ring else 'adamw',.02 if ring else .001,'professor_autoresearch_cycle3_20261103',three),('swiglu128','muon' if ring else 'adamw',.02 if ring else .0003,'muon_pd_20261102',old)]
rows=[]
for family,method,lr,experiment,root in settings:
    cell=f'{ds}__{family}__{method}_{lr}';folder=root/experiment/ds/cell/'seed000';cfg=json.loads((folder/'config.json').read_text());file=folder/'checkpoints/step_0020000.pt'
    state=torch.load(file,map_location=device,weights_only=False);assert state['step']==20000 and state['config_sha256']==base.digest(folder/'config.json')
    manifest_file=Path(cfg['data_root'])/'manifest.json';assert state['dataset_manifest_sha256']==base.digest(manifest_file)
    spec=json.loads(manifest_file.read_text())['datasets'][ds];distribution=base.train.load_distribution(spec)
    fixed=base.train.fixed_eval_data(Path(cfg['data_root'])/ds,distribution,spec['dim'],0,device,4096,256,'test')
    x,t,u=fixed['x'],fixed['t'],fixed['oracle'];xn=x.cpu().numpy();tn=t.cpu().numpy()
    oracle_error=float(np.max(np.abs(distribution.velocity(-xn,tn)+distribution.velocity(xn,tn))))
    assert oracle_error<1e-10
    torch.manual_seed(0);model=(builder2 if family=='pd_direct_raw' else builder3)(spec['dim'],family).to(device);base.train.restore_model(model,state);model.eval()
    with torch.no_grad():
        positive=torch.cat([model(xx,tt) for xx,tt in zip(x.split(1024),t.split(1024))]).double()
        negative=torch.cat([model(-xx,tt) for xx,tt in zip(x.split(1024),t.split(1024))]).double()
    oracle=u.double();odd=(positive-negative)/2;even=(positive+negative)/2
    plus=(positive-oracle).square().mean(dim=1);minus=(negative+oracle).square().mean(dim=1)
    paired=(plus+minus)/2;odd_error=(odd-oracle).square().mean(dim=1);even_energy=even.square().mean(dim=1)
    residual=float((paired-odd_error-even_energy).abs().max());assert residual<1e-10
    final=json.loads((folder/'metrics.jsonl').read_text().splitlines()[-1]);replay_error=abs(float(plus.mean())-final['validation_oracle_mse']);assert replay_error<2e-6
    profile=[]
    for i in range(10):
        mask=(t[:,0]>=i/10)&(t[:,0]<(i+1)/10 if i<9 else t[:,0]<=1)
        profile.append(dict(t_start=i/10,t_end=(i+1)/10,count=int(mask.sum()),paired_field_mse=float(paired[mask].mean()),odd_field_mse=float(odd_error[mask].mean()),even_energy=float(even_energy[mask].mean())))
    row=dict(cell=cell,checkpoint_sha256=base.digest(file),manifest_sha256=base.digest(manifest_file),validation_count=len(x),original_validation_mse=float(plus.mean()),negative_validation_mse=float(minus.mean()),paired_field_mse=float(paired.mean()),odd_field_mse=float(odd_error.mean()),even_energy=float(even_energy.mean()),even_fraction=float(even_energy.mean()/paired.mean()),oracle_symmetry_max_error=oracle_error,decomposition_max_error=residual,original_field_replay_error=replay_error,time_profile=profile)
    rows.append(row);print(json.dumps(row),flush=True)
base.dump(a.save,dict(dataset=ds,torch=torch.__version__,device=a.device,rows=rows,interpretation='Frozen paired diagnostic; no training, odd projection sampling, changed primary metric or causal claim.'))
