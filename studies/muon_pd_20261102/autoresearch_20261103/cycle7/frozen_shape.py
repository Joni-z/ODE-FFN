"""Registered P spectrum/scale interventions; unchanged native evaluator."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

def variants(weight, strength):
    if strength == 0:
        return weight.clone(), weight.clone()
    x=weight.detach().double().cpu()
    if not x.norm():
        return weight.clone(),weight.clone()
    u,s,vh=torch.linalg.svd(x,full_matrices=False)
    alpha=s.norm()/np.sqrt(len(s))
    blended=(1-strength)*s+strength*alpha
    target=s.norm()*blended/blended.norm()
    shape=((u*target)@vh).to(weight)
    scale=(target[0]/s[0]*x).to(weight)
    return shape,scale

def summary(weight):
    s=torch.linalg.svdvals(weight.detach().double().cpu())
    return dict(singular_values=s.tolist(),frobenius=float(s.norm()),sigma_max=float(s[0]),
                sigma_min=float(s[-1]),stable_rank=float(s.square().sum()/s[0].square()) if s[0] else 0.)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--self-test',action='store_true');p.add_argument('--dataset');p.add_argument('--old-output');p.add_argument('--cycle3-output');p.add_argument('--cycle6-output');p.add_argument('--save',required=True)
    a=p.parse_args();source=Path(a.source).resolve();sys.path.insert(0,str(source/'autoresearch_20261103/cycle3'))
    from trial3 import base,build_model
    base.verify_sources();torch.set_num_threads(1);device=torch.device(a.device)
    if a.self_test:
        torch.manual_seed(0)
        matrices=[torch.randn(64,64),torch.diag(torch.arange(64).float()),torch.eye(64),torch.zeros(64,64)]
        error=0.
        for weight in matrices:
            weight=weight.to(device);native=summary(weight)
            for k in [0.,.02,.1,.5,1.]:
                shape,scale=variants(weight,k);s=summary(shape);c=summary(scale)
                if k==0:assert torch.equal(weight,shape) and torch.equal(weight,scale)
                err=abs(s['frobenius']-native['frobenius'])/max(native['frobenius'],1.)
                error=max(error,err);assert err<2e-6
                assert abs(s['sigma_max']-c['sigma_max'])<2e-6*max(native['sigma_max'],1.)
                assert s['sigma_max']<=native['sigma_max']+2e-6*max(native['sigma_max'],1.)
                if k==1 and native['frobenius']:assert s['sigma_max']-s['sigma_min']<2e-6*max(s['sigma_max'],1.)
        for family in ['scalar_k2','pd_direct_raw_k1']:
            torch.manual_seed(0);model=build_model(2,family).to(device);x=torch.randn(16,2,device=device);t=torch.rand(16,1,device=device)
            with torch.no_grad():
                original=model(x,t).clone()
                for _,weight in base.projection_parameters(model):weight.copy_(variants(weight,0)[0])
                assert torch.equal(original,model(x,t))
        base.dump(a.save,dict(passed=True,self_test=True,device=str(device),torch=torch.__version__,max_relative_energy_error=error,source_sha256=base.digest(__file__)))
        return
    assert a.dataset in ['gmm_ring8_2d','rotated_product_bimodal_16d']
    assert all([a.old_output,a.cycle3_output,a.cycle6_output])
    ring=a.dataset=='gmm_ring8_2d';ds=a.dataset
    settings=[('scalar_k2',.006 if ring else .02,20000,'muon_pd_20261102',Path(a.old_output)),
              ('pd_direct_raw_k1',.02 if ring else .006,20000,'professor_autoresearch_cycle3_20261103',Path(a.cycle3_output)),
              ('pd_direct_raw_k1',.02 if ring else .006,80000,'professor_autoresearch_cycle6_20261103',Path(a.cycle6_output))]
    rows=[];bindings=[]
    for family,lr,step,experiment,root in settings:
        cell=f'{ds}__{family}__muon_{lr}'+('__budget80000' if step==80000 else '')
        folder=root/experiment/ds/cell/'seed000';file=folder/f'checkpoints/step_{step:07d}.pt'
        assert (folder/'ADAPTER_COMPLETE').exists()
        cfg=json.loads((folder/'config.json').read_text());state=torch.load(file,map_location=device,weights_only=False)
        manifest_file=Path(cfg['data_root'])/'manifest.json'
        assert state['step']==step and state['config_sha256']==base.digest(folder/'config.json') and state['dataset_manifest_sha256']==base.digest(manifest_file)
        assert cfg['steps']==step and cfg['heun_steps']==32 and cfg['sample_count']==4096 and cfg['seeds']==[0]
        spec=json.loads(manifest_file.read_text())['datasets'][ds]
        fixed=base.train.fixed_eval_data(Path(cfg['data_root'])/ds,base.train.load_distribution(spec),spec['dim'],0,device,4096,256,'test')
        torch.manual_seed(0);model=build_model(spec['dim'],family).to(device);base.train.restore_model(model,state);model.eval()
        selected=base.projection_parameters(model);assert len(selected)==6 and all(n.endswith('.proj.weight') for n,_ in selected)
        original={name:weight.detach().clone() for name,weight in selected};before={n:summary(w) for n,w in original.items()}
        native_test=json.loads((folder/'final_test_sampling.json').read_text())
        with np.load(folder/'final_test_examples.npz',allow_pickle=False) as saved:saved_generated=saved['generated'].copy();assert np.array_equal(saved['target'],fixed['sample_target'].cpu().numpy())
        binding=dict(cell=cell,checkpoint_sha256=base.digest(file),config_sha256=state['config_sha256'],manifest_sha256=state['dataset_manifest_sha256'],before=before)
        bindings.append(binding);anchor_pred=None
        for branch,k in [('anchor',0.)]+[(b,k) for k in [.02,.1,.5,1.] for b in ['shape','scale']]:
            with torch.no_grad():
                for name,weight in selected:
                    new=original[name] if branch=='anchor' else variants(original[name],k)[branch=='scale']
                    weight.copy_(new)
                after={n:summary(w) for n,w in selected}
                prediction=torch.cat([model(x,t) for x,t in zip(fixed['x'].split(1024),fixed['t'].split(1024))]).double()
                assert torch.isfinite(prediction).all(),'Nonfinite field prediction: retain failed invocation and diagnose'
                field=float((prediction-fixed['oracle'].double()).square().mean())
                if branch=='anchor':anchor_pred=prediction.clone()
                displacement=float((prediction-anchor_pred).square().mean())
                generated=base.train.generated_samples(model,fixed['sample_source'],32)
                finite=bool(torch.isfinite(generated).all());fraction=float(torch.isfinite(generated).all(1).float().mean())
                sw2=base.train.sliced_w2(generated,fixed['sample_target'],fixed['directions']) if finite else None
                metric_finite=sw2 is not None and math.isfinite(sw2)
                if not metric_finite:sw2=None
            constraint=max(abs(after[n]['frobenius']/before[n]['frobenius']-1) for n in before)
            if branch in ['anchor','shape']:assert constraint<2e-6
            if branch=='anchor':
                replay=float(np.max(np.abs(generated.cpu().numpy()-saved_generated)));assert replay<2e-5
                assert abs(field-native_test['validation_oracle_mse'])<2e-6 and abs(sw2-native_test['sample']['sliced_w2'])<2e-6
            row=dict(cell=cell,step=step,family=family,branch=branch,strength=k,validation_field_mse=field,field_prediction_displacement_mse=displacement,test_sw2=sw2,finite_generated_fraction=fraction,sampling_metric_finite=metric_finite,matrices=after,max_relative_frobenius_change=constraint)
            if branch=='anchor':row.update(generated_replay_max_error=replay,field_replay_error=abs(field-native_test['validation_oracle_mse']),sw2_replay_error=abs(sw2-native_test['sample']['sliced_w2']),test_reference_distance=native_test['sample']['reference_floor'])
            rows.append(row)
            base.dump(a.save,dict(complete=False,dataset=ds,rows=rows,bindings=bindings,analysis_source_sha256=base.digest(__file__)))
            print(json.dumps({k:v for k,v in row.items() if k!='matrices'}),flush=True)
    assert len(rows)==27
    base.dump(a.save,dict(complete=True,dataset=ds,device=str(device),torch=torch.__version__,rows=rows,bindings=bindings,analysis_source_sha256=base.digest(__file__),scope='Frozen simultaneous P interventions; no training, checkpoint selection or modified primary evaluator.'))

if __name__=='__main__':main()
