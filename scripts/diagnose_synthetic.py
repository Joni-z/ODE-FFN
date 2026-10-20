"""Read-only final-checkpoint diagnostics, including the SwiGLU comparison.

Writes a new analysis file; never changes training runs. Local Jacobian samples
are descriptive probes, not global Lipschitz bounds or independent replicates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from synthetic.models import VelocityNet
from synthetic.train import fixed_eval_data, generated_samples, load_distribution, sliced_w2


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(x):
    x=x.detach().double().reshape(-1)
    return dict(mean=x.mean().item(),median=x.median().item(),max=x.max().item(),
                min=x.min().item())


def polynomial(p,s,k):
    eye=torch.eye(p.shape[0],device=p.device,dtype=p.dtype)
    result=eye.expand(len(s),-1,-1).clone()
    term=result.clone()
    for n in range(1,k+1):
        term=(term@p)*s[:,None,None]/n
        result=result+term
    return result


def layer_diagnostics(model,x,t):
    inputs={}; hooks=[]
    for name,layer in model.ode_layers():
        hooks.append(layer.register_forward_pre_hook(
            lambda m,args,key=name:inputs.__setitem__(key,
                (args[0].detach()[:16],args[1].detach()[:16] if len(args)>1 and args[1] is not None else None))))
    with torch.no_grad():
        model(x,t)
    for hook in hooks:
        hook.remove()
    output=[]
    for name,layer in model.ode_layers():
        h=inputs[name][0].clone().requires_grad_(True)
        s=layer.step_size(h,inputs[name][1]).reshape(-1)
        ds=torch.autograd.grad(s.sum(),h,allow_unused=True)[0]
        ds=(ds if ds is not None else torch.zeros_like(h)).detach().double()
        h=h.detach().double(); s=s.detach().double()
        p=layer.operator().detach().double()
        symmetric=(p+p.T)/2
        skew=(p-p.T)/2
        a=polynomial(p,s,layer.orders)
        lower=polynomial(p,s,layer.orders-1)
        b=torch.einsum('bij,bj->bi',p@lower,h)
        correction=b[:,:,None]*ds[:,None,:]
        jac=a+correction
        exact=torch.matrix_exp(s[:,None,None]*p)
        delta=torch.einsum('bij,bj->bi',a-exact,h)
        out_exact=torch.einsum('bij,bj->bi',exact,h)
        output.append({'layer':name,'probe_count':len(h),'gate_source':layer.gate_source,
            'sigma_max_P':torch.linalg.matrix_norm(p,2).item(),
            'mu_P':torch.linalg.eigvalsh(symmetric).max().item(),
            'symmetric_fro':torch.linalg.matrix_norm(symmetric).item(),
            'skew_fro':torch.linalg.matrix_norm(skew).item(),
            'step':stats(s),'gate_gradient_norm':stats(ds.norm(dim=-1)),
            'frozen_polynomial_norm':stats(torch.linalg.matrix_norm(a,2)),
            'gate_correction_norm':stats(b.norm(dim=-1)*ds.norm(dim=-1)),
            'full_layer_jacobian_norm':stats(torch.linalg.matrix_norm(jac,2)),
            'exact_frozen_flow_norm':stats(torch.linalg.matrix_norm(exact,2)),
            'relative_action_truncation_error':stats(delta.norm(dim=-1)/out_exact.norm(dim=-1).clamp_min(1e-12))})
    return output


def time_diagnostics(model,fixed):
    records=[]
    for i in range(10):
        mask=(fixed['t'][:,0]>=i/10)&(fixed['t'][:,0]<(i+1)/10)
        x=fixed['x'][mask];t=fixed['t'][mask]
        with torch.no_grad():
            pred=model(x,t)
            pair=(pred-fixed['y'][mask]).square().mean().item()
            oracle=((pred-fixed['oracle'][mask]).square().mean().item()
                    if fixed['oracle'] is not None else None)
        # Shared actual x,t subset across architectures, max 128 per bin.
        xx=x[:128].detach().clone().requires_grad_(True)
        pp=model(xx,t[:128])
        jj=torch.stack([torch.autograd.grad(pp[:,j].sum(),xx,retain_graph=True)[0]
                        for j in range(pp.shape[1])],dim=1).detach()
        ss=(jj+jj.transpose(-1,-2))/2
        oo=(jj-jj.transpose(-1,-2))/2
        records.append({'t_left':i/10,'t_right':(i+1)/10,'count':len(x),
                        'pair_mse':pair,'oracle_mse':oracle,'jacobian_probe_count':len(xx),
                        'velocity_jacobian_sigma':stats(torch.linalg.matrix_norm(jj,2)),
                        'velocity_jacobian_mu':stats(torch.linalg.eigvalsh(ss)[:,-1]),
                        'velocity_jacobian_skew_fro':stats(torch.linalg.matrix_norm(oo))})
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output-root',type=Path,default=Path('../runs/synthetic'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','1')))
    root=Path(__file__).resolve().parents[1]
    runs=[]
    for config in ('train_v1.json','train_k1_v1.json','train_swiglu_v1.json'):
        grid=json.loads((root/'configs/synthetic'/config).read_text())
        for kind in grid['models']:
            folder=args.output_root/grid['experiment_id']/args.dataset/kind/'seed000'
            cfg=json.loads((folder/'config.json').read_text())
            provenance=json.loads((folder/'provenance.json').read_text())
            snapshot=folder/'checkpoints'/f'step_{cfg["steps"]:07d}.pt'
            assert (folder/'COMPLETE').read_text().strip()==str(cfg['steps'])
            manifest_path=root/cfg['data_root']/'manifest.json'
            assert digest(manifest_path)==provenance['dataset_manifest_sha256']
            manifest=json.loads(manifest_path.read_text())
            spec=manifest['datasets'][args.dataset]
            state=torch.load(snapshot,map_location=args.device,weights_only=False)
            assert state['step']==cfg['steps']
            assert state['identity']=={'dataset':args.dataset,'model':kind,'seed':0}
            assert state['config_sha256']==digest(folder/'config.json')
            assert state['dataset_manifest_sha256']==provenance['dataset_manifest_sha256']
            model=VelocityNet(spec['dim'],kind,**cfg['model_options']).to(args.device)
            model.load_state_dict(state['model'],strict=True);model.eval()
            fixed=fixed_eval_data(root/cfg['data_root']/args.dataset,load_distribution(spec),
                                  spec['dim'],0,args.device,cfg['sample_count'],cfg['sliced_directions'])
            samples={n:generated_samples(model,fixed['sample_source'],n) for n in (8,16,32,64,128,256)}
            sampling={str(n):{'nfe':2*n,'sw2':sliced_w2(xx,fixed['sample_target'],fixed['directions']),
                                  'endpoint_rms_to_heun256':(xx-samples[256]).square().mean().sqrt().item()}
                      for n,xx in samples.items()}
            last=json.loads((folder/'metrics.jsonl').read_text().splitlines()[-1])
            # Exact protocol reproduction within floating point hardware tolerance.
            assert abs(sampling['32']['sw2']-last['sample']['sliced_w2'])<5e-4
            result={'dataset':args.dataset,'model':kind,'orders':cfg['model_options']['orders'] if kind.startswith('ode_') else None,
                    'run_dir':str(folder.resolve()),'checkpoint_sha256':digest(snapshot),
                    'training_source_sha256':provenance['source_sha256'],
                    'parameters':provenance['model_parameters'],
                    'final_oracle_mse':last['validation_oracle_mse'],
                    'sampling':sampling,'time_bins':time_diagnostics(model,fixed),
                    'layers':layer_diagnostics(model,fixed['probe_x'],fixed['probe_t'])}
            runs.append(result)
            print(json.dumps({'dataset':args.dataset,'model':kind,'K':result['orders'],'sampling':sampling}),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'dataset':args.dataset,'seed':0,'device':args.device,
            'gpu':torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
            'torch':torch.__version__,'script_sha256':digest(__file__),
            'evaluation_source_sha256':{p.name:digest(p) for p in sorted((root/'synthetic').glob('*.py'))},
            'note':'Heun256 is a numerical reference, not ground truth; inspect Heun128-to256 error. Probes are local, not global bounds. Existing training outputs unchanged.',
            'runs':runs}
    with args.output.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':
    main()
