"""Train equal-protocol unconditional FM models on the prepared synthetic data.

Run from repo root: python -m synthetic.train --dataset ... --model ... --seed ...
Outputs are uniquely named, never overwrite historical snapshots, and contain
per-checkpoint P spectra plus actual ODE-layer activation diagnostics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from synthetic.data import GaussianMixture, fm_batch
from synthetic.diagnostics import collect
from synthetic.models import VelocityNet


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_distribution(spec):
    if spec['family'] == 'gaussian_mixture':
        return GaussianMixture(**spec['distribution'])
    return None


def load_targets(path, dim):
    with np.load(path, allow_pickle=False) as data:
        if data.files != ['x1'] or data['x1'].ndim != 2 or data['x1'].shape[1] != dim:
            raise ValueError(f'bad target data: {path}')
        x = data['x1'].copy()
    if x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError(f'nonfinite or non-float32 targets: {path}')
    return x


def json_line(path, record):
    with Path(path).open('a') as f:
        f.write(json.dumps(record, allow_nan=False, sort_keys=True) + '\n')


def git_info(root):
    def run(*args):
        result = subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {'head':run('rev-parse','HEAD'), 'status':run('status','--short')}


def generated_samples(model, source, steps):
    if steps <= 0:
        raise ValueError('sampling steps must be positive')
    x=source.clone()
    dt=1.0/steps
    with torch.no_grad():
        for i in range(steps):
            t=torch.full((len(x),1),i*dt,device=x.device,dtype=x.dtype)
            velocity=model(x,t)
            predictor=x+dt*velocity
            t_next=torch.full_like(t,(i+1)*dt)
            x=x+0.5*dt*(velocity+model(predictor,t_next))
    return x


def sliced_w2(x,y,directions):
    if len(x) != len(y):
        raise ValueError('equal numbers of samples are required')
    u=x@directions.T
    v=y@directions.T
    return float(torch.sqrt((torch.sort(u,dim=0).values-torch.sort(v,dim=0).values).square().mean()).item())


def fixed_eval_data(folder, distribution, dim, seed, device, sample_count, directions_count):
    validation=load_targets(folder/'validation.npz',dim)
    rng=np.random.default_rng(seed+7761)
    vx,vt,vy=fm_batch(validation,rng)
    oracle=(distribution.velocity(vx,vt).astype(np.float32) if distribution is not None else None)
    with np.load(folder/'probe.npz',allow_pickle=False) as probe:
        px=probe['xt'][:256].copy()
        pt=probe['t'][:256].copy()
    test=load_targets(folder/'test.npz',dim)
    if len(test) < 2*sample_count:
        raise ValueError('test split must contain at least 2*sample_count targets')
    rng=np.random.default_rng(seed+4427)
    source=rng.standard_normal((sample_count,dim)).astype(np.float32)
    dirs=rng.standard_normal((directions_count,dim)).astype(np.float32)
    dirs/=np.linalg.norm(dirs,axis=1,keepdims=True).clip(min=1e-12)
    tensor=lambda x:torch.from_numpy(x).to(device)
    return {'x':tensor(vx),'t':tensor(vt),'y':tensor(vy),
            'oracle':tensor(oracle) if oracle is not None else None,
            'probe_x':tensor(px),'probe_t':tensor(pt),
            'sample_source':tensor(source),'sample_target':tensor(test[:sample_count]),
            'sample_target2':tensor(test[sample_count:2*sample_count]),
            'directions':tensor(dirs)}


@torch.no_grad()
def evaluate(model, fixed, step, train_loss, grad_norm, do_sampling, sample_steps):
    model.eval()
    count=0
    pair_sum=0.0
    oracle_sum=0.0
    for sl in range(0,len(fixed['x']),1024):
        x,t,y=[fixed[key][sl:sl+1024] for key in ('x','t','y')]
        pred=model(x,t)
        pair_sum+=float((pred-y).square().sum().item())
        if fixed['oracle'] is not None:
            oracle_sum+=float((pred-fixed['oracle'][sl:sl+1024]).square().sum().item())
        count+=pred.numel()
    record={'step':step,'train_pair_mse':train_loss,'grad_norm_before_clip':grad_norm,
            'validation_pair_mse':pair_sum/count,
            'validation_oracle_mse':oracle_sum/count if fixed['oracle'] is not None else None,
            'ode_layers':collect(model,fixed['probe_x'],fixed['probe_t'])}
    if do_sampling:
        start=time.monotonic()
        generated=generated_samples(model,fixed['sample_source'],sample_steps)
        record['sample']={'sliced_w2':sliced_w2(generated,fixed['sample_target'],fixed['directions']),
                          'reference_floor':sliced_w2(fixed['sample_target'],fixed['sample_target2'],fixed['directions']),
                          'heun_steps':sample_steps,'nfe':sample_steps*2,
                          'seconds':time.monotonic()-start}
    model.train()
    return record


def capture_state(model,optimizer,data_rng,step,config_hash,manifest_hash,identity):
    return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
            'identity':identity,
            'data_rng':data_rng.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
            'config_sha256':config_hash,'dataset_manifest_sha256':manifest_hash}


def save_snapshot(directory,model,optimizer,data_rng,step,config_hash,manifest_hash,identity):
    path=directory/f'step_{step:07d}.pt'
    if path.exists():
        raise FileExistsError(f'refusing to overwrite checkpoint: {path}')
    temporary=path.with_suffix('.pt.tmp')
    torch.save(capture_state(model,optimizer,data_rng,step,config_hash,manifest_hash,identity),temporary)
    os.replace(temporary,path)
    return path


def run(args):
    root=Path(__file__).resolve().parents[1]
    config_path=Path(args.config).resolve()
    cfg=json.loads(config_path.read_text())
    if args.dataset not in cfg['datasets'] or args.model not in cfg['models'] or args.seed not in cfg['seeds']:
        raise ValueError('dataset/model/seed is outside the reviewed experiment grid')
    folder=(root/cfg['data_root']/args.dataset).resolve()
    manifest_path=(root/cfg['data_root']/'manifest.json').resolve()
    manifest=json.loads(manifest_path.read_text())
    spec=manifest['datasets'][args.dataset]
    if spec['source']!='standard_normal' or spec['coupling']!='independent':
        raise ValueError('this trainer requires independent Gaussian-source data')
    dimension=spec['dim']
    device=torch.device(args.device if args.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','1')))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(args.seed)
    model=VelocityNet(dimension,args.model,**cfg['model_options']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    data_rng=torch.Generator(device=device)
    data_rng.manual_seed(args.seed+100000)
    distribution=load_distribution(spec)
    train_target=torch.from_numpy(load_targets(folder/'train.npz',dimension)).to(device)
    fixed=fixed_eval_data(folder,distribution,dimension,args.seed,device,
                          cfg['sample_count'],cfg['sliced_directions'])
    steps=args.steps if args.steps is not None else cfg['steps']
    if steps<1:
        raise ValueError('steps must be positive')
    config_hash=sha256(config_path)
    manifest_hash=sha256(manifest_path)
    identity={'dataset':args.dataset,'model':args.model,'seed':args.seed}
    provenance={'dataset':args.dataset,'model':args.model,'seed':args.seed,'device':str(device),
                'steps':steps,'model_parameters':model.parameter_count(),
                'config_path':str(config_path),'config_sha256':config_hash,
                'dataset_manifest_sha256':manifest_hash,'dataset_specification':spec,
                'git':git_info(root),'torch_version':torch.__version__,
                'numpy_version':np.__version__,
                'source_sha256':{p.name:sha256(p) for p in sorted((root/'synthetic').glob('*.py'))},
                'train_parameters':cfg}
    if args.dry_run:
        model.train()
        idx=torch.randint(len(train_target),(min(16,cfg['batch_size']),),generator=data_rng,device=device)
        target=train_target[idx]
        x0=torch.randn(target.shape,generator=data_rng,device=device)
        t=torch.rand((len(target),1),generator=data_rng,device=device)
        xt=(1-t)*x0+t*target
        loss=(model(xt,t)-(target-x0)).square().mean()
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('nonfinite dry-run loss or gradient')
        print(json.dumps({'dry_run':True,'loss':loss.item(),'parameter_count':model.parameter_count(),
                          'ode_layers':len(list(model.ode_layers())),'device':str(device)}))
        return
    run_dir=Path(args.output_root).resolve()/cfg['experiment_id']/args.dataset/args.model/f'seed{args.seed:03d}'
    checkpoints=run_dir/'checkpoints'
    if args.resume is None:
        if run_dir.exists():
            raise FileExistsError(f'refusing to overwrite existing run: {run_dir}')
        checkpoints.mkdir(parents=True)
        (run_dir/'provenance.json').write_text(json.dumps(provenance,indent=2,default=str)+'\n')
        (run_dir/'config.json').write_bytes(config_path.read_bytes())
        start_step=0
    else:
        if not run_dir.is_dir():
            raise FileNotFoundError(f'cannot resume missing run: {run_dir}')
        checkpoint=torch.load(args.resume,map_location=device,weights_only=False)
        if (checkpoint['config_sha256']!=config_hash or
                checkpoint['dataset_manifest_sha256']!=manifest_hash or
                checkpoint['identity']!=identity):
            raise ValueError('resume configuration, dataset, model, or seed differs')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        data_rng.set_state(checkpoint['data_rng'])
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        start_step=checkpoint['step']
        if start_step>=steps:
            raise ValueError('resume checkpoint already reached configured steps')
        json_line(run_dir/'resumes.jsonl',{'checkpoint':str(Path(args.resume).resolve()),
                  'from_step':start_step,'requested_total_steps':steps,'unix_time':time.time()})
    if start_step==0:
        record=evaluate(model,fixed,0,None,None,True,cfg['heun_steps'])
        record['wall_seconds']=0.0
        json_line(run_dir/'metrics.jsonl',record)
        save_snapshot(checkpoints,model,optimizer,data_rng,0,config_hash,manifest_hash,identity)
    aggregate_loss=0.0
    aggregate_grad=0.0
    aggregate_count=0
    started=time.monotonic()
    model.train()
    for step in range(start_step+1,steps+1):
        idx=torch.randint(len(train_target),(cfg['batch_size'],),generator=data_rng,device=device)
        x1=train_target[idx]
        x0=torch.randn(x1.shape,generator=data_rng,device=device)
        t=torch.rand((len(x1),1),generator=data_rng,device=device)
        xt=(1-t)*x0+t*x1
        target_velocity=x1-x0
        optimizer.zero_grad(set_to_none=True)
        loss=(model(xt,t)-target_velocity).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss at step {step}')
        loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f'nonfinite gradient at step {step}')
        optimizer.step()
        aggregate_loss+=float(loss.detach().item())
        aggregate_grad+=float(grad_norm.detach().item())
        aggregate_count+=1
        final=step==steps
        if step%cfg['eval_interval']==0 or final:
            record=evaluate(model,fixed,step,aggregate_loss/aggregate_count,
                            aggregate_grad/aggregate_count,
                            step%cfg['sample_interval']==0 or final,cfg['heun_steps'])
            record['wall_seconds']=time.monotonic()-started
            json_line(run_dir/'metrics.jsonl',record)
            print(json.dumps({'dataset':args.dataset,'model':args.model,'seed':args.seed,
                              'step':step,'train_pair_mse':record['train_pair_mse'],
                              'validation_pair_mse':record['validation_pair_mse'],
                              'validation_oracle_mse':record['validation_oracle_mse'],
                              'sliced_w2':record.get('sample',{}).get('sliced_w2')}),flush=True)
            aggregate_loss=aggregate_grad=0.0
            aggregate_count=0
        if step%cfg['checkpoint_interval']==0 or final:
            save_snapshot(checkpoints,model,optimizer,data_rng,step,config_hash,manifest_hash,identity)
    (run_dir/'COMPLETE').write_text(f'{steps}\n')


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/synthetic/train_v1.json')
    p.add_argument('--dataset',required=True)
    p.add_argument('--model',required=True,choices=['ffn','swiglu','ode_original','ode_spectral_cap'])
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',default='auto')
    p.add_argument('--output-root',default='../runs/synthetic')
    p.add_argument('--steps',type=int)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',help='explicit checkpoint path; original run output is reused')
    return p.parse_args()


if __name__=='__main__':
    run(parse_args())
