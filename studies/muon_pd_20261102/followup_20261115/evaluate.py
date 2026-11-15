"""Common-bank native sampling and C8 field projection on frozen checkpoints."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from train import HERE, base, build_model, torch
sys.path.insert(0,str(HERE.parent/'autoresearch_20261103/cycle14'))
from audit import bank_arrays, array_hash, tensor_snapshot, same_tensors, field
sys.path.insert(0,str(HERE.parent/'autoresearch_20261103/cycle15'))
from transport import sample, Oracle


class C8(torch.nn.Module):
    """Reynolds projection: mean_k R_k^-1 v(R_k x,t)."""
    def __init__(self, model):
        super().__init__();self.model=model
        theta=torch.arange(8,dtype=torch.float64)*(np.pi/4)
        c,s=theta.cos(),theta.sin()
        self.register_buffer('rotations',torch.stack((c,-s,s,c),dim=1).reshape(8,2,2))

    def forward(self,x,t):
        rot=self.rotations.to(x)
        transformed=torch.einsum('nd,ked->kne',x,rot)
        values=self.model(transformed.reshape(-1,2),t.expand(8,*t.shape).reshape(-1,1)).reshape(8,len(x),2)
        return torch.einsum('kne,ked->knd',values,rot).mean(0)


def rotation(x):
    c=2**-.5
    return torch.stack((c*(x[:,0]-x[:,1]),c*(x[:,0]+x[:,1])),dim=1)


@torch.no_grad()
def equivariance(model,x,t):
    v=model(x,t);error=model(rotation(x),t)-rotation(v)
    return dict(absolute_rms=float(error.double().square().mean().sqrt()),
                relative_rms=float(error.double().square().mean().sqrt()/v.double().square().mean().sqrt().clamp_min(1e-12)))


def self_test(device):
    from synthetic.data import suite
    law=suite()['gmm_ring8_2d'];torch.manual_seed(0)
    x=torch.randn(128,2,device=device,dtype=torch.double)
    t=torch.linspace(0,1,128,device=device,dtype=torch.double)[:,None]
    oracle=Oracle(law);err=equivariance(oracle,x,t)
    assert err['absolute_rms']<1e-10
    # Anisotropic linear field: group mean is trace(A)/2 times identity.
    class Linear(torch.nn.Module):
        def forward(self,x,t):return x*x.new_tensor([2.,-1.])
    projected=C8(Linear()).to(device)
    assert torch.allclose(projected(x,t),.5*x,atol=1e-12,rtol=1e-12)
    assert torch.allclose(C8(projected).to(device)(x,t),projected(x,t),atol=1e-12,rtol=1e-12)
    assert equivariance(projected,x,t)['absolute_rms']<1e-12
    y,path=sample(projected,oracle,x,128,record=True)
    assert torch.equal(y,base.train.generated_samples(projected,x,128))
    yr=base.train.generated_samples(projected,rotation(x),128)
    assert torch.allclose(yr,rotation(y),rtol=1e-11,atol=1e-11)
    assert path.shape==(129,128,2)
    return dict(passed=True,oracle=err,linear_group_projection=True,idempotence=True,
                native_recording_exact=True,rotated_solver_pair=True)


def restore(folder,cfg,manifest,step,device):
    file=folder/'checkpoints'/f'step_{step:07d}.pt'
    state=torch.load(file,map_location=device,weights_only=False)
    assert state['step']==step and state['config_sha256']==base.digest(folder/'config.json')
    assert state['dataset_manifest_sha256']==base.digest(manifest)
    spec=json.loads(manifest.read_text())['datasets'][cfg['cell']['dataset']]
    torch.manual_seed(0);model=build_model(spec['dim'],cfg['cell']['family']).to(device).eval()
    base.train.restore_model(model,state)
    return model,dict(checkpoint_sha256=base.digest(file),config_sha256=state['config_sha256'],manifest_sha256=state['dataset_manifest_sha256'])


def mode_stats(values,law):
    labels=np.argmax(values@law.means.T,axis=1)
    counts=np.bincount(labels,minlength=8)
    return dict(counts=counts.tolist(),sector_l1=float(np.abs(counts/len(values)-law.weights).sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--steps',nargs='+',type=int,default=[15000,20000]);p.add_argument('--c8',action='store_true');p.add_argument('--witness',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);folder=Path(a.folder)
    torch.set_num_threads(1);base.verify_sources();base.dump(out/'self-test.json',self_test(a.device))
    assert (folder/'ADAPTER_COMPLETE').exists()
    cfg=json.loads((folder/'config.json').read_text());cell=cfg['cell']
    assert cfg['steps']==20000 and cfg['seeds']==[0]
    cache=base.verify_data(a.data_root,cell);manifest=cache/'manifest.json'
    spec=json.loads(manifest.read_text())['datasets'][cell['dataset']];law=base.train.load_distribution(spec)
    assert not a.c8 or law.dim==2 and cell['dataset']=='gmm_ring8_2d'
    fixed=base.train.fixed_eval_data(cache/cell['dataset'],law,law.dim,0,torch.device(a.device),4096,256,'test')
    arrays=bank_arrays(law,fixed['directions'].cpu().numpy(),4096,8,2026110400)
    np.savez_compressed(out/'banks.npz',**arrays)
    base.dump(out/'binding.json',dict(source_config=cfg,bank_hashes={k:array_hash(v) for k,v in arrays.items()},source_sha256=base.digest(__file__),heun_steps=128,banks=8,samples_per_bank=4096,c8=a.c8,witness=a.witness,source_folder=str(folder)))
    rows=[];summaries=[];targets=np.concatenate([arrays[f'target_{b}'] for b in range(8)])
    for step in a.steps:
        model,identity=restore(folder,cfg,manifest,step,a.device);snapshot=tensor_snapshot(model)
        prediction,mse=field(model,fixed)
        if step==20000:
            gen=base.train.generated_samples(model,fixed['sample_source'],32).cpu().numpy()
            with np.load(folder/'final_test_examples.npz') as z:
                saved=z['generated'];target_same=np.array_equal(z['target'],fixed['sample_target'].cpu().numpy())
            exact=np.array_equal(gen,saved);passed=bool(exact and target_same)
            base.dump(out/'native-anchor.json',dict(passed=passed,exact=exact,max_abs=float(np.max(np.abs(gen-saved))),oracle_mse=mse,**identity))
            assert passed,'Native restore does not reproduce its source-host final sample'
        for variant,net in [('native',model)]+([('c8',C8(model).to(a.device).eval())] if a.c8 else []):
            _,fmse=field(net,fixed)
            eq=equivariance(net,fixed['x'][:1024],fixed['t'][:1024]) if a.c8 else None
            pooled=[];before=time.monotonic()
            for bank in range(1 if a.witness else 8):
                source=torch.from_numpy(arrays[f'source_{bank}'][:64] if a.witness else arrays[f'source_{bank}']).to(a.device)
                generated=base.train.generated_samples(net,source,128)
                assert bool(torch.isfinite(generated).all())
                if a.witness:continue
                value=generated.cpu().numpy();pooled.append(value)
                np.save(out/f'{step}-{variant}-b{bank}.npy',value,allow_pickle=False)
                target=torch.from_numpy(arrays[f'target_{bank}']).to(a.device)
                sw2=base.train.sliced_w2(generated.double(),target.double(),fixed['directions'].double())
                row=dict(step=step,variant=variant,bank=bank,sw2=sw2,field_mse=fmse,**identity)
                if a.c8:row.update(mode_stats(value,law))
                rows.append(row)
                with (out/'banks.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                base.dump(out/'progress.json',dict(step=step,variant=variant,banks_completed=bank+1,seconds=time.monotonic()-before))
            if a.witness:continue
            values=np.concatenate(pooled)
            pool=base.train.sliced_w2(torch.from_numpy(values).to(a.device).double(),torch.from_numpy(targets).to(a.device).double(),fixed['directions'].double())
            vals=[r['sw2'] for r in rows if r['step']==step and r['variant']==variant]
            summary=dict(step=step,variant=variant,field_mse=fmse,sw2_mean=float(np.mean(vals)),sw2_sd=float(np.std(vals,ddof=1)),pooled_sw2=pool,equivariance=eq,seconds=time.monotonic()-before,**identity)
            if a.c8:
                summary.update(mode_stats(values,law));start=torch.from_numpy(arrays['source_0'][:256]).to(a.device)
                end,path=sample(net,Oracle(law),start,128,record=True)
                end_r,path_r=sample(net,Oracle(law),rotation(start),128,record=True)
                rotated_path=np.stack([rotation(torch.from_numpy(x)).numpy() for x in path])
                pair_error=np.sqrt(np.mean((path_r.astype(float)-rotated_path.astype(float))**2,axis=(1,2)))
                summary['rotated_trajectory_final_rms']=float(pair_error[-1])
                np.savez_compressed(out/f'{step}-{variant}-paths.npz',path=path,rotated_start_path=path_r,pair_rms=pair_error)
                profiles=[]
                for i in range(0,129,4):
                    x=torch.from_numpy(path[i]).to(a.device);t=x.new_full((len(x),1),i/128)
                    profiles.append(dict(time=i/128,**equivariance(net,x,t)))
                base.dump(out/f'{step}-{variant}-equivariance.json',profiles)
            summaries.append(summary);base.dump(out/'summary.json',summaries)
        same_tensors(model,snapshot)
    if a.c8 and not a.witness:
        paired=[]
        for step in a.steps:
            native=np.array([r['sw2'] for r in rows if r['step']==step and r['variant']=='native'])
            sym=np.array([r['sw2'] for r in rows if r['step']==step and r['variant']=='c8'])
            delta=sym-native
            paired.append(dict(step=step,c8_minus_native_sw2=delta.tolist(),mean=float(delta.mean()),sd=float(delta.std(ddof=1)),scope='sampling-bank variation conditional on these weights'))
        base.dump(out/'paired.json',paired)
    base.dump(out/'COMPLETE.json',dict(passed=True,witness=a.witness,steps=a.steps,source_config_sha256=base.digest(folder/'config.json')))
    print(json.dumps(dict(passed=True,output=str(out),witness=a.witness)))

if __name__=='__main__':main()
