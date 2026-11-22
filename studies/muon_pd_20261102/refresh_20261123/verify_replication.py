"""Verify training identities, budgets, and every locked score from saved samples."""
import os
from pathlib import Path
import sys
import numpy as np
from final_banks import DATASETS, HERE, assets, base, ev, read, torch


def run(prep,out):
    os.nice(7)
    torch.set_num_threads(1)
    doc=HERE.parents[2]/'docs/report4/midpoint-refresh'
    lock=read(doc/'bounded-replication-lock.json')
    root=prep/'bounded-locked-2026112350'
    endpoints=read(root/'endpoints.json')
    assert lock['weights']==[{k:v for k,v in m.items() if k!='folder'} for m in endpoints]
    verified=[]
    for ds in DATASETS:
        group=[m for m in endpoints if m['dataset']==ds]
        assert len(group)==6
        candidate,=[m for m in group if m['label']=='candidate']
        cf=Path(candidate['folder'])
        cfg=read(cf/'config.json')
        manifest=Path(cfg['data_root'])/'manifest.json'
        law=base.train.load_distribution(read(manifest)['datasets'][ds])
        claimed=read(root/f'{ds}-banks.json')
        assert claimed['stream']==lock['stream']
        directions,_,_=assets(prep/'coefficient-20261116/assets',ds,'selection','cpu')
        dirs=directions['directions'].astype(float)
        assert dirs.shape==(256,law.dim)
        targets=[]
        for i in range(8):
            rng=np.random.default_rng(lock['stream']+i)
            x=rng.standard_normal((4096,law.dim)).astype(np.float32)
            y=law.sample(rng,4096)
            assert ev.array_hash(x)==claimed['array_hashes'][f'source_{i}']
            assert ev.array_hash(y)==claimed['array_hashes'][f'target_{i}']
            targets.append(y)
        target=np.concatenate(targets).astype(float)
        budgets=read(prep/'bounded-odd-20261123/budget.json')
        cb,=[b for b in budgets if b['dataset']==ds]
        original,=[b for b in read(doc.parent/'equal-compute/budget.json')['rows']
                   if b['dataset']==ds and b['family']=='state_midpoint4']
        prov=read(cf/'provenance.json')
        assert prov['model']==cfg['cell']['id'] and prov['seed']==0 and prov['device']=='cpu'
        assert cfg['cell']['host']=='macmini' and cfg['cell']['family']=='bounded_odd_state_midpoint4'
        assert cb['steps']==candidate['step']==prov['steps']
        assert abs(cb['steps']*cb['step_flops']-original['target_flops'])<=cb['step_flops']/2
        rows=[]
        for m in group:
            folder=Path(m['folder'])
            cfg=read(folder/'config.json')
            file=folder/'checkpoints'/f"step_{m['step']:07d}.pt"
            assert base.digest(file)==m['checkpoint_sha256']
            cp=torch.load(file,map_location='cpu',weights_only=False)
            assert cfg['seeds']==[0] and cp['identity']==dict(seed=0,model=cfg['cell']['id'],dataset=ds)
            assert cp['step']==m['step'] and cp['config_sha256']==base.digest(folder/'config.json')
            assert cp['dataset_manifest_sha256']==base.digest(manifest)
            data=root/ds/m['label']
            samples=np.concatenate([np.load(data/f'bank-{i}.npy',allow_pickle=False) for i in range(8)]).astype(float)
            assert samples.shape==target.shape and np.isfinite(samples).all()
            error=0.
            for block in np.array_split(dirs,16):
                d=np.sort(samples@block.T,axis=0)-np.sort(target@block.T,axis=0)
                error+=np.square(d).sum()
            sw2=float(np.sqrt(error/(len(samples)*len(dirs))))
            logged=read(data/'score.json')['pooled_sw2']
            assert abs(sw2-logged)<1.e-10
            rows.append(dict(label=m['label'],checkpoint_sha256=m['checkpoint_sha256'],
                             sw2=sw2,replay_absolute_error=abs(sw2-logged)))
        candidate_score,=[r['sw2'] for r in rows if r['label']=='candidate']
        archive_scores=[r['sw2'] for r in rows if r['label']!='candidate' and not r['label'].endswith('-symmetry')]
        assert len(archive_scores)==4 and candidate_score<min(archive_scores)
        verified.append(dict(dataset=ds,seed=0,device=prov['device'],host=read(cf/'config.json')['cell']['host'],
                             parameters=prov['model_parameters'],training_steps=cb['steps'],
                             budget_ratio=cb['step_flops']*cb['steps']/original['target_flops'],scores=rows))
    base.dump(out,dict(stream=lock['stream'],samples_per_dataset_model=32768,projections=256,rows=verified))


if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]))
