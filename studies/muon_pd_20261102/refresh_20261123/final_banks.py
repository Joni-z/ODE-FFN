"""Confirm one locked, two-task finalist on a reserved paired sampling stream."""
import argparse
import fcntl
import json
from pathlib import Path
import numpy as np
from odd_trial import HERE, base, build_model, torch
from evaluate_screen import assets, ev

STREAM = 2026112300
DATASETS = ('gmm_ring8_2d','rotated_product_bimodal_16d')


def read(path):
    return json.loads(path.read_text())


def entries(prep, candidate):
    historical = prep/'coefficient-20261116'
    ordinary = read(HERE.parents[2]/'docs/report4/midpoint-refresh/ordinary-selection.json')
    chosen = read(candidate/'selection.json')
    assert len(chosen)==2 and {r['dataset'] for r in chosen}==set(DATASETS)
    models = []
    families = set()
    for result in chosen:
        ds = result['dataset']
        rows = [r for r in ordinary if r['dataset']==ds]
        assert len(rows)==2
        threshold = min(min(r['matched_sw2'],r['original_20k_sw2']) for r in rows)
        if result['pooled_sw2'] >= threshold:
            raise ValueError(f"{ds}: selection SW2 {result['pooled_sw2']} has not beaten {threshold}")
        eval_dir = candidate/'selection'/ds
        done, binding = read(eval_dir/'COMPLETE.json'), read(eval_dir/'binding.json')
        assert done['passed'] and not done['witness'] and done['steps']==[result['step']]
        assert binding['split']=='selection' and binding['source_config']['seeds']==[0]
        assert (binding['banks'],binding['samples_per_bank'],binding['heun_steps'])==(8,4096,128)
        assert all(binding['assets']['sha256']==r['assets_sha256'] for r in rows)
        assert read(eval_dir/'summary.json')==[{k:v for k,v in result.items() if k!='dataset'}]
        def register(label, folder, step, sha):
            cfg=read(folder/'config.json')
            assert cfg['seeds']==[0] and cfg['cell']['dataset']==ds
            assert sha==base.digest(folder/'checkpoints'/f'step_{step:07d}.pt')
            models.append(dict(dataset=ds, label=label, family=cfg['cell']['family'],
                               folder=str(folder), step=step, checkpoint_sha256=sha))
        register('candidate',Path(binding['source_folder']),result['step'],result['checkpoint_sha256'])
        families.add(models[-1]['family'])
        for r in rows:
            orig,=(historical/'analysis/selection').glob(f"{ds}__{r['family']}__*/binding.json")
            cell=orig.parent.name
            matched=read(prep/'report4-equal-compute-20261122/trained-endpoints'/f'{cell}.json')
            register(cell+'-20k',Path(read(orig)['source_folder']),20000,r['original_checkpoint_sha256'])
            register(cell+'-matched',Path(matched['folder']),r['matched_steps'],r['matched_checkpoint_sha256'])
        # These ordinary fields share the candidate's sign symmetry. Keep them
        # in the same paired evaluation for attribution on these two laws.
        extra = [prep/'odd-control-20261123']
        if ds == DATASETS[0] and binding['source_config']['cell']['family'] in ('odd_glu_midpoint192','odd_feedback192'):
            extra.append(prep/'glu-euler-ring-20261123')
        for work in extra:
            path=work/'selection'/ds
            done=read(path/'COMPLETE.json')
            detail=read(path/'binding.json')
            control,=read(path/'summary.json')
            assert done['passed'] and not done['witness'] and done['steps']==[control['step']]
            assert detail['assets']['sha256']==binding['assets']['sha256']
            family=detail['source_config']['cell']['family']
            if result['pooled_sw2'] >= control['pooled_sw2']:
                raise ValueError(f"{ds}: {family} already scores {control['pooled_sw2']} on selection")
            register(family+'-symmetry',Path(detail['source_folder']),control['step'],control['checkpoint_sha256'])
    assert len(families)==1 and all(len([m for m in models if m['dataset']==ds])>=6 for ds in DATASETS)
    return models


@torch.no_grad()
def run(prep, models, out):
    torch.set_num_threads(1)
    ev.build_model=build_model
    lock=out/'endpoints.json'
    if lock.exists():
        if read(lock)!=models:
            raise ValueError('The reserved stream is already tied to different weights.')
    else:
        base.dump(lock,models)
    rows=[]
    for ds in DATASETS:
        group=[m for m in models if m['dataset']==ds]
        cache=Path(read(Path(group[0]['folder'])/'config.json')['data_root'])
        manifest=cache/'manifest.json'
        law=base.train.load_distribution(read(manifest)['datasets'][ds])
        _,fixed,_=assets(prep/'coefficient-20261116/assets',ds,'selection','cpu')
        dirs=fixed['directions'].double()
        arrays=ev.bank_arrays(law,dirs.numpy(),4096,8,STREAM)
        hashes={k:ev.array_hash(v) for k,v in arrays.items()}
        base.dump(out/f'{ds}-banks.json',dict(stream=STREAM,array_hashes=hashes))
        targets=[torch.from_numpy(arrays[f'target_{i}']).double() for i in range(8)]
        for m in group:
            dest=out/ds/m['label']
            dest.mkdir(parents=True,exist_ok=True)
            prior=dest/'score.json'
            if prior.exists():
                score=read(prior)
                assert score['model']==m and score['bank_hashes']==hashes
            else:
                folder=Path(m['folder'])
                model,binding=ev.restore(folder,read(folder/'config.json'),manifest,m['step'],'cpu')
                assert binding['checkpoint_sha256']==m['checkpoint_sha256']
                generated=[]; bank_scores=[]
                for i in range(8):
                    source=torch.from_numpy(arrays[f'source_{i}'])
                    y=base.train.generated_samples(model,source,128).double()
                    if not torch.isfinite(y).all():
                        raise ValueError(f"nonfinite sample: {ds}/{m['label']}/{i}")
                    generated.append(y)
                    bank_scores.append(base.train.sliced_w2(y,targets[i],dirs))
                    np.save(dest/f'bank-{i}.npy',y.float().numpy(),allow_pickle=False)
                pool=base.train.sliced_w2(torch.cat(generated),torch.cat(targets),dirs)
                # Independent projection and sorting by slices to bound temporary memory.
                gen=np.concatenate([np.load(dest/f'bank-{i}.npy') for i in range(8)]).astype(np.float64)
                target=torch.cat(targets).numpy()
                err=0.
                for direction in np.array_split(dirs.numpy(),16):
                    delta=np.sort(gen@direction.T,axis=0)-np.sort(target@direction.T,axis=0)
                    err+=np.square(delta).sum()
                np.testing.assert_allclose(np.sqrt(err/(len(gen)*len(dirs))),pool,rtol=1e-10)
                score=dict(model=m,pooled_sw2=pool,bank_sw2=bank_scores,bank_hashes=hashes)
                base.dump(prior,score)
            rows.append(score)
            base.dump(out/'progress.json',dict(scored=len(rows),total=len(models)))
    wins={}
    for ds in DATASETS:
        group=[r for r in rows if r['model']['dataset']==ds]
        candidate,=[r for r in group if r['model']['label']=='candidate']
        wins[ds]=all(candidate['pooled_sw2']<r['pooled_sw2'] for r in group if r is not candidate)
    base.dump(out/'results.json',dict(rows=rows,beats_all_ordinary=wins))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('prep',type=Path)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--check-eligibility',action='store_true')
    args=parser.parse_args()
    models=entries(args.prep,args.candidate)
    if args.check_eligibility:
        print(json.dumps(models,indent=2))
    else:
        out=args.prep/'refresh-locked-2026112300'
        out.mkdir(exist_ok=True)
        with (out/'score.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(args.prep,models,out)
