"""Outer Heun refinement for the completed shared/released pair; no training."""
import argparse,json,time
from pathlib import Path
import numpy as np
from sharing_trial import base,torch
import sharing_evaluate as evaluation
ev=evaluation.evaluator.ev

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--samples-per-bank',type=int,default=512);p.add_argument('--banks',type=int,default=1);a=p.parse_args()
    assert 1<=a.samples_per_bank<=4096 and 1<=a.banks<=8
    root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);rows=[]
    for case in json.loads((root/'sharing-anchors.json').read_text())['cases']:
        folder=Path(case['folder']);cfg=json.loads((folder/'config.json').read_text());cell=cfg['cell']
        if cell['family'] not in ['state_midpoint4','stage_operator_midpoint4']:continue
        assert (folder/'ADAPTER_COMPLETE').exists()
        cache=base.verify_data(a.data_root,cell);model,identity=ev.restore(folder,cfg,cache/'manifest.json',20000,'cpu')
        arrays,fixed,binding=evaluation.evaluator.assets(root/'sharing-assets',cell['dataset'],'confirmation','cpu')
        n=a.samples_per_bank;source=torch.from_numpy(np.concatenate([arrays[f'source_{i}'][:n] for i in range(a.banks)]))
        target=torch.from_numpy(np.concatenate([arrays[f'target_{i}'][:n] for i in range(a.banks)]));values={};scores=[];snapshot=ev.tensor_snapshot(model)
        for steps in [128,256,512]:
            start=time.monotonic();value=base.train.generated_samples(model,source,steps);assert torch.isfinite(value).all()
            score=base.train.sliced_w2(value.double(),target.double(),fixed['directions'].double());f=case['id']+f'-heun{steps}.npy'
            np.save(out/f,value.numpy(),allow_pickle=False);values[steps]=value.numpy()
            scores.append(dict(heun_steps=steps,nfe=2*steps,sw2=score,generated_file=f,generated_sha256=base.digest(out/f),seconds=time.monotonic()-start))
            base.dump(out/'progress.json',dict(case=case['id'],steps=steps,cases_completed=len(rows)))
        for row in scores:
            delta=values[row['heun_steps']].astype(float)-values[512].astype(float)
            row['rms_euclidean_shift_to512']=float(np.sqrt(np.mean(np.sum(delta**2,axis=1))))
        ev.same_tensors(model,snapshot)
        rows.append(dict(id=case['id'],dataset=cell['dataset'],family=cell['family'],checkpoint=identity,assets_sha256=binding['sha256'],scores=scores))
    assert len(rows)==4
    base.dump(out/'COMPLETE.json',dict(passed=True,samples_per_bank=n,banks=a.banks,rows=rows,
        scope='bounded common-source numerical check' if n*a.banks<32768 else 'full common-bank numerical check',source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,cases=len(rows),samples=n*a.banks,output=str(out))))

if __name__=='__main__':main()
