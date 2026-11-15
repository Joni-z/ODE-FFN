"""Finite-orbit C8 error decomposition and frozen-field projection intervention."""
import argparse,json
from pathlib import Path
import numpy as np
from gpu_resolution import base,torch,load,arrays
from evaluate_screen import ev

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root);out=Path(a.output);out.mkdir(exist_ok=False);torch.set_num_threads(1);rows=[]
    for case in json.loads((root/'portable.json').read_text())['cases']:
        if case['dataset']!='gmm_ring8_2d':continue
        model=load(root,case,'cpu');snapshot={k:v.clone() for k,v in model.state_dict().items()}
        data=arrays(root,case);law=base.train.load_distribution(json.loads((root/case['dataset_manifest']).read_text())['datasets'][case['dataset']])
        model=model.double();projected=ev.C8(model)
        x=torch.from_numpy(data['field_x'][:256]).double();t=torch.from_numpy(data['field_t'][:256]).double();rot=projected.rotations
        transformed=torch.einsum('nd,ked->kne',x,rot);times=t.expand(8,*t.shape).reshape(-1,1)
        with torch.no_grad():
            v=model(transformed.reshape(-1,2),times).reshape(8,len(x),2)
            pulled=torch.einsum('kne,ked->knd',v,rot);mean=pulled.mean(0)
            truth=torch.from_numpy(law.velocity(x.numpy(),t.numpy()))
            group_truth=torch.from_numpy(law.velocity(transformed.reshape(-1,2).numpy(),times.numpy())).reshape(8,len(x),2)
            truth_back=torch.einsum('kne,ked->knd',group_truth,rot)
            assert float((truth_back-truth).abs().max())<1e-8
            total=float((pulled-truth).square().mean());invariant=float((mean-truth).square().mean());broken=float((pulled-mean).square().mean())
            assert abs(total-invariant-broken)<1e-10
        model=model.float();projected=ev.C8(model).float()
        source=torch.from_numpy(data['source_0'][:512]);target=torch.from_numpy(data['target_0'][:512]);directions=torch.from_numpy(data['directions']).double()
        original=base.train.generated_samples(model,source,128);symmetrized=base.train.generated_samples(projected,source,128)
        values={}
        for name,value in [('original',original),('projected',symmetrized)]:
            filename=case['id']+'-'+name+'.npy';np.save(out/filename,value.numpy(),allow_pickle=False)
            values[name]=dict(sw2_512=base.train.sliced_w2(value.double(),target.double(),directions),**ev.mode_stats(value.numpy(),law),file=filename,sha256=base.digest(out/filename))
        assert all(torch.equal(snapshot[k],v) for k,v in model.state_dict().items())
        rows.append(dict(id=case['id'],family=case['family'],field_base_states=256,orbit_states=2048,
                     total_orbit_mse=total,invariant_mse=invariant,symmetry_breaking_mse=broken,
                     decomposition_residual=abs(total-invariant-broken),sampling=values,
                     checkpoint_sha256=case['checkpoint_sha256'],assets_sha256=case['assets_sha256']))
        base.dump(out/'progress.json',dict(completed=len(rows),rows=rows))
    base.dump(out/'COMPLETE.json',dict(passed=True,scope='bounded symmetry intervention,512samples; not pooled full-bank ranking',rows=rows,source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,cases=len(rows),output=str(out))))

if __name__=='__main__':main()
