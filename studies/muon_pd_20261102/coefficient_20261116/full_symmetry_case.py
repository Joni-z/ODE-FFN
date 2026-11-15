"""Paired C8-stratified sampling with an equivariant sampling shortcut."""
import argparse,json,time
from pathlib import Path
import numpy as np
from gpu_resolution import base,torch,load,arrays
from evaluate_screen import ev

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);p.add_argument('--output',required=True);p.add_argument('--base-samples',type=int,default=4096)
    a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(1)
    case=next(c for c in json.loads((root/'portable.json').read_text())['cases'] if c['id']==a.case);assert case['dataset']=='gmm_ring8_2d'
    model=load(root,case,'cpu');data=arrays(root,case);n=a.base_samples;assert n<=4096
    model.double();projected=ev.C8(model);probe=torch.from_numpy(data['source_0'][:8]).double()
    with torch.no_grad():
        first=base.train.generated_samples(projected,probe,128)
        second=base.train.generated_samples(projected,ev.rotation(probe),128)
        symmetry_error=float((second-ev.rotation(first)).abs().max());assert symmetry_error<1e-9
    model.float();projected=ev.C8(model).float();rot=projected.rotations
    probe=probe.float()
    with torch.no_grad():
        first32=base.train.generated_samples(projected,probe,128)
        second32=base.train.generated_samples(projected,ev.rotation(probe),128)
        symmetry_error32=float((second32-ev.rotation(first32)).abs().max());assert symmetry_error32<1e-4
    source=torch.from_numpy(data['source_0'][:n]);target=torch.from_numpy(data['target_0'][:n]);directions=torch.from_numpy(data['directions']).double()
    rotate=lambda v:torch.einsum('nd,ked->kne',v,rot).reshape(-1,2)
    all_source=rotate(source);all_target=rotate(target);rows=[]
    for mode in ['original','projected']:
        started=time.monotonic()
        generated=(base.train.generated_samples(model,all_source,128) if mode=='original' else rotate(base.train.generated_samples(projected,source,128)))
        assert torch.isfinite(generated).all();value=generated.numpy();name=mode+'.npy';np.save(out/name,value,allow_pickle=False)
        means=np.array([[3*np.cos(i*np.pi/4),3*np.sin(i*np.pi/4)] for i in range(8)])
        labels=np.argmax(value@means.T,axis=1);counts=np.bincount(labels,minlength=8)
        if mode=='projected':assert np.max(counts)-np.min(counts)<=2
        rows.append(dict(mode=mode,samples=8*n,independent_base_samples=n,
                    stratified_sw2=base.train.sliced_w2(generated.double(),all_target.double(),directions),
                    counts=counts.tolist(),sector_l1=float(np.abs(counts/(8*n)-.125).sum()),seconds=time.monotonic()-started,
                    generated_file=name,generated_sha256=base.digest(out/name)))
        base.dump(out/'progress.json',dict(completed=len(rows),rows=rows))
    base.dump(out/'COMPLETE.json',dict(passed=True,id=case['id'],family=case['family'],dataset=case['dataset'],
            scope='paired C8-stratified evaluation, not iid pooled-bank scores',rows=rows,
            symmetry_shortcut_max_error=symmetry_error,symmetry_shortcut_float32_max_error=symmetry_error32,
            checkpoint_sha256=case['checkpoint_sha256'],assets_sha256=case['assets_sha256'],source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,id=case['id'],base_samples=n)))

if __name__=='__main__':main()
