"""Independent NumPy audit of frozen-weight resolution and C8 interventions."""
import argparse,csv,hashlib,json,subprocess
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def sw2(x,y,d):
    a=np.sort(x.astype(float)@d.astype(float).T,axis=0)
    b=np.sort(y.astype(float)@d.astype(float).T,axis=0)
    return float(np.sqrt(np.mean((a-b)**2)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--portable',required=True);p.add_argument('--resolution',required=True)
    p.add_argument('--symmetry');p.add_argument('--source-commit');p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.portable);res=Path(a.resolution);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    portable=read(root/'portable.json');cases=portable['cases'];assert cases
    terminal=read(res/'queue-finished.json')
    bykey={(r['id'],r['mode']):r for r in terminal['completed']}
    assert len(bykey)==len(terminal['completed'])
    rows=[];banks=[];bindings=[];maximum=0.
    state_families=('state_midpoint4','free_stage_midpoint4','stage_operator_midpoint4','contrast_operator_midpoint4')
    for case in cases:
        for key in ['checkpoint','config','dataset_manifest','assets']:
            assert sha(root/case[key])==case[key+'_sha256'],(case['id'],key)
        if a.source_commit:
            for filename,digest in case.get('builder_sources_sha256',{}).items():
                code=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[3]),'show',a.source_commit+':'+filename])
                assert hashlib.sha256(code).hexdigest()==digest,filename
        with np.load(root/case['assets']) as z:data={k:z[k].copy() for k in z.files}
        outer=[mode for mode in ['outer-subset','full-outer'] if (case['id'],mode) in bykey]
        assert len(outer)==1
        modes=outer+(['full-internal'] if case['family'] in state_families else [])
        for mode in modes:
            receipt=bykey[(case['id'],mode)];assert receipt['exit_code']==0 and receipt['complete']
            folder=res/case['id']/mode;file=folder/'COMPLETE.json';done=read(file)
            assert done['passed'] and done['case']['checkpoint_sha256']==case['checkpoint_sha256']
            if mode=='full-internal':
                assert [(r['inner_substeps'],r['heun_steps']) for r in done['rows']] in [[(4,128),(16,128)],[(4,128),(8,128),(16,128)]]
            else:
                inner=4 if case['family'] in state_families else None
                assert [(r['inner_substeps'],r['heun_steps']) for r in done['rows']]==[(inner,s) for s in ([64,128,256,512] if mode=='outer-subset' else [128,256,512])]
            if a.source_commit:
                code=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[3]),'show',a.source_commit+':studies/muon_pd_20261102/coefficient_20261116/cpu_resolution.py'])
                assert hashlib.sha256(code).hexdigest()==done['source_sha256']
            target=data['target_0'][:512] if mode=='outer-subset' else np.concatenate([data[f'target_{i}'] for i in range(8)])
            values={}
            for row in done['rows']:
                f=folder/row['generated_file'];assert sha(f)==row['generated_sha256']
                x=np.load(f,allow_pickle=False);assert x.shape==target.shape and np.isfinite(x).all()
                score=sw2(x,target,data['directions']);error=abs(score-row['pooled_sw2'])
                if mode!='outer-subset':
                    scores=[sw2(x[i*4096:(i+1)*4096],target[i*4096:(i+1)*4096],data['directions']) for i in range(8)]
                    error=max(error,float(np.max(np.abs(np.array(scores)-row['bank_sw2']))))
                    banks.extend(dict(id=case['id'],mode=mode,inner=row['inner_substeps'],outer=row['heun_steps'],bank=i,score=score) for i,score in enumerate(scores))
                maximum=max(maximum,error);assert error<1e-10,(case['id'],mode,error)
                values[row['generated_file']]=x
                rows.append(dict(id=case['id'],dataset=case['dataset'],family=case['family'],mode=mode,
                            inner=row['inner_substeps'],outer=row['heun_steps'],score=score,samples=len(x),
                            checkpoint_sha256=case['checkpoint_sha256'],sample_sha256=sha(f)))
            reference=values['inner16-heun128.npy' if mode=='full-internal' else 'inner4-heun512.npy']
            for row in done['rows']:
                delta=values[row['generated_file']].astype(float)-reference.astype(float)
                rms=float(np.sqrt(np.mean(np.sum(delta**2,axis=1))))
                assert np.isclose(rms,row['rms_euclidean_shift_to_finest'],atol=1e-6,rtol=1e-6)
            bindings.append(dict(id=case['id'],mode=mode,completion_sha256=sha(file)))
    assert len(bindings)==sum(1+int(c['family'] in state_families) for c in cases)
    if a.symmetry:
        sym=Path(a.symmetry);end=read(sym/'queue-finished.json');assert end['passed'] and len(end['completed'])==sum(c['dataset']=='gmm_ring8_2d' for c in cases) and not end['unstarted']
        for receipt in end['completed']:
            assert receipt['exit_code']==0 and receipt['complete']
            case=next(c for c in cases if c['id']==receipt['id']);folder=sym/case['id']/'symmetry-full';file=folder/'COMPLETE.json';done=read(file)
            assert done['passed'] and done['dtype']=='float64' and done['base_samples']==4096
            assert done['checkpoint_sha256']==case['checkpoint_sha256'] and done['assets_sha256']==case['assets_sha256']
            assert sha(folder/'stratified-source.npy')==done['stratified_source_sha256']
            assert sha(folder/'stratified-target.npy')==done['stratified_target_sha256']
            with np.load(root/case['assets']) as z:data={k:z[k].copy() for k in z.files}
            theta=np.arange(8)*np.pi/4;c=np.cos(theta);s=np.sin(theta);rot=np.stack((c,-s,s,c),axis=1).reshape(8,2,2)
            for key in ['source','target']:
                expected=np.einsum('nd,ked->kne',data[key+'_0'].astype(float),rot).reshape(-1,2)
                assert np.allclose(expected,np.load(folder/('stratified-'+key+'.npy')),atol=1e-12,rtol=1e-12)
            target=np.load(folder/'stratified-target.npy');assert len(done['rows'])==2
            means=np.array([[3*np.cos(i*np.pi/4),3*np.sin(i*np.pi/4)] for i in range(8)])
            for row in done['rows']:
                f=folder/row['generated_file'];assert sha(f)==row['generated_sha256']
                x=np.load(f);assert x.shape==(32768,2) and np.isfinite(x).all()
                score=sw2(x,target,data['directions']);error=abs(score-row['stratified_sw2']);maximum=max(maximum,error);assert error<1e-10
                counts=np.bincount(np.argmax(x@means.T,axis=1),minlength=8);assert counts.tolist()==row['counts']
                assert abs(float(np.abs(counts/32768-.125).sum())-row['sector_l1'])<1e-12
                rows.append(dict(id=case['id'],dataset=case['dataset'],family=case['family'],mode='C8-stratified-'+row['mode'],inner=None,outer=128,score=score,samples=32768,checkpoint_sha256=case['checkpoint_sha256'],sample_sha256=sha(f)))
            bindings.append(dict(id=case['id'],mode='symmetry-full',completion_sha256=sha(file)))
    with (out/'scores.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if banks:
        with (out/'banks.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(banks[0]));w.writeheader();w.writerows(banks)
    result=dict(passed=True,scope='resolution plus complete float64 C8' if a.symmetry else 'complete registered numerical tasks; symmetry excluded',
                cases=len(bindings),rows=len(rows),max_numpy_difference=maximum,bindings=bindings,
                portable_manifest_sha256=sha(root/'portable.json'),source_sha256=sha(__file__), producer_commit=a.source_commit)
    (out/'COMPLETE.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='bindings'}))

if __name__=='__main__':main()
