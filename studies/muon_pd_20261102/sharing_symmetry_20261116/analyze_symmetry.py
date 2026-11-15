"""Independent NumPy verification of group sampling, error splits and mode flux."""
import argparse,hashlib,json,math,subprocess
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ah(x):
    x=np.ascontiguousarray(x);return hashlib.sha256(str((x.shape,x.dtype.str)).encode()+x.tobytes()).hexdigest()
def sw2(x,y,d):
    a=np.sort(x.astype(float)@d.astype(float).T,axis=0);b=np.sort(y.astype(float)@d.astype(float).T,axis=0)
    return float(np.sqrt(np.mean((a-b)**2)))
def bins(x,spec):
    law=spec['distribution']
    if spec['dim']==2:return np.argmax(x@np.array(law['means']).T,axis=1)
    latent=x@np.array(law['rotation']);return (latent[:,[0,7,15]]>0).astype(int)@np.array([1,2,4])
def counts(x,spec):return np.bincount(bins(x,spec),minlength=8)
def flux(source,value,spec):
    matrix=np.bincount(8*bins(source,spec)+bins(value,spec),minlength=64).reshape(8,8)
    return dict(matrix=matrix.tolist(),changed_fraction=float(1-np.trace(matrix)/len(value)),
                net_mass_change=((matrix.sum(0)-matrix.sum(1))/len(value)).tolist(),net_l1=float(np.abs(matrix.sum(0)-matrix.sum(1)).sum()/len(value)))
def balance(x,spec):return float(np.abs(counts(x,spec)/len(x)-.125).sum())
def gaps(x,spec):
    if spec['dim']==2:return {}
    law=spec['distribution'];latent=x@np.array(law['rotation']);observed=[];expected=[]
    phi=lambda z:(1+math.erf(z/math.sqrt(2)))/2
    for axis in [0,7,15]:
        m=law['locations'][axis];s=math.sqrt(law['variances'][axis]);observed.append(float(np.mean(np.abs(latent[:,axis])<m/2)))
        expected.append(.5*(phi(-m/(2*s))-phi(-3*m/(2*s))+phi(3*m/(2*s))-phi(m/(2*s))))
    return dict(gap_mass=observed,gap_expected=expected,gap_error_max=float(np.max(np.abs(np.array(observed)-expected))))

def verify_case(root,folder,allow_witness=False,expected_commit=None):
    root=Path(root);folder=Path(folder);manifest=read(root/'portable.json');done=read(folder/'COMPLETE.json')
    assert done['passed'] and done['parameters_unchanged'] and done['dtype']=='float64'
    assert allow_witness or not done['witness']
    for filename,digest in done['source_sha256'].items():
        if expected_commit:
            file='studies/muon_pd_20261102/sharing_symmetry_20261116/'+filename
            content=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[3]),'show',expected_commit+':'+file])
            assert hashlib.sha256(content).hexdigest()==digest
        else:assert sha(Path(__file__).with_name(filename))==digest
    n=done['base_samples'];banks=done['banks'];points=done['field_points']
    assert (n,banks,points)==((16,2,32) if done['witness'] else (4096,4,4096))
    case=next(c for c in manifest['cases'] if c['id']==done['id']);assert done['checkpoint_sha256']==case['checkpoint_sha256'] and done['assets_sha256']==case['assets_sha256']
    for key in ['checkpoint','config','dataset_manifest','assets']:assert sha(root/case[key])==case[key+'_sha256']
    assert done['portable_manifest_sha256']==sha(root/'portable.json')
    spec=read(root/case['dataset_manifest'])['datasets'][case['dataset']]
    with np.load(root/case['assets']) as z:data={k:z[k].copy() for k in z.files}
    binding=next(b for b in manifest['asset_bindings'] if b['dataset']==case['dataset'])
    assert {k:ah(v) for k,v in data.items()}==binding['arrays']
    d=spec['dim'];group=data['group'];assert group.shape==(8,d,d)
    if d==2:
        theta=np.arange(8)*np.pi/4;c=np.cos(theta);s=np.sin(theta);expected_group=np.stack((c,-s,s,c),axis=1).reshape(8,2,2)
    else:
        q=np.array(spec['distribution']['rotation']);expected_group=[]
        for bits in np.ndindex(2,2,2):
            signs=np.ones(d);signs[[0,7,15]]=1-2*np.array(bits);expected_group.append((q*signs[None])@q.T)
        expected_group=np.stack(expected_group)
    expected_group[0]=np.eye(d);assert np.allclose(group,expected_group,atol=1e-12,rtol=1e-12)
    for kind in ['source','target']:
        for b in range(4):
            expected=np.einsum('nd,ked->kne',data[f'{kind}_{b}'].astype(float),group).reshape(-1,d)
            assert np.array_equal(expected,data[f'orbit_{kind}_{b}'])
    assert sha(folder/'field-pulled.npy')==done['field_pulled_sha256'];pulled=np.load(folder/'field-pulled.npy');assert pulled.shape==(8,points,d)
    oracle=data['field_oracle'][:points];mean=pulled.mean(0)
    field=dict(total=float(np.mean((pulled-oracle[None])**2)),projected=float(np.mean((mean-oracle)**2)),breaking=float(np.mean((pulled-mean[None])**2)),base_mse=float(np.mean((pulled[0]-oracle)**2)))
    maximum=max(abs(field[k]-done['field'][k]) for k in field);assert maximum<1e-10
    assert abs(field['total']-field['projected']-field['breaking'])<1e-10
    for profile in done['field']['time_profiles']:
        t=data['field_t'][:points,0];mask=(t>=profile['start'])&((t<profile['end']) if profile['end']<1 else (t<=1));assert int(mask.sum())==profile['base_points']
        actual=[np.mean((pulled[:,mask]-oracle[None,mask])**2),np.mean((mean[mask]-oracle[mask])**2),np.mean((pulled[:,mask]-mean[None,mask])**2)]
        error=max(abs(float(v)-profile[k]) for v,k in zip(actual,['total','projected','breaking']));maximum=max(maximum,error);assert error<1e-10
    orbit_slice=lambda kind,b,m:data[f'orbit_{kind}_{b}'].reshape(8,4096,d)[:,:m].reshape(-1,d)
    assert len(done['sample_rows'])==2*banks and len(done['pooled'])==2 and len(done['resolution'])==4
    modes={};rows=[];pool=[]
    for mode in ['original','projected']:
        values=[];targets=[]
        for b in range(banks):
            row=next(r for r in done['sample_rows'] if r['mode']==mode and r['bank']==b);f=folder/row['generated_file'];assert sha(f)==row['generated_sha256']
            x=np.load(f);target=orbit_slice('target',b,n);source=orbit_slice('source',b,n);assert x.shape==target.shape and np.isfinite(x).all()
            score=sw2(x,target,data['directions']);maximum=max(maximum,abs(score-row['stratified_sw2']));assert abs(score-row['stratified_sw2'])<1e-10
            count=counts(x,spec);key='counts' if d==2 else 'selected_sign_counts';assert row[key]==count.tolist()
            key='sector_l1' if d==2 else 'selected_sign_l1';assert abs(balance(x,spec)-row[key])<1e-12
            coupling=flux(source,x,spec);assert max(counts(source,spec))-min(counts(source,spec))<=2
            assert abs(coupling['net_l1']-balance(x,spec))<1e-12
            gap=gaps(x,spec)
            for k,v in gap.items():assert np.max(np.abs(np.array(v)-row[k]))<1e-12
            values.append(x);targets.append(target)
            rows.append(dict(id=case['id'],dataset=case['dataset'],family=case['family'],mode=mode,bank=b,sw2=score,balance_l1=balance(x,spec),coupling=coupling,**gap))
        joined=np.concatenate(values);target=np.concatenate(targets);score=sw2(joined,target,data['directions']);ref=next(p for p in done['pooled'] if p['mode']==mode)
        maximum=max(maximum,abs(score-ref['pooled_stratified_sw2']));assert abs(score-ref['pooled_stratified_sw2'])<1e-10
        assert counts(joined,spec).tolist()==ref['counts' if d==2 else 'selected_sign_counts']
        gap=gaps(joined,spec)
        for k,v in gap.items():assert np.max(np.abs(np.array(v)-ref[k]))<1e-12
        pool.append(dict(id=case['id'],dataset=case['dataset'],family=case['family'],mode=mode,pooled_sw2=score,bank_mean=float(np.mean([r['sw2'] for r in rows if r['mode']==mode])),bank_sd=float(np.std([r['sw2'] for r in rows if r['mode']==mode],ddof=1)),balance_l1=balance(joined,spec),**gap))
        modes[mode]=values
    for r in done['resolution']:
        f=folder/r['generated_file'];assert sha(f)==r['generated_sha256'];x=np.load(f);target=orbit_slice('target',0,r['base_samples'])
        score=sw2(x,target,data['directions']);maximum=max(maximum,abs(score-r['stratified_sw2']));assert abs(score-r['stratified_sw2'])<1e-10
        fine=next(v for v in done['resolution'] if v['mode']==r['mode'] and v['heun_steps']==256);finest=np.load(folder/fine['generated_file'])
        shift=float(np.sqrt(np.mean(np.sum((x-finest)**2,axis=1))));assert abs(shift-r['rms_endpoint_shift_to256'])<1e-12
    for r in done['paths']:
        f=folder/r['file'];assert sha(f)==r['sha256'];x=np.load(f);assert x.shape==(129,16,d) and np.isfinite(x).all();assert np.array_equal(x[0],data['source_0'][:16].astype(float))
    return dict(passed=True,witness=done['witness'],base_samples=n,banks=banks,field_points=points,id=case['id'],field=field,bank_rows=rows,pooled=pool,max_numpy_error=maximum,completion_sha256=sha(folder/'COMPLETE.json'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--runtime');p.add_argument('--witness-case');p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);assert not out.exists()
    if a.witness_case:
        result=verify_case(a.root,a.witness_case,True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,witness=True,id=result['id'],max_numpy_error=result['max_numpy_error'])));return
    assert a.runtime;runtime=Path(a.runtime);terminal=read(runtime/'queue-finished.json');assert terminal['passed'] and len(terminal['completed'])==8 and not terminal['unstarted']
    assert terminal['portable_manifest_sha256']==sha(Path(a.root)/'portable.json')
    results=[]
    for row in terminal['completed']:
        assert row['exit_code']==0 and row['complete'] and row['physical_gpu'] in [4,5,6,7]
        folder=runtime/'cases'/row['id'];assert read(folder/'COMPLETE.json')['gpu_uuid']==row['gpu_uuid']
        results.append(verify_case(a.root,folder,expected_commit=terminal['frozen_commit']))
    assert len({r['id'] for r in results})==8
    out.mkdir(parents=True);(out/'verified-cases.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    (out/'COMPLETE.json').write_text(json.dumps(dict(passed=True,cases=8,bank_rows=64,pooled_rows=16,max_numpy_error=max(r['max_numpy_error'] for r in results),source_sha256=sha(__file__),queue_receipt_sha256=sha(runtime/'queue-finished.json')),indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=8,output=str(out))))

if __name__=='__main__':main()
