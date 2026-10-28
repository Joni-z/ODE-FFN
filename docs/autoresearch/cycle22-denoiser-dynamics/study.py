"""Check objective-level identities on fixed existing-law probes; no training."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,hashlib,importlib.util,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
WORK=Path('/Users/mr.z/ode-workspace-prep')
CHECK=str(WORK/'.venv-test/bin/python')
LAWS={'gaussian_aniso_2d':WORK/'data/synthetic/v1','gmm_ring8_2d':WORK/'data/synthetic/v1','rotated_product_bimodal_16d':WORK/'data/synthetic/product16_v1'}
TIMES=(.1,.3,.5,.7,.9)
def load(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def paths(root):return {name:Path(p) for name,p in load(root/'settings.json')['law_roots'].items()}
def verify(root):
    for file,digest in load(root/'bindings.json').items():assert sha(file)==digest,file
    for name,parent in paths(root).items():
        manifest=load(parent/'manifest.json')
        for file in ('distribution.json','validation.npz'):
            path=parent/name/file;assert sha(path)==manifest['files'][f'{name}/{file}']['sha256'],path
def source_data(root):
    name='cycle22_frozen_data';path=root/'source/frozen_data.py'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return sys.modules[name]
def numpy_law(root,name):
    d=load(paths(root)[name]/name/'distribution.json')['distribution'];data=source_data(root)
    return data.RotatedProductMixture(**d) if 'locations' in d else data.GaussianMixture(**d)
def moments(root,name,y,r):
    d=load(paths(root)[name]/name/'distribution.json')['distribution'];cast=lambda x:torch.as_tensor(x,dtype=torch.float64)
    if 'locations' in d:
        a=cast(d['locations']);v=cast(d['variances']);rotation=cast(d['rotation']);q=y@rotation;s=v+r;h=torch.tanh(a*q/s)
        mean=(v/s*q+r/s*a*h)@rotation.T
        var=v*r/s+(r*a/s)**2*(1-h*h);cov=(rotation*var)@rotation.T
        return mean,cov
    mu=cast(d['means']);v=cast(d['variances']);w=cast(d['weights']);s=v+r;residual=y-mu
    probability=torch.softmax(torch.log(w)-.5*torch.sum(torch.log(s)+residual**2/s,dim=-1),dim=0)
    cm=mu+v/s*residual;mean=torch.sum(probability[:,None]*cm,dim=0);center=cm-mean
    cov=torch.einsum('k,ki,kj->ij',probability,center,center)+torch.diag(torch.sum(probability[:,None]*v*r/s,dim=0))
    return mean,cov
def prepare(root):
    assert not root.exists();root.mkdir();(root/'source').mkdir();doc=Path(__file__).resolve().parent;repo=doc.parents[2]
    for file in doc.iterdir():
        if file.is_file():shutil.copyfile(file,root/'source'/file.name)
    shutil.copyfile(repo/'synthetic/data.py',root/'source/frozen_data.py')
    write(root/'settings.json',dict(source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),law_roots={k:str(p) for k,p in LAWS.items()},times=TIMES,validation_indices=list(range(8)),fixture_noise_seed=0,new_training=False))
    files=[*(root/'source').iterdir(),root/'settings.json']
    for name,parent in LAWS.items():files.extend([parent/'manifest.json',parent/name/'distribution.json',parent/name/'validation.npz'])
    write(root/'bindings.json',{str(p):sha(p) for p in files if p.is_file()});verify(root);print('PREPARED',flush=True)
def controls():
    t=torch.tensor(.5,dtype=torch.float64);z=torch.tensor([1.],dtype=torch.float64)
    def mean(q,t,v=1.):return t*v*q/(t*t*v+(1-t)**2)
    def endpoint(q,t):return q/torch.sqrt(t*t+(1-t)**2)
    def material(fn):
        u=(mean(z,t)-z)/(1-t)
        return torch.func.jacrev(lambda time:fn(z,time))(t)+torch.func.jacrev(lambda q:fn(q,t))(z)@u
    d_drift=float(material(mean)[0]);f_drift=float(material(endpoint)[0]);assert abs(d_drift-2)<1e-12 and abs(f_drift)<1e-12
    wrong_excess=(4/3-1)**2*.5/.5**2;endpoint_excess=2*(np.sqrt(2)-1)**2;assert abs(wrong_excess-2/9)<1e-12
    atomic=[]
    for label,atoms,q in [('symmetric',[-1.,1.],[.5,.5]),('skew',[-.5,2.],[.8,.2])]:
        x=torch.tensor(atoms,dtype=torch.float64);desired=torch.tensor(q,dtype=torch.float64);prior=desired*torch.exp(x*x/2);prior=prior/prior.sum()
        def posterior(precision):return torch.sum(torch.softmax(prior.log()-.5*precision*x*x,dim=0)*x)
        precision=torch.tensor(1.,dtype=torch.float64);weight=torch.softmax(prior.log()-.5*x*x,dim=0);m=posterior(precision);center=x-m;variance=torch.sum(weight*center**2);third=torch.sum(weight*center**3)
        derivative=torch.func.jacrev(posterior)(precision);rhs=variance*(-m)-third/2
        assert abs(float(derivative-rhs))<1e-12 and abs(float(m))<1e-12 and abs(float(variance)-1)<1e-12
        expected=0. if label=='symmetric' else -.75;assert abs(float(derivative)-expected)<1e-12
        atomic.append(dict(label=label,atoms=atoms,prior_weights=prior.tolist(),posterior_weights=weight.tolist(),mean=float(m),variance=float(variance),third_moment=float(third),precision_derivative=float(derivative),covariance_only_drift=float(-variance*m)))
    return dict(gaussian_at_t_half_z_one=dict(denoiser=1.,endpoint=float(np.sqrt(2)),denoiser_material_drift=d_drift,endpoint_material_drift=f_drift,endpoint_excess_weighted_risk=endpoint_excess,wrong_prior_excess_weighted_risk=wrong_excess),atomic=atomic)
def run(root):
    torch.set_num_threads(1);verify(root);out=root/'output';out.mkdir();(out/'cases').mkdir();start=time.monotonic();rows=[]
    for name,parent in paths(root).items():
        with np.load(parent/name/'validation.npz') as f:clean=f['x1'][:8].astype(np.float64)
        noise=np.random.default_rng(0).standard_normal(clean.shape);oracle=numpy_law(root,name)
        for time_value in TIMES:
            for index,(target,source) in enumerate(zip(clean,noise)):
                assert time.monotonic()-start<300
                t=torch.tensor(time_value,dtype=torch.float64);z=t*torch.from_numpy(target)+(1-t)*torch.from_numpy(source);y=z/t;r=((1-t)/t)**2
                mean,cov=moments(root,name,y,r);fn=lambda value:moments(root,name,value,r)[0]
                jac=torch.func.jacrev(fn)(y);hessian=torch.func.hessian(fn)(y);lap=hessian.diagonal(dim1=-2,dim2=-1).sum(-1)
                dr=torch.func.jacrev(lambda variance:moments(root,name,y,variance)[0])(r)
                dt=torch.func.jacrev(lambda time:moments(root,name,z/time,((1-time)/time)**2)[0])(t)
                jz=jac/t;lapz=lap/(t*t);variance_terms=(.5*lap,jac@(mean-y)/r)
                fm_terms=(-(1-t)/t*lapz,-2/(1-t)*(jz@mean),(1+t)/(t*(1-t))*(jz@z))
                variance_residual=dr-sum(variance_terms);fm_residual=dt-sum(fm_terms)
                vs=max(float(x.abs().max()) for x in [dr,*variance_terms]);ts=max(float(x.abs().max()) for x in [dt,*fm_terms])
                assert float(variance_residual.abs().max())<=1e-8*(1+vs) and float(fm_residual.abs().max())<=1e-8*(1+ts)
                assert float((r*jac-cov).abs().max())<=1e-9 and float((jac-jac.T).abs().max())<=1e-10
                minimum=float(torch.linalg.eigvalsh((jac+jac.T)/2).min());assert minimum>=-1e-10
                native_mean=z.numpy()+(1-time_value)*oracle.velocity(z.numpy()[None],time_value)[0];error=float(np.max(np.abs(mean.numpy()-native_mean)));assert error<=1e-9
                drift=dt+jz@((mean-z)/(1-t));label=f'{name}-{time_value:.1f}-{index}'
                arrays=dict(z=z.numpy(),y=y.numpy(),t=t.numpy(),r=r.numpy(),mean=mean.numpy(),covariance=cov.numpy(),jacobian=jac.numpy(),laplacian=lap.numpy(),variance_derivative=dr.numpy(),time_derivative=dt.numpy(),variance_lap_term=variance_terms[0].numpy(),variance_transport_term=variance_terms[1].numpy(),fm_lap_term=fm_terms[0].numpy(),fm_D_term=fm_terms[1].numpy(),fm_z_term=fm_terms[2].numpy(),material_drift=drift.numpy(),variance_residual=variance_residual.numpy(),fm_residual=fm_residual.numpy(),target=target,source=source)
                np.savez(out/'cases'/f'{label}.npz',**arrays)
                rows.append(dict(dataset=name,time=time_value,index=index,dim=len(y),mean_oracle_maxerror=error,variance_PDE_maxerror=float(variance_residual.abs().max()),FM_PDE_maxerror=float(fm_residual.abs().max()),jacobian_covariance_maxerror=float((r*jac-cov).abs().max()),jacobian_min_eigenvalue=minimum,laplacian_RMS=float(torch.sqrt(torch.mean(lap**2))),variance_derivative_RMS=float(torch.sqrt(torch.mean(dr**2))),material_drift_RMS=float(torch.sqrt(torch.mean(drift**2)))))
                write(out/'status.json',dict(state='RUNNING',processed=len(rows),total=120,seconds=time.monotonic()-start));print(json.dumps(dict(processed=len(rows),dataset=name)),flush=True)
    assert len(rows)==120;write(out/'controls.json',controls());write(out/'result.json',dict(status='PASS',cases=rows,new_training=False,seconds=time.monotonic()-start));verify(root)
    write(out/'status.json',dict(state='COMPLETE',processed=120,total=120,seconds=time.monotonic()-start))
def audit(root):
    torch.set_num_threads(1);verify(root);out=root/'audit';out.mkdir();formal=load(root/'output/result.json');assert formal['status']=='PASS' and len(formal['cases'])==120
    original={str(f):sha(f) for f in (root/'output').rglob('*') if f.is_file()};records=[];started=time.monotonic()
    for row in formal['cases']:
        name=row['dataset'];label=f"{name}-{row['time']:.1f}-{row['index']}";oracle=numpy_law(root,name)
        def mean_y(q,r):
            time_value=1/(1+np.sqrt(r));z=time_value*q
            return z+(1-time_value)*oracle.velocity(z[None],time_value)[0]
        def mean_z(z,t):return z+(1-t)*oracle.velocity(z[None],t)[0]
        with np.load(root/'output/cases'/f'{label}.npz') as bank:a={k:bank[k] for k in bank.files}
        y=a['y'];r=float(a['r']);t=float(a['t']);z=a['z'];m=mean_y(y,r);np.testing.assert_allclose(m,a['mean'],rtol=0,atol=1e-9)
        for h in (1e-4,5e-5):
            jac=np.zeros_like(a['jacobian']);lap=np.zeros_like(m)
            for j in range(len(y)):
                d=np.zeros_like(y);d[j]=h;plus=mean_y(y+d,r);minus=mean_y(y-d,r);jac[:,j]=(plus-minus)/(2*h);lap+=(plus-2*m+minus)/(h*h)
            dr=(mean_y(y,r+h)-mean_y(y,r-h))/(2*h);dt=(mean_z(z,t+h)-mean_z(z,t-h))/(2*h)
            checks={'jacobian':jac,'laplacian':lap,'variance_derivative':dr,'time_derivative':dt};errors={}
            for key,value in checks.items():
                error=float(np.max(np.abs(value-a[key])));scale=1+float(np.max(np.abs(a[key])));assert error<=2e-3*scale,(label,h,key,error,scale);errors[key+'_maxerror']=error
            variance_terms=(.5*lap,jac@(m-y)/r);vr=dr-sum(variance_terms);vs=max(np.max(np.abs(v)) for v in [dr,*variance_terms])
            jz=jac/t;lapz=lap/t**2;terms=(-(1-t)/t*lapz,-2/(1-t)*(jz@m),(1+t)/(t*(1-t))*(jz@z));tr=dt-sum(terms);ts=max(np.max(np.abs(v)) for v in [dt,*terms])
            assert np.max(np.abs(vr))<=1e-2*(1+vs) and np.max(np.abs(tr))<=1e-2*(1+ts)
            records.append(dict(dataset=name,time=t,index=row['index'],h=h,variance_PDE_maxerror=float(np.max(np.abs(vr))),FM_PDE_maxerror=float(np.max(np.abs(tr))),**errors))
    control=load(root/'output/controls.json');g=control['gaussian_at_t_half_z_one'];assert abs(g['denoiser_material_drift']-2)<1e-12 and abs(g['endpoint_material_drift'])<1e-12
    assert abs(g['wrong_prior_excess_weighted_risk']-2/9)<1e-12 and abs(g['endpoint_excess_weighted_risk']-2*(np.sqrt(2)-1)**2)<1e-12
    for row in control['atomic']:
        x=np.array(row['atoms']);prior=np.array(row['prior_weights']);w=prior*np.exp(-x*x/2);w=w/w.sum();m=w@x;v=w@(x-m)**2;third=w@(x-m)**3
        np.testing.assert_allclose([m,v,third,-v*m-third/2],[row['mean'],row['variance'],row['third_moment'],row['precision_derivative']],rtol=1e-12,atol=1e-12)
    for file,digest in original.items():assert sha(file)==digest,file
    verify(root);write(out/'result.json',dict(status='PASS',point_cases=120,finite_difference_cases=240,both_steps_retained=True,records=records,formal_files_unchanged=len(original),seconds=time.monotonic()-started));print('AUDIT_PASS',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','audit'));p.add_argument('--root',type=Path,required=True);a=p.parse_args();globals()[a.action](a.root)
