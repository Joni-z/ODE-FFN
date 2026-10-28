"""Data-free fixed-prior algebra; no learned model or quality experiment."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import argparse,csv,hashlib,itertools,json,platform,resource,sys,time
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import torch

TOL=3e-10

def softmax(x):
    w=np.exp(x-x.max());return w/w.sum()

def priors():
    def make(name,pi,mu,cov=None,factors=None):
        mu=np.asarray(mu,np.float64)
        if factors is None:factors=[np.linalg.cholesky(v) for v in np.asarray(cov,np.float64)]
        factors=[np.asarray(v,np.float64) for v in factors]
        return dict(name=name,pi=np.asarray(pi,np.float64)/sum(pi),mu=mu,factors=factors,cov=np.stack([v@v.T for v in factors]))
    empty=lambda d,k:[np.empty((d,0)) for _ in range(k)]
    q=np.array([.8,.2]);atoms=np.array([-.5,2.]);p=q*np.exp(atoms*atoms/2)
    common=np.array([[.6,.15],[.15,.35]])
    yield make('symmetric_atoms',[.5,.5],[[-1.],[1.]],factors=empty(1,2))
    yield make('skew_atoms',p,[[-.5],[2.]],factors=empty(1,2))
    yield make('shared2',[.2,.5,.3],[[-1.2,.3],[.7,-.8],[.4,1.1]],cov=[common]*3)
    yield make('unequal3',[.25,.4,.35],[[-1.1,.2,.6],[.8,-.7,.3],[.1,1.,-.9]],cov=[
        [[.7,.1,-.05],[.1,.4,.08],[-.05,.08,.3]],
        [[.3,-.06,.04],[-.06,.8,.02],[.04,.02,.5]],
        [[.45,.07,.02],[.07,.35,-.04],[.02,-.04,.6]]])
    yield make('singular2',[.4,.25,.35],[[-.8,.2],[.5,-.6],[.4,.9]],factors=[[[np.sqrt(.3)],[0]],np.empty((2,0)),[[.3],[.4]]])
    yield make('single2',[1.],[[.2,-.4]],cov=[[[.7,.2],[.2,.3]]])

def state(p,y,lam):
    k,d=p['mu'].shape;means=[];cov=[];logits=[]
    for pi,mu,sigma in zip(p['pi'],p['mu'],p['cov']):
        matrix=np.eye(d)+lam*sigma;a=np.linalg.solve(matrix,np.eye(d));s=np.linalg.solve(matrix,sigma)
        r=y-mu;means.append(mu+lam*s@r);cov.append(s)
        sign,logdet=np.linalg.slogdet(matrix);assert sign>0
        logits.append(np.log(pi)-.5*logdet-.5*lam*(r@a@r))
    b,s,ell=np.array(means),np.array(cov),np.array(logits);w=softmax(ell);m=w@b;delta=b-m
    c=np.einsum('k,kij->ij',w,s)+np.einsum('k,ki,kj->ij',w,delta,delta)
    kappa=np.einsum('k,ki->i',w,delta*np.sum(delta*delta,axis=1)[:,None]+delta*np.trace(s,axis1=1,axis2=2)[:,None]+2*np.einsum('kij,kj->ki',s,delta))
    return dict(b=b,s=s,ell=ell,w=w,m=m,c=c,kappa=kappa)

def spectral(p,y,lam):
    # Independent backend and spectral coordinates, without NumPy solve paths.
    b=[];s=[];ell=[]
    for pi,mu,sigma in zip(p['pi'],p['mu'],p['cov']):
        mu=torch.tensor(mu);e,u=torch.linalg.eigh(torch.tensor(sigma));r=u.T@(y-mu);den=1+lam*e
        b.append(mu+u@((lam*e/den)*r));s.append((u*(e/den))@u.T)
        ell.append(np.log(pi)-.5*torch.log1p(lam*e).sum()-.5*lam*(r*r/den).sum())
    b,s,ell=torch.stack(b),torch.stack(s),torch.stack(ell);w=torch.softmax(ell,0);m=w@b;delta=b-m
    c=torch.einsum('k,kij->ij',w,s)+torch.einsum('k,ki,kj->ij',w,delta,delta)
    return dict(b=b,s=s,ell=ell,w=w,m=m,c=c)

def cubature(p,y,lam,values):
    points=[];weights=[];nodes=np.array([-np.sqrt(3.),0,np.sqrt(3.)]);mass=np.array([1/6,2/3,1/6])
    for w,b,f in zip(values['w'],values['b'],p['factors']):
        r=f.shape[1]
        if r:
            eig,u=np.linalg.eigh(np.eye(r)+lam*f.T@f);assert np.all(eig>0)
            factor=(f@u)/np.sqrt(eig)
        else:factor=f
        for indices in itertools.product(range(3),repeat=r):
            index=np.array(indices,dtype=int);points.append(b+factor@nodes[index]);weights.append(w*mass[index].prod())
    x,a=np.array(points),np.array(weights);mean=a@x;delta=x-mean
    return mean,np.einsum('k,ki,kj->ij',a,delta,delta),np.einsum('k,ki,k->i',a,delta,np.sum(delta*delta,axis=1))

def ordinary(p,y,lam):
    mu=p['mu'];d=mu.shape[1]
    if np.all(p['cov']==0):
        logits=mu@(lam*y)+np.log(p['pi'])-.5*lam*np.sum(mu*mu,axis=1)
        return softmax(logits)@mu,'prototype_attention'
    if np.all(p['cov']==p['cov'][0]):
        a=np.linalg.inv(np.eye(d)+lam*p['cov'][0]);q=lam*a@y
        bias=np.log(p['pi'])-.5*lam*np.einsum('ki,ij,kj->k',mu,a,mu)
        return (np.eye(d)-a)@y+softmax(mu@q+bias)@(mu@a.T),'shared_covariance_attention'
    logits=[];values=[]
    for pi,center,sigma in zip(p['pi'],mu,p['cov']):
        matrix=np.eye(d)+lam*sigma;a=np.linalg.inv(matrix)
        logits.append(np.log(pi)-.5*np.linalg.slogdet(matrix)[1]-.5*lam*y@a@y+lam*y@a@center-.5*lam*center@a@center)
        values.append((np.eye(d)-a)@y+a@center)
    return softmax(np.array(logits))@np.array(values),'quadratic_gated_affine_experts'

def grids(d):return [np.zeros(d),np.linspace(-.7,.8,d),np.linspace(1.1,-.6,d)]
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def table(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(root):
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);tick=time.monotonic();settings=json.loads((root/'settings.json').read_text())
    bindings=json.loads((root/'bindings.json').read_text())
    for path,digest in bindings.items():assert sha(path)==digest
    out=root/'results';out.mkdir();families=list(priors());fixed=[];times=[];ode=[];maxima={}
    def check(name,actual,expected):
        actual=np.asarray(actual);expected=np.asarray(expected);error=float(np.max(np.abs(actual-expected))) if actual.size else 0.
        maxima[name]=max(maxima.get(name,0.),error);np.testing.assert_allclose(actual,expected,rtol=TOL,atol=TOL,err_msg=name)
        return error
    for p in families:
        k,d=p['mu'].shape
        for index,y in enumerate(grids(d)):
            for lam in (0.,.2,1.,4.):
                v=state(p,y,lam);yt=torch.tensor(y);lt=torch.tensor(lam);other=spectral(p,yt,lt)
                discrepancy=max(check('independent_'+name,other[name].detach().numpy(),v[name]) for name in ('b','s','ell','w','m','c'))
                qm,qc,qk=cubature(p,y,lam,v)
                qerror=max(check('cubature_'+name,a,b) for name,a,b in [('mean',qm,v['m']),('covariance',qc,v['c']),('third',qk,v['kappa'])])
                def packed(value):
                    a=spectral(p,yt,value);return torch.cat([a[name].reshape(-1) for name in ('b','s','ell','w','m')])
                actual=torch.autograd.functional.jacobian(packed,lt,vectorize=True).detach().numpy()
                bdot=np.einsum('kij,kj->ki',v['s'],y-v['b']);sdot=-v['s']@v['s']
                adot=-.5*(np.trace(v['s'],axis1=1,axis2=2)+np.sum((y-v['b'])**2,axis=1));wdot=v['w']*(adot-v['w']@adot)
                drift=v['c']@(y-v['m'])-.5*v['kappa']
                expected=np.concatenate([a.reshape(-1) for a in (bdot,sdot,adot,wdot,drift)])
                derivative=check('complete_precision_state',actual,expected)
                jac=torch.autograd.functional.jacobian(lambda obs:spectral(p,obs,lt)['m'],yt,vectorize=True).detach().numpy()
                check('state_Jacobian',jac,lam*v['c']);endpoint,kind=ordinary(p,y,lam)
                ordinary_error=check('ordinary_endpoint',endpoint,v['m'])
                hessian=0.
                if k>1:
                    h=torch.autograd.functional.hessian(lambda obs:torch.diff(spectral(p,obs,lt)['ell'][:2])[0],yt,vectorize=True).detach().numpy()
                    # diff is ell_1 - ell_0.
                    a0=np.linalg.inv(np.eye(d)+lam*p['cov'][0]);a1=np.linalg.inv(np.eye(d)+lam*p['cov'][1])
                    check('logit_ratio_Hessian',h,-lam*(a1-a0));hessian=float(np.linalg.norm(h))
                fixed.append(dict(prior=p['name'],observation=index,precision=lam,conditioning_max=discrepancy,cubature_max=qerror,
                    precision_state_max=derivative,ordinary_kind=kind,ordinary_endpoint_max=ordinary_error,logit_ratio_Hessian_norm=hessian,
                    third_moment_norm=float(np.linalg.norm(v['kappa'])),covariance_only_drift_error=float(np.linalg.norm(.5*v['kappa']))))
            # Exactly one integration per observation, with three retained endpoints.
            shape=(k,d);n=k*d;s=k*d*d
            initial=np.concatenate([p['mu'].ravel(),p['cov'].ravel(),np.log(p['pi'])])
            def rhs(value,packed):
                b=packed[:n].reshape(shape);cov=packed[n:n+s].reshape(k,d,d)
                db=np.einsum('kij,kj->ki',cov,y-b);ds=-cov@cov
                dl=-.5*(np.trace(cov,axis1=1,axis2=2)+np.sum((y-b)**2,axis=1))
                return np.concatenate([db.ravel(),ds.ravel(),dl])
            solution=solve_ivp(rhs,(0,4),initial,method='DOP853',t_eval=[.2,1.,4.],rtol=1e-10,atol=1e-12)
            assert solution.success and solution.y.shape[1]==3
            for lam,packed in zip(solution.t,solution.y.T):
                v=state(p,y,lam);b=packed[:n].reshape(shape);cov=packed[n:n+s].reshape(k,d,d);ell=packed[n+s:]
                errors=[float(np.max(np.abs(a-z))) for a,z in [(b,v['b']),(cov,v['s']),(ell,v['ell']),(softmax(ell),v['w']),(softmax(ell)@b,v['m'])]]
                assert max(errors)<5e-8
                minimum=float(np.linalg.eigvalsh(cov).min());assert minimum>-TOL
                assert np.max(np.abs(cov-cov.transpose(0,2,1)))<TOL
                ode.append(dict(prior=p['name'],observation=index,precision=float(lam),full_state_and_mean_max=max(errors),minimum_covariance_eigenvalue=minimum,function_evaluations=solution.nfev))
        for index,z in enumerate(grids(d)):
            for t in (.15,.5,.85,.98):
                lam=t*t/(1-t)**2;y=z/t;v=state(p,y,lam);zd=torch.tensor(z);td=torch.tensor(t)
                def denoiser(obs,clock):return spectral(p,obs/clock,clock*clock/(1-clock)**2)['m']
                partial=torch.autograd.functional.jacobian(lambda clock:denoiser(zd,clock),td).detach().numpy()
                jac=torch.autograd.functional.jacobian(lambda obs:denoiser(obs,td),zd,vectorize=True).detach().numpy()
                ldot=2*t/(1-t)**3;precision_drift=v['c']@(y-v['m'])-.5*v['kappa']
                fixed_z=ldot*precision_drift-(lam/t)*v['c']@y
                te=check('FM_fixed_z_time',partial,fixed_z);je=check('FM_state',jac,(lam/t)*v['c'])
                def component_time(clock):
                    a=spectral(p,zd/clock,clock*clock/(1-clock)**2)
                    return torch.cat([a[name].reshape(-1) for name in ('b','s','ell','w','m')])
                full_time=torch.autograd.functional.jacobian(component_time,td,vectorize=True).detach().numpy()
                ydot=-y/t;residual=y-v['b'];adot=-.5*(np.trace(v['s'],axis1=1,axis2=2)+np.sum(residual*residual,axis=1))
                s_time=-ldot*(v['s']@v['s'])
                b_time=ldot*np.einsum('kij,kj->ki',v['s'],residual)+lam*np.einsum('kij,j->ki',v['s'],ydot)
                ell_time=ldot*adot-lam*(residual@ydot);w_time=v['w']*(ell_time-v['w']@ell_time)
                full_error=check('FM_complete_component_time',full_time,np.concatenate([a.reshape(-1) for a in (b_time,s_time,ell_time,w_time,fixed_z)]))
                ordinary_time_error=check('FM_ordinary_endpoint',ordinary(p,y,lam)[0],v['m'])
                velocity=(v['m']-z)/max(1-t,.05);vt=torch.tensor(velocity)
                material=torch.autograd.functional.jacobian(lambda clock:denoiser(zd+vt*(clock-td),clock),td).detach().numpy()
                chain=fixed_z+(lam/t)*v['c']@velocity;me=check('FM_capped_material',material,chain)
                times.append(dict(prior=p['name'],state=index,time=t,cap_active=(1-t)<.05,fixed_z_max=te,state_Jacobian_max=je,material_max=me,
                    complete_component_time_max=full_error,ordinary_endpoint_max=ordinary_time_error,
                    dropped_observation_term_error=float(np.max(np.abs(ldot*precision_drift-partial)))))
                assert time.monotonic()-tick<120
    counter=[]
    for p in families[:2]:
        v=state(p,np.zeros(1),1.);check('counterexample_mean',v['m'],[0.]);check('counterexample_covariance',v['c'],[[1.]])
        counter.append(dict(prior=p['name'],mean=v['m'].tolist(),covariance=v['c'].tolist(),third=v['kappa'].tolist(),
            correct_precision_drift=(-.5*v['kappa']).tolist(),covariance_only_drift=[0.]))
    check('counterexample_skew_third',counter[1]['third'],[1.5])
    assert (len(fixed),len(times),len(ode))==(72,72,54)
    assert max(v['logit_ratio_Hessian_norm'] for v in fixed if v['prior']=='unequal3')>1e-2
    assert max(v['dropped_observation_term_error'] for v in times)>1e-2
    for path,digest in bindings.items():assert sha(path)==digest
    table(out/'fixed-observation.csv',fixed);table(out/'FM-coordinates.csv',times);table(out/'integrated-states.csv',ode);write(out/'counterexample.json',counter)
    write(out/'priors.json',[dict(name=p['name'],pi=p['pi'].tolist(),mean=p['mu'].tolist(),covariance=p['cov'].tolist(),factors=[v.tolist() for v in p['factors']]) for p in families])
    summary=dict(status='PASS',source_commit=settings['source_commit'],data_free=True,fixed_priors=6,fixed_observation_cases=72,FM_coordinate_cases=72,integrated_endpoints=54,
        maximum_absolute_errors=maxima,integrated_state_maximum=max(v['full_state_and_mean_max'] for v in ode),
        unequal_covariance_logit_ratio_Hessian_max=max(v['logit_ratio_Hessian_norm'] for v in fixed if v['prior']=='unequal3'),
        dropping_FM_observation_term_max=max(v['dropped_observation_term_error'] for v in times),
        finite_family_closure_correct=True,all_priors_plain_linear_key_attention=False,solver_repackaging_nominated=False,
        neural_training=False,image_prior_fitted=False,new_quality_score=False,seconds=time.monotonic()-tick,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
        environment=dict(python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,device='cpu',threads=torch.get_num_threads()))
    write(out/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);main(parser.parse_args().root.resolve())
