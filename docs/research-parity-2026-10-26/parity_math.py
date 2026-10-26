"""Antithetic identities and raw-linear weighted projection statistics."""
import numpy as np

def local_stats(kind,a,b):
    if kind=='silu':
        m=(a['pre']+b['pre'])/2;d=(a['pre']-b['pre'])/2
        odd=(a['out']-b['out'])/2;base=d/2;r=odd-base;bound=.75*np.abs(m)*np.minimum(np.abs(d),1)
        violation=np.max(np.abs(r)-bound,axis=1)
        assert violation.max()<1e-11
        return dict(center=np.mean(m*m,1),input_odd=np.mean(d*d,1),odd=np.mean(odd*odd,1),
            base=np.mean(base*base,1),added=np.mean(r*r,1),bound=np.mean(bound*bound,1),bound_violation=violation)
    if kind=='sigmoid':
        m=(a['pre']+b['pre'])/2;d=(a['pre']-b['pre'])/2;odd=(a['out']-b['out'])/2
        return dict(center=np.mean(m*m,1),input_odd=np.mean(d*d,1),odd=np.mean(odd*odd,1))
    if kind=='clock':
        even=(a['out']+b['out'])/2;odd=(a['out']-b['out'])/2
        return dict(even_value=even.mean(1),even_square=(even*even).mean(1),odd=(odd*odd).mean(1))
    du=(a['gate']-b['gate'])/2;dv=(a['value']-b['value'])/2;odd=(a['out']-b['out'])/2
    intrinsic=.5*du*dv*np.tanh(du/2) if kind=='product_swiglu' else dv/2
    return dict(odd=np.mean(odd*odd,1),intrinsic=np.mean(intrinsic*intrinsic,1),
        cross=np.mean(odd*intrinsic,1),center_effect=np.mean((odd-intrinsic)**2,1))

def projection_stats(x,plus,minus,oracle,weights):
    """Weights[resample,row]; fields[model,row,dim]. No component clipping."""
    odd=(plus-minus)/2;even=(plus+minus)/2;m,n,d=odd.shape
    y=np.concatenate([odd.transpose(1,0,2).reshape(n,m*d),oracle],axis=1)
    w=np.asarray(weights,dtype=np.float64);sums=w.sum(1)
    gram=(w@np.einsum('ni,nj->nij',x,x).reshape(n,d*d)/sums[:,None]).reshape(-1,d,d)
    cross=(w@np.einsum('ni,nj->nij',x,y).reshape(n,-1)/sums[:,None]).reshape(len(w),d,-1)
    coefficients=np.linalg.solve(gram,cross)
    cg=cross.reshape(len(w),d,m+1,d);bg=coefficients.reshape(len(w),d,m+1,d)
    raw_f=w@np.mean(odd*odd,axis=2).T/sums[:,None]
    raw_v=w@np.mean(oracle*oracle,axis=1)/sums
    raw_cross=w@np.mean(odd*oracle[None],axis=2).T/sums[:,None]
    leakage=w@np.mean(even*even,axis=2).T/sums[:,None]
    linear_f=np.sum(cg[:,:,:m]*bg[:,:,:m],axis=(1,3))/d
    linear_v=np.sum(cg[:,:,m]*bg[:,:,m],axis=(1,2))/d
    linear_cross=np.sum(cg[:,:,:m]*bg[:,:,m,None,:],axis=(1,3))/d
    residual_f=raw_f-linear_f;residual_v=raw_v-linear_v;residual_cross=raw_cross-linear_cross
    linear_error=np.sum((cg[:,:,:m]-cg[:,:,m,None,:])*(bg[:,:,:m]-bg[:,:,m,None,:]),axis=(1,3))/d
    nonlinear_error=residual_f+residual_v[:,None]-2*residual_cross
    odd_error=raw_f+raw_v[:,None]-2*raw_cross
    total=odd_error+leakage
    np.testing.assert_allclose(total,linear_error+nonlinear_error+leakage,atol=1e-11,rtol=1e-11)
    return dict(total=total,even=leakage,linear_odd=linear_error,nonlinear_odd=nonlinear_error,
        nonlinear_alignment=residual_cross/residual_v[:,None],nonlinear_energy_ratio=residual_f/residual_v[:,None],
        oracle_nonlinear_energy=residual_v,coefficients=coefficients)

def self_test():
    rho=lambda x:x/(1+np.exp(-x));sigmoid=lambda x:1/(1+np.exp(-x))
    z=np.linspace(-20,20,501);np.testing.assert_allclose(rho(z)-rho(-z),z,atol=1e-13)
    rng=np.random.default_rng(0);x=rng.normal(size=(128,3));oracle=np.tanh(x)+.2*x
    f=np.stack([.7*oracle+.1*x,.5*oracle+.2*x*x]);fm=np.stack([-.7*oracle-.1*x,-.5*oracle+.2*x*x])
    weights=np.stack([np.ones(128),rng.multinomial(128,np.full(128,1/128))]);stats=projection_stats(x,f,fm,oracle,weights)
    paired=np.mean(((f-oracle)**2+(fm+oracle)**2)/2,axis=2)
    np.testing.assert_allclose(stats['total'],weights@paired.T/weights.sum(1)[:,None],atol=1e-13)
    for j,w in enumerate(weights):
        root=np.sqrt(w[:,None]);error=((f-fm)/2-oracle).transpose(1,0,2).reshape(128,-1)
        beta=np.linalg.lstsq(x*root,error*root,rcond=1e-12)[0]
        residual=(error-x@beta).reshape(128,2,3)
        np.testing.assert_allclose(stats['nonlinear_odd'][j],np.average(np.mean(residual**2,2),axis=0,weights=w),atol=1e-13)
    p=rng.normal(size=(128,9));q=rng.normal(size=(128,9));a={'pre':p,'out':rho(p)};b={'pre':q,'out':rho(q)}
    local=local_stats('silu',a,b);assert local['bound_violation'].max()<1e-12
    u=rng.normal(size=(128,9));v=rng.normal(size=(128,9))
    np.testing.assert_allclose((rho(u)*v+rho(-u)*v)/2,.5*u*v*np.tanh(u/2),atol=1e-13)
    np.testing.assert_allclose((sigmoid(u)*v+sigmoid(-u)*v)/2,v/2,atol=1e-13)
    affine=x@rng.normal(size=(3,3));lin=projection_stats(x,affine[None],-affine[None],oracle,weights)
    np.testing.assert_allclose(lin['nonlinear_energy_ratio'],0,atol=1e-12)
    return dict(status='PASS',checks=['SiLU identity','antithetic risk split','weighted SVD projection witness',
        'offset bound','SwiGLU intrinsic odd term','sigmoid-GLU intrinsic odd term','pure linear field witness'])
