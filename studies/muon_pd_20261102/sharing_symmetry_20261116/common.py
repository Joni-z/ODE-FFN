"""Exact physical-space symmetry groups; pinned model, sampler and metric reuse."""
import json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coefficient_20261116'))
from gpu_resolution import base,torch,load,arrays,exclusive_gpu
from evaluate_screen import ev,cases as native_sample_stats
AXES=(0,7,15)

def law_for(root,case):
    spec=json.loads((Path(root)/case['dataset_manifest']).read_text())['datasets'][case['dataset']]
    return base.train.load_distribution(spec)

def group_matrices(dataset,law):
    if dataset=='gmm_ring8_2d':
        theta=np.arange(8)*np.pi/4;c=np.cos(theta);s=np.sin(theta)
        group=np.stack((c,-s,s,c),axis=1).reshape(8,2,2)
    else:
        assert dataset=='rotated_product_bimodal_16d' and law.dim==16
        group=[]
        for bits in np.ndindex(2,2,2):
            signs=np.ones(law.dim);signs[list(AXES)]=1-2*np.array(bits)
            group.append((law.rotation*signs[None,:])@law.rotation.T)
        group=np.stack(group)
    group[0]=np.eye(law.dim)
    return group.astype(np.float64)

def orbit(x,group):return torch.einsum('nd,ked->kne',x,group).reshape(-1,x.shape[-1])
def orbit_np(x,group):return np.einsum('nd,ked->kne',x.astype(float),group).reshape(-1,x.shape[-1])
def pullback(values,group):return torch.einsum('kne,ked->knd',values,group)

class Projection(torch.nn.Module):
    def __init__(self,model,group):
        super().__init__();self.model=model;self.register_buffer('group',group)
    def forward(self,x,t):
        count=len(x);g=self.group.to(x);transformed=orbit(x,g)
        times=t.expand(len(g),*t.shape).reshape(-1,1)
        values=self.model(transformed,times).reshape(len(g),count,x.shape[-1])
        return pullback(values,g).mean(0)

def sample_stats(value,law,dataset):
    result=native_sample_stats(value,law)
    if dataset=='rotated_product_bimodal_16d':
        latent=value@law.rotation;labels=(latent[:,AXES]>0).astype(int)@np.array([1,2,4])
        counts=np.bincount(labels,minlength=8)
        result.update(selected_sign_axes=list(AXES),selected_sign_counts=counts.tolist(),selected_sign_l1=float(np.abs(counts/len(value)-.125).sum()))
    return result

def decompose(pulled,oracle,times):
    average=pulled.mean(0);total=(pulled-oracle[None]).square().mean()
    symmetric=(average-oracle).square().mean();breaking=(pulled-average[None]).square().mean()
    residual=float((total-symmetric-breaking).abs());assert residual<1e-10*max(1.,float(total))
    profiles=[]
    for index in range(5):
        mask=(times[:,0]>=index/5)&(times[:,0]<(index+1)/5 if index<4 else times[:,0]<=1)
        if not bool(mask.any()):continue
        profiles.append(dict(start=index/5,end=(index+1)/5,base_points=int(mask.sum()),
            total=float((pulled[:,mask]-oracle[None,mask]).square().mean()),
            projected=float((average[mask]-oracle[mask]).square().mean()),
            breaking=float((pulled[:,mask]-average[None,mask]).square().mean())))
    return dict(total=float(total),projected=float(symmetric),breaking=float(breaking),identity_residual=residual,time_profiles=profiles)

def kernel_checks(law,dataset,device='cpu'):
    group_np=group_matrices(dataset,law);dim=law.dim
    assert np.max(np.abs(group_np.transpose(0,2,1)@group_np-np.eye(dim)))<1e-12
    for a in group_np:
        for b in group_np:assert min(np.max(np.abs(a@b-c)) for c in group_np)<1e-12
    torch.manual_seed(0);group=torch.from_numpy(group_np).to(device)
    x=torch.randn(32,dim,dtype=torch.double,device=device);t=torch.linspace(.05,.95,32,dtype=torch.double,device=device)[:,None]
    oracle=torch.from_numpy(law.velocity(x.cpu().numpy(),t.cpu().numpy())).to(device)
    gx=orbit(x,group);gt=t.expand(8,*t.shape).reshape(-1,1)
    actual=law.velocity(gx.cpu().numpy(),gt.cpu().numpy());expected=orbit(oracle,group).cpu().numpy()
    oracle_error=float(np.max(np.abs(actual-expected)));assert oracle_error<1e-9
    latent=torch.randn(dim,dim,dtype=torch.double,device=device)*.03
    if dim==2:
        expected_matrix=torch.trace(latent)/2*torch.eye(2,dtype=torch.double,device=device)+(latent-latent.T)/2
        matrix=latent
    else:
        q=torch.from_numpy(law.rotation).to(device);matrix=q@latent@q.T;mask=torch.ones_like(latent)
        for axis in AXES:mask[axis,:]=0;mask[:,axis]=0;mask[axis,axis]=1
        expected_matrix=q@(latent*mask)@q.T
    class Linear(torch.nn.Module):
        def forward(self,z,time):return z@matrix.T
    class Nonlinear(torch.nn.Module):
        def forward(self,z,time):return torch.sin(z@matrix.T)+.1*z.square()+time*z
    linear=Projection(Linear(),group);model=Nonlinear();projected=Projection(model,group)
    linear_error=float((linear(x,t)-x@expected_matrix.T).abs().max());assert linear_error<1e-10
    idem=float((Projection(projected,group)(x,t)-projected(x,t)).abs().max());assert idem<1e-10
    equiv=float((projected(gx,gt)-orbit(projected(x,t),group)).abs().max());assert equiv<1e-9
    pulled=pullback(model(gx,gt).reshape(8,len(x),dim),group);decomp=decompose(pulled,oracle,t)
    source=x[:4];base_end=base.train.generated_samples(linear,source,16)
    direct=base.train.generated_samples(linear,orbit(source,group),16)
    shortcut_error=float((direct-orbit(base_end,group)).abs().max());assert shortcut_error<1e-9
    end,path=ev.sample(linear,None,source,16,record=True);assert torch.equal(end,base_end) and path.shape==(17,4,dim)
    return dict(passed=True,dataset=dataset,oracle_max_error=oracle_error,linear_projection_max_error=linear_error,
        idempotence_max_error=idem,equivariance_max_error=equiv,mse_identity_residual=decomp['identity_residual'],
        linear_shortcut_max_error=shortcut_error,native_recording_exact=True)
