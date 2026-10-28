"""Observed image frames and finite VJP-generated response subspaces."""
import copy,math,types
import torch
from torch.nn import functional as F

class PatchFrame:
    def __init__(self,basis):
        assert basis.ndim==2 and basis.shape[0]==48 and basis.dtype==torch.float64
        self.basis=basis;self.rank=basis.shape[1];self.dimension=64*self.rank
    def encode(self,x):
        p=F.unfold(x,kernel_size=4,stride=4).transpose(1,2)
        return (p@self.basis).reshape(len(x),-1)
    def lift(self,q):
        q=q.reshape(-1,64,self.rank);p=q@self.basis.T
        return F.fold(p.transpose(1,2),output_size=(32,32),kernel_size=4,stride=4)

def encoder_space(model):
    e=model.x_embedder
    assert not e.adaptive_bottleneck and not e.multiscale_patch_embed
    assert e.fine_proj1 is None and e.proj1.bias is None
    assert e.proj1.kernel_size==e.proj1.stride==(4,4) and e.num_patches==64
    a=e.proj1.weight.detach().double().reshape(32,48);b=e.proj2.weight.detach().double().reshape(64,32)@a
    _,s,v=torch.linalg.svd(b,full_matrices=True);rank=int((s>1e-10*s[0]).sum());assert 0<rank<=32
    u=v[:rank].T;null=v[rank:].T
    torch.testing.assert_close(u.T@u,torch.eye(rank,dtype=u.dtype),rtol=0,atol=1e-11)
    assert float((b@(torch.eye(48,dtype=b.dtype)-u@u.T)).abs().max())<1e-10
    return u,null,b,s

def intersection(left_null,right_null):
    null=torch.cat((left_null.T,right_null.T),0);_,s,v=torch.linalg.svd(null,full_matrices=True)
    rank=int((s>1e-10*s[0]).sum());u=v[rank:].T
    if u.shape[1]:
        torch.testing.assert_close(u.T@u,torch.eye(u.shape[1],dtype=u.dtype),rtol=0,atol=1e-11)
        assert float((null@u).abs().max())<1e-10
    return u

def norm_forward(self,h):
    variance=h.pow(2).mean(-1,keepdim=True)
    return self.weight*(h*torch.rsqrt(variance+self.variance_epsilon))

def attention_forward(self,x,rope,t_emb=None,distance_matrix=None):
    assert not any((self.time_dependent_qkv,self.adaptive_temperature,self.position_bias,self.head_specialization))
    b,n,c=x.shape;q,k,v=self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4)
    q,k=rope(self.q_norm(q)),rope(self.k_norm(k))
    score=torch.matmul(q,k.transpose(-2,-1))*(1.0/math.sqrt(q.size(-1)))
    weights=self.attn_drop(torch.softmax(score,dim=-1).to(v.dtype))
    h=torch.matmul(weights,v).transpose(1,2).reshape(b,n,c)
    return self.proj_drop(self.proj(h))

def literal_model(native,*,promote=True):
    model=copy.deepcopy(native);norms=attentions=0
    for m in model.modules():
        if m.__class__.__name__=='RMSNorm':m.forward=types.MethodType(norm_forward,m);norms+=1
        if m.__class__.__name__=='Attention':
            assert not any((m.time_dependent_qkv,m.adaptive_temperature,m.position_bias,m.head_specialization))
            m.forward=types.MethodType(attention_forward,m);attentions+=1
    assert norms>0 and attentions==3
    if promote:
        model.double();model.t_embedder.mlp.register_forward_pre_hook(lambda m,args:(args[0].to(m[0].weight.dtype),))
    return model.eval().requires_grad_(False)

def input_action(model,image,t,frame):
    x=image.detach().clone().requires_grad_(True)
    output=model(x,torch.full((1,),t,dtype=x.dtype),torch.zeros(1,dtype=torch.long))
    def action(q):
        grad=torch.autograd.grad(output,x,grad_outputs=frame.lift(q),retain_graph=True,create_graph=False)[0]
        return frame.encode(grad)[0].detach()
    return output.detach(),action

def sketch(actions,dimension,steps=24):
    assert dimension>=steps
    generator=torch.Generator().manual_seed(0);q=torch.randn(dimension,generator=generator,dtype=torch.float64)
    basis=[];products=[[] for _ in actions];breakdowns=[]
    for i in range(steps):
        for attempt in range(5):
            if basis:
                mat=torch.stack(basis,1)
                for _ in range(2):q=q-mat@(mat.T@q)
            length=torch.linalg.vector_norm(q)
            if float(length)>=1e-12:break
            breakdowns.append(dict(step=i,attempt=attempt));q=torch.randn(dimension,generator=generator,dtype=torch.float64)
        else:raise RuntimeError('Krylov orthogonalization breakdown')
        q=q/length;basis.append(q)
        for j,action in enumerate(actions):products[j].append(action(q))
        q=products[i%len(actions)][-1]
    basis=torch.stack(basis,1);identity=torch.eye(steps,dtype=basis.dtype)
    orth=float((basis.T@basis-identity).abs().max());assert orth<1e-11
    results=[]
    for action,columns in zip(actions,products):
        transpose=basis.T@torch.stack(columns,1);symmetric=(transpose+transpose.T)/2
        eigen,vectors=torch.linalg.eigh(symmetric);direction=basis@vectors[:,0];rayleigh=float(direction@action(direction))
        assert abs(rayleigh-float(eigen[0]))<1e-10
        norm=float(torch.linalg.matrix_norm(transpose));skew=(transpose-transpose.T)/2
        results.append(dict(compressed_transpose=transpose,eigenvalues=eigen,witness=direction,
            min_ritz=float(eigen[0]),rayleigh=rayleigh,negative_hinge=max(0.,-rayleigh),
            negative_symmetric_energy=float(eigen.clamp_max(0).square().sum()),
            skew_fraction=float(torch.linalg.matrix_norm(skew))/(norm+1e-30),
            compressed_operator_norm=float(torch.linalg.svdvals(transpose)[0]),orthogonality_max=orth))
    return basis,results,breakdowns

@torch.no_grad()
def finite_compression(model,image,t,frame,basis,h=1e-4):
    directions=frame.lift(basis.T);x=torch.cat((image+h*directions,image-h*directions),0)
    output=model(x,torch.full((len(x),),t,dtype=x.dtype),torch.zeros(len(x),dtype=torch.long));k=basis.shape[1]
    changes=frame.encode((output[:k]-output[k:])/(2*h))
    return changes@basis

@torch.no_grad()
def finite_rayleigh(model,image,t,frame,witness,h):
    v=frame.lift(witness);x=torch.cat((image+h*v,image-h*v),0)
    output=model(x,torch.full((2,),t,dtype=x.dtype),torch.zeros(2,dtype=torch.long))
    return float(witness@frame.encode(((output[0]-output[1])/(2*h))[None])[0])

def fixtures():
    g=torch.Generator().manual_seed(0);u=torch.linalg.qr(torch.randn(48,32,generator=g,dtype=torch.float64)).Q;frame=PatchFrame(u)
    q=torch.randn(frame.dimension,generator=g,dtype=torch.float64)
    torch.testing.assert_close(frame.encode(frame.lift(q))[0],q,rtol=1e-11,atol=1e-11)
    torch.testing.assert_close(frame.lift(q).square().sum(),q.square().sum(),rtol=1e-11,atol=1e-11)
    atoms=torch.randn(11,6,generator=g,dtype=torch.float64);obs=torch.linalg.qr(torch.randn(6,3,generator=g,dtype=torch.float64)).Q
    maximum=0.
    for t in (13/128,64/128,115/128):
        for z in torch.randn(5,6,generator=g,dtype=torch.float64):
            def mean(x):return torch.softmax(-((x@obs)[None]-t*(atoms@obs)).square().sum(1)/(2*(1-t)**2),0)@atoms
            w=torch.softmax(-((z@obs)[None]-t*(atoms@obs)).square().sum(1)/(2*(1-t)**2),0);center=atoms-w@atoms;cov=center.T@(w[:,None]*center)
            actual=torch.autograd.functional.jacobian(mean,z);expected=t/(1-t)**2*cov@obs@obs.T
            torch.testing.assert_close(actual,expected,rtol=3e-11,atol=3e-11);maximum=max(maximum,float((actual-expected).abs().max()))
            assert float(torch.linalg.eigvalsh((obs.T@actual@obs+obs.T@actual.T@obs)/2)[0])>=-1e-11
    full=torch.tensor([[1.,0.],[.8,0.]],dtype=torch.float64);wrong=float(torch.linalg.eigvalsh((full+full.T)/2)[0]);assert wrong<-.1 and full[0,0]>0
    a=torch.randn(48,48,generator=g,dtype=torch.float64)/8;b=torch.randn(48,48,generator=g,dtype=torch.float64)/8
    basis,result,_=sketch([lambda q:a.T@q,lambda q:b.T@q],48,steps=12)
    for matrix,row in zip((a,b),result):torch.testing.assert_close(row['compressed_transpose'],basis.T@matrix.T@basis,rtol=3e-12,atol=3e-12)
    return dict(status='PASS',conditional_mean_cases=15,max_identity_error=maximum,full_space_PSD_counterexample_minimum=wrong,patch_isometry=True,known_matrix_joint_sketch=True)
