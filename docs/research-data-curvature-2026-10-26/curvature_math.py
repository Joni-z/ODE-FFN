"""Closed derivatives of scalar-clock polynomial solves and SwiGLU modules."""
import torch


def activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s = torch.sigmoid(x)
    return x*s, s+x*s*(1-s), s*(1-s)*(2+x*(1-2*s))


def solve_data(layer: torch.nn.Module, h: torch.Tensor) -> dict:
    p = layer.proj.weight
    w = layer.t_func[0].weight
    r = layer.t_func[-1].weight.reshape(-1)
    pre = w@h+layer.t_func[0].bias
    value, first, second = activation(pre)
    logit = r@value+layer.t_func[-1].bias.squeeze()
    s = torch.sigmoid(logit/layer.tau)
    a = layer.scale*s+layer.shift
    logit_g = w.T@(r*first)
    logit_h = w.T@((r*second)[:,None]*w)
    g = (layer.scale/layer.tau)*s*(1-s)*logit_g
    ha = (layer.scale/layer.tau)*s*(1-s)*logit_h
    ha = ha+(layer.scale/layer.tau**2)*s*(1-s)*(1-2*s)*torch.outer(logit_g,logit_g)
    p2 = p@p
    q = torch.eye(len(h),dtype=h.dtype,device=h.device)+a*p+.5*a*a*p2
    c = p+a*p2
    b = c@h
    jac = q+torch.outer(b,g)
    return dict(value=q@h,jac=jac,q=q,c=c,b=b,g=g,ha=ha,p2h=p2@h,a=a)


def solve_hessian(data: dict, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cu = data['c']@u
    gu = data['g']@u
    cross = cu[:,:,None]*gu[None,None,:]+cu[:,None,:]*gu[None,:,None]
    full = cross+data['p2h'][:,None,None]*torch.outer(gu,gu)[None]
    full = full+data['b'][:,None,None]*(u.T@data['ha']@u)[None]
    return full,cross


def left(matrix: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return (matrix@tensor.reshape(tensor.shape[0],-1)).reshape(matrix.shape[0],*tensor.shape[1:])


def local(module: torch.nn.Module, h: torch.Tensor, u: torch.Tensor, family: str) -> dict:
    if family != 'ode_k2':
        weights = module.w12.weight
        gate,value = (weights@h+module.w12.bias).chunk(2)
        wg,wv = weights.chunk(2,0)
        a,ap,app = activation(gate)
        gu,vu = wg@u,wv@u
        hidden_h = (value*app)[:,None,None]*gu[:,:,None]*gu[:,None,:]
        hidden_h = hidden_h+ap[:,None,None]*(gu[:,:,None]*vu[:,None,:]+vu[:,:,None]*gu[:,None,:])
        jac = module.w3.weight@((value*ap)[:,None]*wg+a[:,None]*wv)
        return dict(value=module.w3.weight@(a*value)+module.w3.bias,
                    jac=jac,hessian=left(module.w3.weight,hidden_h))
    first = solve_data(module.ode1,h)
    y,act1,act2 = activation(first['value'])
    second = solve_data(module.ode2,y)
    j1u = first['jac']@u
    uy = act1[:,None]*j1u
    h1,h1cross = solve_hessian(first,u)
    h2,h2cross = solve_hessian(second,uy)
    middle = left(second['jac'],act2[:,None,None]*j1u[:,:,None]*j1u[:,None,:])
    clock1 = left(second['jac'],act1[:,None,None]*h1)
    full = middle+clock1+h2
    q1u = first['q']@u
    frozen_h = left(second['q'],act2[:,None,None]*q1u[:,:,None]*q1u[:,None,:])
    frozen_j = second['q']@(act1[:,None]*first['q'])
    origin_omission = middle+left(second['jac'],act1[:,None,None]*h1cross)+h2cross
    return dict(value=second['value'],jac=second['jac']@(act1[:,None]*first['jac']),
                hessian=full,middle=middle,clock1=clock1,clock2=h2,
                frozen_hessian=frozen_h,frozen_jac=frozen_j,
                origin_omission=origin_omission,first=first,second=second,act1=act1,act2=act2)


def normalized_rows(matrix: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(matrix,dim=1,keepdim=True)
    return torch.where(norms>1e-15,matrix/norms.clamp_min(1e-15),torch.zeros_like(matrix))


def frame(data: dict, u: torch.Tensor) -> dict:
    first,second,act1 = data['first'],data['second'],data['act1']
    constraints = torch.stack([first['g']@u,(second['g']*act1)@first['q']@u])
    normalized = normalized_rows(constraints)
    _,singular,vh = torch.linalg.svd(normalized,full_matrices=True)
    rank = int((singular>singular.max()*1e-10).sum()) if float(singular.max())>0 else 0
    v = vh[rank:].T
    b1 = second['jac']@(act1*first['b'])
    b2 = second['b']
    output = torch.stack([b1,b2],dim=1)
    output_normalized = normalized_rows(output.T).T
    basis,singular_out,_ = torch.linalg.svd(output_normalized,full_matrices=False)
    output_rank = int((singular_out>singular_out.max()*1e-10).sum()) if float(singular_out.max())>0 else 0
    basis = basis[:,:output_rank]
    return dict(v=v,output_basis=basis,input_constraint_rank=rank,output_removed_rank=output_rank,
                input_constraint_residual=float((normalized@v).abs().max()),
                retained_tangent_singular=torch.linalg.svdvals(u@v).tolist())


def reduced(tensor: torch.Tensor, definition: dict) -> torch.Tensor:
    v,basis = definition['v'],definition['output_basis']
    value = torch.einsum('oij,ia,jb->oab',tensor,v,v)
    return value-left(basis,left(basis.T,value))
