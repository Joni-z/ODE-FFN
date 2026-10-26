"""Function-matched independent copies and orthogonal depth-gradient algebra."""
from __future__ import annotations
import copy
import math
import torch
from torch import nn


def independent_copies(shared):
    if shared.layout!='shared' or shared.raw.mode!='shared' or len(shared.raw.raw.blocks)!=1:
        raise ValueError('expected the actual three-visit shared predictor')
    cloned=copy.deepcopy(shared)
    block=cloned.raw.raw.blocks[0]
    cloned.raw.raw.blocks=nn.ModuleList([copy.deepcopy(block) for _ in range(3)])
    cloned.raw.mode='cloned';cloned.layout='cloned'
    return cloned


def depth_basis(like):
    return like.new_tensor([[1/math.sqrt(3)]*3,[-1/math.sqrt(2),0,1/math.sqrt(2)],
                            [1/math.sqrt(6),-2/math.sqrt(6),1/math.sqrt(6)]])


def coefficients(gradient):
    if gradient.ndim!=2 or gradient.shape[0]!=3:
        raise ValueError('expected three independent depth gradients')
    return depth_basis(gradient)@gradient


def describe(gradient):
    coeff=coefficients(gradient)
    energy=coeff.square().sum(-1)
    total=gradient.square().sum()
    torch.testing.assert_close(energy.sum(),total,rtol=1e-11,atol=1e-14)
    gram=gradient@gradient.T
    total_value=float(total)
    cosines=[]
    for i,j in ((0,1),(0,2),(1,2)):
        denom=float((gram[i,i]*gram[j,j]).sqrt())
        cosines.append(float(gram[i,j])/denom if denom>0 else None)
    return dict(E0=float(energy[0]),E1=float(energy[1]),E2=float(energy[2]),total=total_value,
        retained=float(energy[0]/total) if total_value>0 else None,
        linear_retained=float((energy[0]+energy[1])/total) if total_value>0 else None,
        linear_share_of_lost=float(energy[1]/energy[1:].sum()) if float(energy[1:].sum())>0 else None,
        cross_energy=float(gradient.sum(0).square().sum()-total),
        cosine_01=cosines[0],cosine_02=cosines[1],cosine_12=cosines[2])


def projected(gradient,modes):
    basis=depth_basis(gradient)
    coeff=coefficients(gradient)
    return basis[list(modes)].T@coeff[list(modes)]


def transferred_alignment(gradient_a,gradient_b,modes):
    direction=projected(gradient_a,modes)
    denom=float(direction.norm()*gradient_b.norm())
    return float((direction*gradient_b).sum())/denom if denom>0 else None


def block_coordinates(model,family):
    names=[];sizes=[];groups={};cursor=0
    for name,parameter in model.raw.raw.blocks[0].named_parameters():
        if name.startswith('norm.'):group='norm'
        elif name.startswith('condition.'):group='condition'
        elif family=='ode_k2' and '.proj.' in name:group='P'
        elif family=='ode_k2' and '.t_func.' in name:group='clock'
        elif family.startswith('swiglu') and name.startswith('ffn.w12.'):group='ffn_input'
        elif family.startswith('swiglu') and name.startswith('ffn.w3.'):group='ffn_output'
        else:raise ValueError('unclassified block parameter '+name)
        size=parameter.numel();names.append(name);sizes.append(size)
        groups.setdefault(group,[]).extend(range(cursor,cursor+size));cursor+=size
    assert sorted(i for v in groups.values() for i in v)==list(range(cursor))
    groups['FFN']=sorted(i for key,value in groups.items() if key in ('P','clock','ffn_input','ffn_output') for i in value)
    groups['all']=list(range(cursor))
    return names,sizes,groups


def loss_gradients(model,x,t,target):
    parameters=dict(model.named_parameters())
    prediction=model(x,t)
    loss=(prediction-target).square().mean()
    values=torch.autograd.grad(loss,tuple(parameters.values()))
    gradients=dict(zip(parameters,values))
    count=1 if model.layout=='shared' else 3
    names=[name for name,_ in model.raw.raw.blocks[0].named_parameters()]
    block=torch.stack([torch.cat([gradients[f'raw.raw.blocks.{i}.'+name].reshape(-1) for name in names])
                       for i in range(count)])
    common={name:value for name,value in gradients.items() if '.blocks.' not in name}
    return prediction.detach(),float(loss.detach()),block.detach(),{k:v.detach() for k,v in common.items()}


def copy_block_vectors(model,vectors):
    if model.layout!='cloned' or vectors.shape[0]!=3:raise ValueError('independent copies required')
    with torch.no_grad():
        for block,vector in zip(model.raw.raw.blocks,vectors):
            cursor=0
            for parameter in block.parameters():
                size=parameter.numel();parameter.copy_(vector[cursor:cursor+size].reshape_as(parameter));cursor+=size
            if cursor!=vector.numel():raise ValueError('block vector length mismatch')


def block_vectors(model):
    return torch.stack([torch.cat([p.detach().reshape(-1) for p in block.parameters()]) for block in model.raw.raw.blocks])
