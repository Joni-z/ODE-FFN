"""Exact target cumulant ranks and narrowly scoped single-FFN derivative checks."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
root=args.root.resolve();sys.path.insert(0,str(root/'source'))
import joint_study as study
study.setup(root);torch.set_num_threads(1)
target=study.distribution(root);signs=target.signs.astype(np.int64);d=16
eye=np.eye(d,dtype=np.int64)
four=np.einsum('bi,bj,bk,bl->ijkl',signs,signs,signs,signs,optimize=True)
assert np.all(four%32==0)
code=(four//32-np.einsum('ij,kl->ijkl',eye,eye)
      -np.einsum('ik,jl->ijkl',eye,eye)-np.einsum('il,jk->ijkl',eye,eye))
product=np.zeros_like(code)
for j in range(d):product[j,j,j,j]=-2


def modular_rank(matrix: np.ndarray, prime: int=1000003) -> int:
    """An exact finite-field lower bound on the rational matrix rank."""
    a=matrix.copy()%prime;rank=0
    for column in range(a.shape[1]):
        choices=np.flatnonzero(a[rank:,column])
        if not len(choices):continue
        pivot=rank+int(choices[0]);a[[rank,pivot]]=a[[pivot,rank]]
        a[rank]=a[rank]*pow(int(a[rank,column]),-1,prime)%prime
        factors=a[rank+1:,column].copy()
        a[rank+1:]=(a[rank+1:]-factors[:,None]*a[rank])%prime
        rank+=1
        if rank==len(a):break
    return rank


groups=np.array([i^j for i in range(d) for j in range(d)])
matrix=code.reshape(d*d,d*d)
assert not matrix[groups[:,None]!=groups[None,:]].any()
blocks=[]
for group in range(d):
    idx=np.flatnonzero(groups==group);block=matrix[np.ix_(idx,idx)]
    rank=modular_rank(block)
    eigenvalues=np.linalg.eigvalsh(block.astype(float))
    if group==0:
        np.testing.assert_array_equal(block,-2*np.eye(d,dtype=int));assert rank==16
    else:
        np.testing.assert_allclose(np.sort(eigenvalues),np.r_[np.full(7,-2.),np.zeros(8),14.],atol=1e-12)
        assert rank==8
    blocks.append(dict(xor_group=group,rank=rank))
rank_code=modular_rank(matrix);rank_product=modular_rank(product.reshape(d*d,d*d))
assert rank_code==sum(b['rank'] for b in blocks)==136 and rank_product==16

# Native scalar-GMM third derivatives at the origin, with all fixed means retained.
mu=torch.tensor(target.locations,dtype=torch.float64);means=torch.tensor(target.means,dtype=torch.float64)
t=.3;variance=(1-t)**2+.04*t*t;alpha=(.04*t-(1-t))/variance;beta=(1-t)/variance;kappa=t/variance
def mean_velocity(q):
    probability=torch.softmax(kappa*(q@means.T)-.5*t*t/variance*means.square().sum(1),dim=0)
    return alpha*q+beta*(probability@means)
q=torch.zeros(d,dtype=torch.float64,requires_grad=True)
first=torch.autograd.grad(mean_velocity(q)[0],q,create_graph=True)[0]
second=torch.autograd.grad(first[1],q,create_graph=True)[0]
third=torch.autograd.grad(second[2],q)[0][3]
expected=beta*kappa**3*mu[:4].prod()
torch.testing.assert_close(third,expected,rtol=1e-10,atol=1e-11)

# Validate the plain and multiplicative FFN rank bounds on small independent fixtures.
torch.manual_seed(0)
width=3;dimension=4
a=torch.randn(width,dimension,dtype=torch.float64)*.4
b=torch.randn(width,dimension,dtype=torch.float64)*.4
out=torch.randn(dimension,width,dtype=torch.float64)
bias=torch.linspace(-.3,.4,width,dtype=torch.float64)
plain=lambda x:out@((a@x+bias)*torch.rsqrt(1+1.2*(a@x+bias).square()))
swiglu=lambda x:out@(torch.nn.functional.silu(a@x+bias)*(b@x+.2))
point=torch.linspace(-.2,.3,dimension,dtype=torch.float64)
third_tensor=lambda fn,x:torch.func.jacrev(torch.func.jacrev(torch.func.jacrev(fn)))(x)
fixtures=[]
for name,fn,bound in [('plain',plain,width),('swiglu',swiglu,2*width)]:
    jet=third_tensor(fn,point).detach().numpy().reshape(dimension**2,dimension**2)
    rank=int(np.linalg.matrix_rank(jet,tol=1e-10));assert rank<=bound
    fixtures.append(dict(family=name,hidden=width,rank=rank,bound=bound))

# A scalar clock is NOT generally restricted to low cubic-jet rank.
from synthetic.models import OriginalODELayer
layer=OriginalODELayer(dimension,2*dimension,orders=2).double()
with torch.no_grad():
    layer.proj.weight.copy_(torch.eye(dimension,dtype=torch.float64))
    layer.t_func[0].weight.copy_(torch.cat([torch.eye(dimension),-torch.eye(dimension)]).double())
    layer.t_func[0].bias.zero_();layer.t_func[-1].weight.fill_(1.);layer.t_func[-1].bias.zero_()
jet=third_tensor(lambda x:layer(x[None])[0],torch.zeros(dimension,dtype=torch.float64)).detach().numpy()
identity=np.eye(dimension)
expected_jet=.032*(np.einsum('ij,kl->ijkl',identity,identity)
    +np.einsum('ik,jl->ijkl',identity,identity)+np.einsum('il,jk->ijkl',identity,identity))
np.testing.assert_allclose(jet,expected_jet,rtol=1e-10,atol=1e-11)
scalar_rank=int(np.linalg.matrix_rank(jet.reshape(dimension**2,dimension**2),tol=1e-10))
assert scalar_rank==dimension*(dimension+1)//2

folder=root/'interaction';folder.mkdir(exist_ok=False)
np.save(folder/'product-cumulant.npy',product);np.save(folder/'code-cumulant.npy',code)
receipt=dict(status='PASS',target_product_rank=rank_product,target_code_rank=rank_code,
    exact_modular_prime=1000003,block_ranks=blocks,
    mixed_third_derivative=float(third),expected_mixed_third_derivative=float(expected),
    generic_fixtures=fixtures,scalar_ode_fixture_dimension=dimension,scalar_ode_fixture_rank=scalar_rank,
    full_16d_scalar_clock_constructive_rank=136,
    scope='target third derivatives and single-FFN fixtures; not a complete-network lower bound')
study.write(folder/'receipt.json',receipt)
print(json.dumps(receipt),flush=True)
