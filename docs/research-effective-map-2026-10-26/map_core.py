"""The same SiLU core learned through P or direct effective matrices."""
import torch
from torch import nn
from torch.nn import functional as F


def effective(p: torch.Tensor) -> torch.Tensor:
    return torch.eye(p.shape[0],dtype=p.dtype,device=p.device)+p+.5*(p@p)


class MapCore(nn.Module):
    def __init__(self,first: torch.Tensor,second: torch.Tensor,coordinates: str):
        super().__init__()
        if coordinates not in ('p_map','w_map'):
            raise ValueError('unsupported matrix coordinates')
        self.coordinates=coordinates
        self.matrix1=nn.Parameter(first.detach().clone())
        self.matrix2=nn.Parameter(second.detach().clone())

    def effective_matrices(self) -> tuple[torch.Tensor,torch.Tensor]:
        if self.coordinates=='p_map':
            return effective(self.matrix1),effective(self.matrix2)
        return self.matrix1,self.matrix2

    def forward(self,z: torch.Tensor) -> torch.Tensor:
        first,second=self.effective_matrices()
        return F.linear(F.silu(F.linear(z,first)),second)


def direct_copy(module: MapCore) -> MapCore:
    first,second=module.effective_matrices()
    return MapCore(first,second,'w_map')


def sequential(module: MapCore,z: torch.Tensor) -> torch.Tensor:
    """Independent original polynomial action; no preformed P-squared matrix."""
    if module.coordinates=='w_map':
        return F.linear(F.silu(F.linear(z,module.matrix1)),module.matrix2)
    h=z
    for p,activate in [(module.matrix1,True),(module.matrix2,False)]:
        first=F.linear(h,p)
        second=F.linear(first,p)/2
        h=(h+first)+second
        if activate:h=F.silu(h)
    return h


def input_jacobian(module: MapCore,z: torch.Tensor) -> torch.Tensor:
    first,second=module.effective_matrices()
    pre=F.linear(z,first)
    sigmoid=torch.sigmoid(pre)
    derivative=sigmoid+pre*sigmoid*(1-sigmoid)
    return (second[None]*derivative[:,None,:])@first
