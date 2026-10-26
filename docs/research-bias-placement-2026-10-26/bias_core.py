"""A single same-sized offset before the nonlinearity or after the readout."""
import torch
from torch import nn
from torch.nn import functional as F


class BiasCore(nn.Module):
    def __init__(self,first: torch.Tensor,second: torch.Tensor,placement: str):
        super().__init__()
        if placement not in ('hidden_bias','output_bias'):raise ValueError('unsupported offset placement')
        self.placement=placement
        self.matrix1=nn.Parameter(first.detach().clone())
        self.matrix2=nn.Parameter(second.detach().clone())
        self.offset=nn.Parameter(torch.zeros(first.shape[0],dtype=first.dtype,device=first.device))

    def forward(self,z: torch.Tensor) -> torch.Tensor:
        hidden=F.linear(z,self.matrix1)
        if self.placement=='hidden_bias':hidden=hidden+self.offset
        output=F.linear(F.silu(hidden),self.matrix2)
        return output+self.offset if self.placement=='output_bias' else output


def input_jacobian(module: BiasCore,z: torch.Tensor) -> torch.Tensor:
    hidden=F.linear(z,module.matrix1)
    if module.placement=='hidden_bias':hidden=hidden+module.offset
    s=torch.sigmoid(hidden);derivative=s+hidden*s*(1-s)
    return (module.matrix2[None]*derivative[:,None,:])@module.matrix1
