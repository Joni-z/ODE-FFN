"""Literal input-clock ODE baselines in the existing JiT image backbone."""
from __future__ import annotations
import sys
from pathlib import Path
import torch


def import_native(source):
    source=Path(source).resolve()
    if str(source) not in sys.path:sys.path.insert(0,str(source))
    from util.compile_control import set_compile_enabled
    set_compile_enabled(False)
    import model_jit,ffn_blocks
    assert Path(model_jit.__file__).resolve().parent==source
    return model_jit,ffn_blocks


def build(source,family):
    native,ffn=import_native(source)
    class InputClockODEFFN(ffn.ODEFFN):
        def forward(self,x,cond=None,return_aux=False):
            # JiT still uses conditioning for attention and AdaLN. Only the
            # professor's local controller must consume its own feature input.
            return super().forward(x,None,return_aux=return_aux)
    torch.manual_seed(0)
    model=native.JiT(input_size=32,patch_size=4,in_channels=3,hidden_size=64,depth=3,
        num_heads=4,mlp_ratio=2.,num_classes=1,bottleneck_dim=32,in_context_len=0,
        ffn_type='swiglu',attn_drop=0.,proj_drop=0.)
    torch.manual_seed(0)
    for block in model.blocks:
        if family in ('ode_k1','ode_k2'):
            block.mlp=InputClockODEFFN(dim=64,hidden_dim=128,ode_hidden_features=64,
                orders=1 if family=='ode_k1' else 2,drop=0.,t_embed_dim=None)
        elif family in ('swiglu86','swiglu129'):
            block.mlp=ffn.SwiGLUFFN(dim=64,hidden_dim=129 if family=='swiglu86' else 194,drop=0.)
        else:raise ValueError(family)
    return model


def ffn_cost(model):
    count=0;hooks=[]
    def hook(layer,inputs,out):
        nonlocal count
        count+=out.numel()*layer.in_features
    for block in model.blocks:
        for layer in block.mlp.modules():
            if isinstance(layer,torch.nn.Linear):hooks.append(layer.register_forward_hook(hook))
    with torch.no_grad():
        model.blocks[0].mlp(torch.zeros(1,1,64),torch.zeros(1,64))
    for hook in hooks:hook.remove()
    # Only block0 is executed above. All three FFNs have identical shapes.
    return dict(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        ffn_parameters_per_block=sum(p.numel() for p in model.blocks[0].mlp.parameters()),
        dense_FFN_forward_MACs_per_token_per_block=count,
        scope='Dense FFN only; attention/backbone are identical. Pointwise work excluded.')
