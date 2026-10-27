"""Expose the legacy BxN versus BxD condition ambiguity at N=D=64."""
import argparse,json,sys
from pathlib import Path
import torch

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
sys.path.insert(0,str(root/'source'));from image_model import build,import_native
_,ffn=import_native(root/'source/jit');torch.set_num_threads(1);torch.manual_seed(0)
layer=ffn.ODELayer(64,64,orders=2,t_embed_dim=64);x=torch.randn(2,64,64);c=torch.randn(2,64)
y=layer(x,c);s=.2+.8*torch.sigmoid(c[:,:,None]/10)
term=x;expected=x
for i in range(2):term=layer.proj(term)*s/(i+1);expected=expected+term
torch.testing.assert_close(y,expected,atol=0,rtol=0)
params=[*layer.t_func.parameters(),*layer.t_from_embed.parameters()]
grad=torch.autograd.grad(y.square().mean(),params,allow_unused=True)
assert all(v is None for v in grad)
adapter=build(root/'source/jit','ode_k2').blocks[0].mlp
torch.testing.assert_close(adapter(x,c),adapter(x,3*c+1),atol=0,rtol=0)
result=dict(status='PASS',tokens=64,hidden_width=64,
    legacy_condition_treated_as_tokenwise_scalars=True,both_legacy_controller_MLPs_unused=True,
    direct_condition_gate_formula_bitwise=True,professor_adapter_avoids_ambiguity=True,
    scope='Current source shape semantics only; no historical training attribution and no legacy source modification.')
(root/'shape-semantics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
