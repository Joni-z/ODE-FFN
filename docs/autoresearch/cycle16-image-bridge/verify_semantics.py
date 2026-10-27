"""A source-bound, non-trained witness for the two different clock inputs."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
root=args.root;sys.path.insert(0,str(root/'source'));from image_model import build,import_native
_,ffn=import_native(root/'source/jit');torch.set_num_threads(1);torch.manual_seed(0)
x=torch.randn(2,5,64);c=torch.randn(2,64)
legacy=ffn.ODEFFN(64,128,t_embed_dim=64,orders=2)
a=legacy(x,c);b=legacy(x,3*c+1)
unused=torch.autograd.grad(a.square().mean(),list(legacy.ode1.t_func.parameters()),allow_unused=True,retain_graph=True)
active=torch.autograd.grad(a.square().mean(),list(legacy.ode1.t_from_embed.parameters()))
assert all(g is None for g in unused) and any(g is not None and torch.count_nonzero(g) for g in active)
adapted=build(root/'source/jit','ode_k2').blocks[0].mlp
p=adapted(x,c);q=adapted(x,3*c+1);torch.testing.assert_close(p,q,atol=0,rtol=0)
input_grads=torch.autograd.grad(p.square().mean(),list(adapted.ode1.t_func.parameters()))
assert any(torch.count_nonzero(g) for g in input_grads)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
result=dict(status='PASS',legacy_local_condition_change_max=float((a-b).detach().abs().max()),
    legacy_input_controller_unused=True,legacy_condition_controller_active=True,
    professor_adapter_local_condition_change_max=float((p-q).detach().abs().max()),
    professor_input_controller_active=True,
    scope='Local FFN only. Whole JiT retains timestep/AdaLN/attention conditioning. No historical checkpoint attribution.',
    source_hashes={str(p.relative_to(root)):sha(p) for p in (root/'source/image_model.py',root/'source/jit/time_condition.py',root/'source/jit/ffn_blocks.py')})
(root/'semantics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
