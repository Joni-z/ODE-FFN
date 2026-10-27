"""Seeded real-kernel witness for the declared image environments."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch

ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('training','metrics'),required=True);args=ap.parse_args()
torch.set_num_threads(1);torch.manual_seed(0)
if args.mode=='training':
    parent=Path('/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle16-feasibility-v2/source')
    sys.path.insert(0,str(parent));from image_model import build
    model=build(parent/'jit','ode_k2');x=torch.randn(2,3,32,32);t=torch.full((2,),.4)
    with torch.no_grad():
        model.final_layer.linear.weight.fill_(.01)
        for b in model.blocks:b.adaLN_modulation.linear.bias[320:384]=.1
    y=model(x,t,torch.zeros(2,dtype=torch.long));y.square().mean().backward()
    norms=[float(b.mlp.ode1.proj.weight.grad.norm()) for b in model.blocks]
    assert y.shape==(2,3,32,32) and all(n>0 for n in norms) and torch.isfinite(y).all()
    result=dict(mode='training',device='cpu',shape=list(y.shape),P_gradient_norms=norms)
else:
    cache=Path('/Users/mr.z/ode-workspace-prep/image-bridge-inception');cache.mkdir(exist_ok=True)
    torch.hub.set_dir(str(cache))
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    assert torch.backends.mps.is_available()
    model=FeatureExtractorInceptionV3('inception-v3-compat',['2048']).eval().requires_grad_(False).to('mps')
    x=torch.randint(256,(2,3,32,32),dtype=torch.uint8)
    with torch.no_grad():y=model(x.to('mps'))[0].cpu()
    torch.mps.synchronize();assert y.shape==(2,2048) and torch.isfinite(y).all() and torch.count_nonzero(y)>0
    weights=list((cache/'checkpoints').glob('*.pth'));assert len(weights)==1
    with weights[0].open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    result=dict(mode='metrics',device='mps',shape=list(y.shape),feature_sum=float(y.sum()),
        weights=str(weights[0]),weights_sha256=digest,weights_bytes=weights[0].stat().st_size)
print('WITNESS_PASS '+json.dumps(result),flush=True)
