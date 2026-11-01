"""Compare with the pinned author's code; a tiny fixture is not a benchmark."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
import argparse
import copy
import hashlib
import json
import sys
import tempfile
import types
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import model_jit
from denoiser import Denoiser
from ffn import FreeISRU
from main_jit import get_args_parser
from engine_jit import train_one_epoch
from util import misc


def module(name, text):
    value = types.ModuleType(name)
    sys.modules[name] = value
    exec(compile(text, name, 'exec'), value.__dict__)
    return value


def same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right): same(a, b)
    else:
        assert left == right


def active(model):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'adaLN_modulation' in name or 'final_layer.linear' in name:
                parameter.normal_(0, .02)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    manifest=json.loads((ROOT/'UPSTREAM.json').read_text())
    for name,digest in manifest['imported_files'].items():
        assert hashlib.sha256((a.upstream/name).read_bytes()).hexdigest()==digest, name
    torch.set_num_threads(1)
    torch._dynamo.config.disable = True
    # Only placement is adapted in this in-memory CPU reference. The author's
    # loss, attention, initialization, stepper, EMA and update code are executed.
    utility = (a.upstream/'util/model_util.py').read_text().replace('.cuda()', '')
    module('reference_util', utility)
    core = (a.upstream/'model_jit.py').read_text().replace('from util.model_util import', 'from reference_util import')
    core = core.replace('.cuda()', '.to(query.device)')
    reference_core = module('reference_model', core)
    ref = module('reference_denoiser', (a.upstream/'denoiser.py').read_text().replace('from model_jit import', 'from reference_model import'))
    engine = (a.upstream/'engine_jit.py').read_text().replace('torch.cuda.synchronize()', 'None')
    engine = engine.replace("autocast('cuda', dtype=torch.bfloat16)", "autocast('cuda', dtype=torch.bfloat16, enabled=False)")
    reference_engine = module('reference_engine', engine)
    fixture = dict(depth=2, hidden_size=32, num_heads=4, patch_size=4,
                   bottleneck_dim=16, in_context_len=2, in_context_start=1)
    model_jit.JiT_models['_fixture'] = lambda **kw: model_jit.JiT(**fixture, **kw)
    reference_core.JiT_models['_fixture'] = lambda **kw: reference_core.JiT(**fixture, **kw)
    args = get_args_parser().parse_args(['--model','_fixture','--img_size','32','--class_num','10',
            '--label_drop_prob','1','--cfg','0','--num_sampling_steps','3','--num_workers','0','--device','cpu'])
    args.lr = 1e-4
    torch.manual_seed(0); native = ref.Denoiser(args)
    native_rng = torch.get_rng_state().clone()
    torch.manual_seed(0); baseline = Denoiser(args)
    same(native_rng, torch.get_rng_state()); same(native.state_dict(), baseline.state_dict())
    torch.manual_seed(0); active(native)
    baseline.load_state_dict(native.state_dict())
    x = torch.randn(2,3,32,32,requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    t = torch.tensor([.2,.8]); labels = torch.tensor([1,7])
    first = native.net(x,t,labels); second = baseline.net(y,t,labels)
    same(first,second);first.square().sum().backward();second.square().sum().backward()
    same(x.grad,y.grad)
    for p,q in zip(native.parameters(),baseline.parameters()):
        if p.grad is not None: same(p.grad,q.grad)
    assert baseline.net.blocks[0].mlp.w12.weight.grad.abs().sum() > 0
    for model in (native,baseline):
        model.ema_params1=copy.deepcopy(list(model.parameters()))
        model.ema_params2=copy.deepcopy(list(model.parameters()))
    optimizers = [torch.optim.AdamW(misc.add_weight_decay(m,0),lr=1e-4,betas=(.9,.95)) for m in (native,baseline)]
    pixels = torch.randint(256,(2,3,32,32),dtype=torch.uint8)
    batches = [(pixels,labels)]
    torch.manual_seed(91)
    reference_engine.train_one_epoch(native,native,batches,optimizers[0],torch.device('cpu'),6,args=args)
    torch.manual_seed(91)
    train_one_epoch(baseline,baseline,batches,optimizers[1],torch.device('cpu'),6,args=args)
    same(native.state_dict(),baseline.state_dict());same(optimizers[0].state_dict(),optimizers[1].state_dict())
    same(native.ema_params1,baseline.ema_params1);same(native.ema_params2,baseline.ema_params2)
    for method in ('euler','heun'):
        native.method=baseline.method=method
        torch.manual_seed(92); expected=native.generate(labels)
        torch.manual_seed(92); actual=baseline.generate(labels)
        same(expected,actual)
    # Candidate replacement preserves every non-FFN tensor and the training RNG.
    args.ffn='swiglu';torch.manual_seed(0);cold=Denoiser(args);rng=torch.get_rng_state().clone()
    args.ffn='isru';torch.manual_seed(0);candidate=Denoiser(args)
    same(rng,torch.get_rng_state())
    for name,value in cold.state_dict().items():
        if not (name.startswith('net.blocks.') and '.mlp.' in name):
            same(value,candidate.state_dict()[name])
    assert candidate.net.blocks[0].mlp.matrix1.data_ptr()!=candidate.net.blocks[0].mlp.matrix2.data_ptr()
    active(candidate);candidate.net(x.detach(),t,labels).square().mean().backward()
    assert all(p.grad.abs().sum()>0 for p in candidate.net.blocks[0].mlp.parameters())
    small=FreeISRU(4).double(); probe=torch.randn(2,4,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(small,(probe,),fast_mode=True)
    with tempfile.TemporaryDirectory() as temporary:
        args.ffn='swiglu'
        args.output_dir=temporary
        misc.save_model(args,baseline,optimizers[1],epoch=6,epoch_name='last')
        state=torch.load(Path(temporary)/'checkpoint-last.pth',weights_only=False,map_location='cpu')
        same(state['model'],baseline.state_dict());same(state['optimizer'],optimizers[1].state_dict())
    # Brief real MPS dispatch, isolated from the old frozen training source.
    mps=False
    if torch.backends.mps.is_available():
        candidate=candidate.to('mps');candidate.zero_grad(set_to_none=True)
        loss=candidate(x.detach().to('mps'),labels.to('mps'))
        loss.backward()
        assert torch.isfinite(loss)
        torch.optim.AdamW(candidate.parameters(),lr=1e-4).step()
        torch.mps.synchronize();mps=True
    result=dict(status='PASS',upstream='cbc743a2ada5e9762697da2c83f8c4f8379e8c17',
        CPU_reference='placement-only adaptations for the CUDA-only author code',
        native_state_and_RNG_bitwise=True,active_output_input_and_parameter_gradients_bitwise=True,
        official_epoch_Adam_and_two_EMAs_bitwise=True,official_Euler_and_Heun_bitwise=True,
        candidate_non_FFN_and_RNG_unchanged=True,candidate_gradcheck=True,
        official_checkpoint_roundtrip=True,MPS_forward_backward_optimizer=mps,
        quality_or_FID_claim=False)
    if a.output:
        assert not a.output.exists();a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('UPSTREAM_COMPATIBILITY_PASS')


if __name__=='__main__': main()
