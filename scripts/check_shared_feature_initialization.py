"""Audit shared-feature implementation and paired initialization; no training study.

The default compatibility check makes three deterministic optimizer updates on
synthetic probe batches only. It does not evaluate an architecture's quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.check_rate_order_reference import check as check_default_reference
from scripts.run_architecture_study import macs
from synthetic.models import VelocityNet


def measured_macs(model,dim,dtype):
    linear=F.linear
    calls=[]
    def measured(x,weight,bias=None):
        calls.append(dict(input_shape=list(x.shape),weight_shape=list(weight.shape),
                          macs=(x.numel()//x.shape[-1])*weight.numel()))
        return linear(x,weight,bias)
    with patch.object(F,'linear',measured),torch.no_grad():
        model(torch.zeros(1,dim,dtype=dtype),torch.zeros(1,1,dtype=dtype))
    return sum(row['macs'] for row in calls),calls


def check(reference_file):
    torch.set_num_threads(1)
    previous=torch.get_default_dtype()
    rows=[]
    try:
        for dtype in (torch.float32,torch.float64):
            torch.set_default_dtype(dtype)
            for dim in (2,16):
                generator=torch.Generator().manual_seed(0)
                head=torch.randn(dim,64,generator=generator)*.2
                x=torch.randn(31,dim,generator=generator)
                t=torch.rand(31,1,generator=generator)
                models={};rngs={}
                for mode,hidden in (('independent',32),('shared_projection',64),('paired_untied',64)):
                    torch.manual_seed(0)
                    model=VelocityNet(dim,'ode_vector',orders=1,gate_hidden=hidden,
                        gate_reference_hidden=64,ode_solve_init='constant_paired',ode_feature_mode=mode)
                    rngs[mode]=torch.get_rng_state().clone()
                    with torch.no_grad():model.output.weight.copy_(head)
                    models[mode]=model
                reference=models['independent'];reference_state=reference.state_dict()
                y=reference(x,t).detach()
                for mode,model in models.items():
                    torch.testing.assert_close(rngs[mode],rngs['independent'],atol=0,rtol=0)
                    common_keys=[]
                    for name,value in model.state_dict().items():
                        if '.ffn.' not in name or name.endswith('.proj.weight'):
                            torch.testing.assert_close(value,reference_state[name],atol=0,rtol=0)
                            common_keys.append(name)
                    for name,value in model.named_buffers():
                        torch.testing.assert_close(value,dict(reference.named_buffers())[name],atol=0,rtol=0)
                    value=model(x,t)
                    torch.testing.assert_close(value,y,atol=0,rtol=0)
                    measured,calls=measured_macs(model,dim,dtype)
                    counted=macs(model,dim)
                    assert measured==counted
                    value.square().mean().backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                    layers=[]
                    for name,layer in model.ode_layers():
                        if mode!='independent':
                            torch.testing.assert_close(layer.t_func[0].weight,layer.proj.weight,atol=0,rtol=0)
                            assert torch.count_nonzero(layer.t_func[0].bias)==0
                            assert torch.count_nonzero(layer.t_func[0].bias.grad)==0
                            assert (layer.t_func[0].weight is layer.proj.weight)==(mode=='shared_projection')
                        assert torch.count_nonzero(layer.t_func[-1].weight)==0
                        assert torch.count_nonzero(layer.t_func[-1].bias)==0
                        layers.append(dict(name=name,parameters=sum(p.numel() for p in layer.parameters()),
                                           shared_alias=layer.t_func[0].weight is layer.proj.weight))
                    rows.append(dict(dtype=str(dtype),dim=dim,mode=mode,forward_maximum_difference=0.,
                        post_constructor_rng_identical=True,common_state_keys=common_keys,
                        parameters=model.parameter_count(),forward_dense_macs=measured,
                        actual_linear_calls=calls,layers=layers))
                shared,untied=models['shared_projection'],models['paired_untied']
                for name,value in shared.state_dict().items():
                    torch.testing.assert_close(value,untied.state_dict()[name],atol=0,rtol=0)
                for (name,a),(_,b) in zip(shared.ode_layers(),untied.ode_layers()):
                    torch.testing.assert_close(a.proj.weight.grad,b.proj.weight.grad,atol=0,rtol=0)
                    assert torch.count_nonzero(b.t_func[0].weight.grad)==0
                    for pa,pb in zip(a.t_func[-1].parameters(),b.t_func[-1].parameters()):
                        torch.testing.assert_close(pa.grad,pb.grad,atol=0,rtol=0)
                # Both new modes also retain the parent's P/value gradients at Wg=0.
                for (_,a),(_,b) in zip(reference.ode_layers(),shared.ode_layers()):
                    torch.testing.assert_close(a.proj.weight.grad,b.proj.weight.grad,atol=0,rtol=0)
    finally:
        torch.set_default_dtype(previous)
    compatibility=check_default_reference(reference_file)
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return dict(state='PASS',device='cpu',torch=torch.__version__,training_seed=0,
        scope='initialization, algebra, executed dense MACs, and default implementation compatibility; no architecture-quality evidence',
        rows=rows,default_reference=compatibility,reference_file=str(reference_file.resolve()),
        reference_sha256=digest(reference_file),model_sha256=digest(ROOT/'synthetic/models.py'),
        checker_sha256=digest(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=check(args.reference_file)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(state=result['state'],initialization_rows=len(result['rows']),
                         default_reference=result['default_reference'],output=str(args.output))))
