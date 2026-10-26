"""Check closed scalar-K2 sensitivities on actual learned layer parameters."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.conservative_readout import build
from synthetic.train import restore_model


def verify(parent,new,output):
    torch.set_num_threads(1)
    rows=[];bindings={}
    for root,trained in ((parent,'gradient'),(new,'directional')):
        config=json.loads((root/'study.json').read_text())
        for dataset in config['datasets']:
            checkpoint=root/'runs'/dataset/('ode_k2_'+trained)/'step_0005000.pt'
            bindings[str(checkpoint)]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            dim=16 if dataset=='rotated_product_bimodal_16d' else 2
            model=build(dim,'ode_k2','direct').double()
            restore_model(model,torch.load(checkpoint,map_location='cpu',weights_only=False))
            for name,layer in model.ode_layers():
                assert layer.orders==2 and layer.gate_source=='input' and layer.spectral_cap is None
                assert layer.gate_transform=='sigmoid' and layer.clock_curvature==1.
                width=layer.proj.weight.shape[0]
                coordinates=torch.linspace(-1.,1.,width,dtype=torch.float64)
                h=torch.stack((coordinates,coordinates.sin(),coordinates.cos())).requires_grad_(True)
                r=torch.stack((coordinates.roll(1),coordinates.roll(2),coordinates.roll(3)))
                q=torch.stack((coordinates.roll(4),coordinates.roll(5),coordinates.roll(6)))
                first,last=layer.t_func[0],layer.t_func[2]
                pre=F.linear(h,first.weight,first.bias)
                logit=F.linear(F.silu(pre),last.weight,last.bias)
                probability=torch.sigmoid(logit/layer.tau)
                a=probability*layer.scale+layer.shift
                sigmoid=torch.sigmoid(pre)
                phi_prime=sigmoid*(1+pre*(1-sigmoid))
                da=layer.scale/layer.tau*probability*(1-probability)*F.linear(phi_prime*last.weight,first.weight.T)
                p=layer.proj.weight
                ph=F.linear(h,p);p2h=F.linear(ph,p);u=ph+a*p2h
                pr=F.linear(r,p);p2r=F.linear(pr,p)
                forward=r+a*pr+.5*a.square()*p2r+u*(da*r).sum(-1,keepdim=True)
                pq=F.linear(q,p.T);p2q=F.linear(pq,p.T)
                adjoint=q+a*pq+.5*a.square()*p2q+da*(u*q).sum(-1,keepdim=True)
                value=layer(h)
                reference_adjoint=torch.autograd.grad(value,h,grad_outputs=q)[0]
                _,reference_forward=torch.autograd.functional.jvp(layer,h,r)
                torch.testing.assert_close(forward,reference_forward,rtol=1e-11,atol=1e-12)
                torch.testing.assert_close(adjoint,reference_adjoint,rtol=1e-11,atol=1e-12)
                torch.testing.assert_close((forward*q).sum(),(r*adjoint).sum(),rtol=1e-11,atol=1e-12)
                rows.append(dict(dataset=dataset,trained=trained,layer=name,
                    forward_error=float((forward-reference_forward).abs().max().detach()),
                    adjoint_error=float((adjoint-reference_adjoint).abs().max().detach())))
    assert len(rows)==36
    result=dict(status='PASS',layer_parameter_sets=36,probe_inputs_per_layer=3,
        scope='actual learned weights; deterministic algebra probes, not task-quality outcomes',
        maximum_forward_error=max(r['forward_error'] for r in rows),maximum_adjoint_error=max(r['adjoint_error'] for r in rows),
        rows=rows,checkpoint_sha256=bindings,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','checkpoint_sha256')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--new',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.parent.resolve(),a.new.resolve(),a.output.resolve())
