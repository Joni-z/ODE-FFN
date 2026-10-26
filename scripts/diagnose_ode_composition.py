"""Measure intermediate SiLU derivatives and clock saturation on validation inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.models import OriginalODEFFN,VelocityNet
from synthetic.train import load_distribution,load_targets,sha256


def diagnose(study: Path,data_root: Path,output: Path) -> dict:
    torch.set_num_threads(1)
    assert json.loads((study/'status.json').read_text())['state']=='COMPLETE'
    plan=json.loads((study/'provenance.json').read_text())['plan']
    manifest=json.loads((data_root/'manifest.json').read_text())
    rows=[]
    for dataset in plan['datasets']:
        dist=load_distribution(manifest['datasets'][dataset])
        targets=load_targets(data_root/dataset/'validation.npz',dist.dim)[:1024]
        sources=np.random.default_rng(0).standard_normal(targets.shape).astype(np.float32)
        for arm in plan['arms']:
            if arm['model']=='swiglu':continue
            cfg_path=study/'configs'/f'{arm["label"]}.json'
            cfg=json.loads(cfg_path.read_text())
            checkpoint=study/'runs'/cfg['experiment_id']/dataset/arm['model']/'seed000'/f'checkpoints/step_{plan["steps"]:07d}.pt'
            model=VelocityNet(dist.dim,arm['model'],**cfg['model_options']).eval()
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True)
            for t in (.1,.3,.5,.7,.9):
                handles=[]
                for name,module in model.named_modules():
                    if not isinstance(module,OriginalODEFFN):continue
                    def hook(m,args,result,block=name):
                        u=args[0].detach()
                        logistic=torch.sigmoid(u)
                        derivative=logistic*(1+u*(1-logistic))
                        rows.append(dict(dataset=dataset,label=arm['label'],time=t,block=block,
                            quantity='intermediate_silu',negative_derivative_fraction=float((derivative<0).float().mean()),
                            weak_derivative_fraction=float((derivative.abs()<.05).float().mean()),
                            derivative_quantiles=torch.quantile(derivative.flatten(),torch.tensor([0.,.1,.5,.9,1.])).tolist()))
                    handles.append(module.act.register_forward_hook(hook))
                for name,layer in model.ode_layers():
                    def gate_hook(m,args,block=name):
                        s=m.step_size(args[0]).detach()
                        p=(s-m.shift)/m.scale
                        rows.append(dict(dataset=dataset,label=arm['label'],time=t,block=block,
                            quantity='clock',s_mean=float(s.mean()),s_std=float(s.std(unbiased=False)),
                            low_saturation_fraction=float((p<.01).float().mean()),
                            high_saturation_fraction=float((p>.99).float().mean())))
                    handles.append(layer.register_forward_pre_hook(gate_hook))
                with torch.no_grad():model(torch.from_numpy((1-t)*sources+t*targets),torch.full((len(targets),1),t))
                for handle in handles:handle.remove()
    result=dict(state='COMPLETE',split='validation',points=1024,rows=rows,
        source_sha256=sha256(Path(__file__)),manifest_sha256=sha256(data_root/'manifest.json'))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','data-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=diagnose(args.study,args.data_root,args.output)
    print(json.dumps(dict(state=result['state'],rows=len(result['rows']))))
