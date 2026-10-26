"""Anchored, validation-only attribution to first/second ODE solve clocks."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import numpy as np
import torch


@contextmanager
def hold_gates(model,values: dict,solve: str):
    restored=[]
    try:
        for name,layer in model.ode_layers():
            if not name.endswith(solve):continue
            previous=layer.__dict__.get('step_size')
            restored.append((layer,previous))
            layer.step_size=lambda x,context=None,key=name:values[key]
        yield
    finally:
        for layer,previous in restored:
            if previous is None:del layer.step_size
            else:layer.step_size=previous


def diagnose(study: Path,data_root: Path,source: Path,buffers: Path,output: Path) -> dict:
    sys.path.insert(0,str(source))
    from synthetic.models import VelocityNet
    from synthetic.train import load_distribution,load_targets,sha256
    from scripts.check_gate_geometry import gate_context,jacobian,oracle_geometry
    torch.set_num_threads(1)
    assert json.loads((study/'status.json').read_text())['state']=='COMPLETE'
    plan=json.loads((study/'provenance.json').read_text())['plan']
    manifest=json.loads((data_root/'manifest.json').read_text())
    frequencies=json.loads(buffers.read_text())['time.frequencies']
    rows=[]
    for dataset in plan['datasets']:
        dist=load_distribution(manifest['datasets'][dataset])
        target=load_targets(data_root/dataset/'validation.npz',dist.dim)[:512].astype(np.float64)
        source_points=np.random.default_rng(0).standard_normal(target.shape)
        for arm in plan['arms']:
            if arm['label'] not in ('scalar_k1','vector_k1'):continue
            config=study/'configs'/f'{arm["label"]}.json'
            cfg=json.loads(config.read_text())
            checkpoint=study/'runs'/cfg['experiment_id']/dataset/arm['model']/'seed000'/f'checkpoints/step_{plan["steps"]:07d}.pt'
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert state['config_sha256']==sha256(config)
            model=VelocityNet(dist.dim,arm['model'],**cfg['model_options']).double().eval()
            model.load_state_dict(state['model'],strict=True)
            model.time.frequencies.copy_(torch.tensor(frequencies,dtype=torch.float64))
            for time in (.1,.3,.5,.7,.9):
                x=torch.from_numpy((1-time)*source_points+time*target)
                t=torch.full((len(x),1),time,dtype=torch.float64)
                truth=torch.from_numpy(oracle_geometry(dist,x.numpy(),time)['oracle_jacobian'])
                values={}
                with gate_context(model,values,capture=True):native,full=jacobian(model,x,t)
                for solve in ('ode1','ode2'):
                    with hold_gates(model,values,solve):anchored,frozen=jacobian(model,x,t)
                    torch.testing.assert_close(native,anchored,atol=1e-11,rtol=1e-11)
                    correction=full-frozen
                    error_full=(full-truth).square().sum((1,2))
                    error_frozen=(frozen-truth).square().sum((1,2))
                    rows.append(dict(dataset=dataset,label=arm['label'],time=time,solve=solve,
                        native_error=float(error_full.mean()),anchored_error=float(error_frozen.mean()),
                        error_reduction_from_gate_derivatives=float((error_frozen-error_full).mean()),
                        correction_energy=float(correction.square().sum((1,2)).mean()),
                        forward_anchor_max_difference=float((native-anchored).abs().max())))
    result=dict(state='COMPLETE',split='validation',points=512,rows=rows,
        interpretation='Anchored derivative attribution at trained weights; not a gate-removal training ablation, causal necessity claim or additive decomposition.',
        source_sha256=sha256(Path(__file__)),checkpoint_study=str(study),runtime_buffers_sha256=sha256(buffers))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','data-root','source','buffers','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    result=diagnose(args.study,args.data_root,args.source,args.buffers,args.output)
    print(json.dumps(dict(state=result['state'],rows=len(result['rows']))))
