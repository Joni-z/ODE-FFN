"""Freeze two single-factor waves and arithmetic-budget controls. No submission."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from synthetic.models import VelocityNet


def forward_macs(kind,k,options):
    opts={**options,'orders':k}
    model=VelocityNet(2,kind,**opts)
    total=[0]
    def count(m,args,out):
        total[0]+=args[0].numel()*m.out_features
    hooks=[m.register_forward_hook(count) for m in model.modules() if isinstance(m,torch.nn.Linear)]
    with torch.no_grad():model(torch.zeros(1,2),torch.zeros(1,1))
    for hook in hooks:hook.remove()
    # P applications use F.linear rather than calling proj.forward.
    total[0]+=sum(k*layer.proj.weight.numel() for _,layer in model.ode_layers())
    return total[0]


def main():
    root=Path(__file__).resolve().parents[1]
    base=json.loads((root/'configs/synthetic/train_v1.json').read_text())
    folder=root/'configs/synthetic/stage_20261020'
    folder.mkdir(parents=True,exist_ok=True)
    entries=[]
    specs=[('gate_k1_input','A','ode_original',1,5000),
           ('gate_k1_time','A','ode_time',1,5000),
           ('gate_k2_input','A','ode_original',2,5000),
           ('gate_k2_time','A','ode_time',2,5000),
           ('silu_5000','A','ffn',2,5000),
           ('swiglu_5000','A','swiglu',2,5000),
           ('order_k4_input','B','ode_original',4,5000),
           ('order_k8_input','B','ode_original',8,5000)]
    ode_macs={str(k):forward_macs('ode_original',k,base['model_options']) for k in (1,2,4,8)}
    budgets={}
    for label,kind in [('silu','ffn'),('swiglu','swiglu')]:
        mac=forward_macs(kind,2,base['model_options'])
        match={k:math.ceil(v*5000/mac) for k,v in ode_macs.items()}
        budgets[label]={'model':kind,'forward_macs':mac,'matched_steps':match}
        specs.append((label+'_budget','B',kind,2,max(match.values())))
    for label,wave,kind,k,steps in specs:
        cfg=copy.deepcopy(base)
        cfg.update(experiment_id='odeffn_stage_20261020_'+label,models=[kind],steps=steps)
        cfg['model_options']['orders']=k
        evaluations=[5000]
        if label.endswith('_budget'):
            evaluations=sorted(set([5000,*budgets[label.split('_')[0]]['matched_steps'].values()]))
            cfg['milestone_steps']=evaluations
        path=folder/(label+'.json')
        text=json.dumps(cfg,indent=2)+'\n'
        if path.exists() and path.read_text()!=text:
            raise FileExistsError(f'configuration already frozen with different content: {path}')
        path.write_text(text)
        entries.append({'label':label,'wave':wave,'model':kind,'orders':k if kind.startswith('ode_') else None,
                        'config':str(path.relative_to(root)), 'sha256':hashlib.sha256(text.encode()).hexdigest(),
                        'steps':steps,'evaluate_steps':evaluations,
                        'parameters':VelocityNet(2,kind,**cfg['model_options']).parameter_count(),
                        'forward_macs':forward_macs(kind,k,cfg['model_options'])})
    manifest={'protocol':'ODELayer gate and Taylor-order ablations, 2026-10-20',
              'datasets':base['datasets'],'seed':0,'entries':entries,
              'ode_forward_macs':ode_macs,'baseline_budgets':budgets,
              'budget_definition':'Total network forward linear MACs per example, including time embedding and FiLM, excluding bias/nonlinear ops/backward/optimizer/evaluation. A proxy, not measured equal GPU time.',
              'run_count':len(entries)*len(base['datasets']),
              'eval_checkpoint_count':sum(len(e['evaluate_steps']) for e in entries)*len(base['datasets'])}
    out=folder/'manifest.json'
    text=json.dumps(manifest,indent=2)+'\n'
    if out.exists() and out.read_text()!=text:raise FileExistsError(out)
    out.write_text(text)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
