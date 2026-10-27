"""Fixed32-node reference checks, separate from the trained two-node model."""
from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def gate_reference(activation: torch.Tensor,transport: torch.Tensor,count: int) -> torch.Tensor:
    m=transport.shape[0];left,right=activation[:,:m],activation[:,m:]
    displacement=F.linear(right,transport)
    nodes,weights=np.polynomial.legendre.leggauss(count)
    receiver=sum(float(w/2)*F.silu(left+float((s+1)/2)*displacement) for s,w in zip(nodes,weights))
    return torch.cat((receiver,F.silu(right)),-1)


class ReferenceFFN(nn.Module):
    def __init__(self,module: nn.Module) -> None:
        super().__init__()
        self.w12=copy.deepcopy(module.w12);self.w3=copy.deepcopy(module.w3)
        self.transport=nn.Parameter(module.transport.detach().clone())

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        activation,value=self.w12(x).chunk(2,-1)
        return self.w3(gate_reference(activation,self.transport,32)*value)


def relative_rms(difference: torch.Tensor,reference: torch.Tensor) -> float:
    return float((difference.double().square().sum()/reference.double().square().sum().clamp_min(1e-30)).sqrt())


def run(root: Path,out: Path,study,native,settings: dict) -> bool:
    gates=[];fields=[];panels=[]
    for entry in settings['laws']:
        law=entry['law'];banks=Path(entry['banks']);folder=root/'output/runs'/law/'quadrature2'
        model=study.build(root,law,'quadrature2')
        native.restore_model(model,torch.load(folder/'step_0005000.pt',weights_only=False));model.eval()
        hooks=[]
        for index,block in enumerate(model.raw.blocks):
            def capture(module,args,index=index):
                with torch.no_grad():
                    activation=module.w12(args[0]).chunk(2,-1)[0]
                    current=module.integrated_gate(activation).double()
                    q32=gate_reference(activation.double(),module.transport.double(),32)
                    q64=gate_reference(activation.double(),module.transport.double(),64)
                    ratio=relative_rms(current-q32,q32)
                    gates.append(dict(law=law,layer=index,anchors=len(activation),
                        gate_q2_vs32_relative_rms=ratio,reference32_vs64_relative_rms=relative_rms(q32-q64,q64),
                        within_one_percent=ratio<=.01))
            hooks.append(block.ffn.register_forward_pre_hook(capture))
        with np.load(banks/'probe.npz') as data:
            with torch.no_grad():model(torch.from_numpy(data['x'][:32]),torch.from_numpy(data['t'][:32]))
        for hook in hooks:hook.remove()
        reference=copy.deepcopy(model)
        for block in reference.raw.blocks:block.ffn=ReferenceFFN(block.ffn)
        # No training, optimizer step or new checkpoint is created for this reference.
        checked=native.legacy.field_evaluation(reference,banks,save=out/f'{law}-reference32-field.npz')
        original=study.load(folder/'result.json');change=abs(checked['oracle_mse']/original['oracle_mse']-1)
        fields.append(dict(law=law,q2_field_mse=original['oracle_mse'],reference32_field_mse=checked['oracle_mse'],
            relative_mse_change=change,within_one_percent=change<=.01))
        for clock in study.TIMES:
            with np.load(banks/f'panel_{clock:.1f}.npz') as data:x,t=data['x'],data['t']
            with np.load(folder/'diagnostics'/f'panel_{clock:.1f}.npz') as data:actual=torch.from_numpy(data['jacobian'])
            value,jac=native.legacy.jacobian(reference,torch.from_numpy(x),torch.from_numpy(t))
            ratio=relative_rms(actual-jac,jac)
            panels.append(dict(law=law,time=clock,anchors=len(x),complete_J_q2_vs32_relative_rms=ratio,within_one_percent=ratio<=.01))
            np.savez_compressed(out/f'{law}-reference32-panel-{clock:.1f}.npz',prediction=value.numpy(),jacobian=jac.numpy())
        print(dict(numerical_reference=law,field_relative_change=change),flush=True)
    for name,rows in [('quadrature-gates.csv',gates),('quadrature-fields.csv',fields),('quadrature-panels.csv',panels)]:
        with (out/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    passed=all(row['within_one_percent'] for row in gates+fields+panels)
    study.write(out/'numerical-accuracy.json',dict(status='PASS' if passed else 'FAIL',gate_probes=len(gates),field_banks=len(fields),jacobian_panels=len(panels),
        frozen_weights_no_retraining=True,finite_bank_not_global_certificate=True,
        max_gate_relative_rms=max(r['gate_q2_vs32_relative_rms'] for r in gates),
        max_reference32_vs64_relative_rms=max(r['reference32_vs64_relative_rms'] for r in gates),
        max_field_mse_relative_change=max(r['relative_mse_change'] for r in fields),
        max_complete_jacobian_relative_rms=max(r['complete_J_q2_vs32_relative_rms'] for r in panels)))
    return passed
