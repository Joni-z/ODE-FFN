"""Six paired Product16 cold runs for the gate-adjoint mechanism."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.gate_adjoint import build,visit_diagnostics
from scripts import run_conservative_readout as harness
from scripts import run_recurrent_depth as common

DOC='docs/research-gate-adjoint-2026-10-26'
ARMS={f'{family}_{readout}':dict(family=family,readout=readout)
      for family in ('ode_k2','swiglu128','swiglu192') for readout in ('gradient','gate_omitted')}


def train_cell(config,output,dataset,arm,steps):
    harness.train_cell(config,output,dataset,arm,steps=steps,build_model=build,
        arm_settings=ARMS,visit_collector=visit_diagnostics)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('run','cell'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--dataset',choices=common.DATASETS)
    p.add_argument('--arm',choices=tuple(ARMS))
    p.add_argument('--data',type=Path)
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--arms',nargs='+',choices=tuple(ARMS),default=list(ARMS))
    a=p.parse_args()
    if a.action=='cell':
        train_cell(a.config.resolve(),a.output.resolve(),a.dataset,a.arm,int(json.loads(a.config.read_text())['steps']))
    else:
        config=dict(seed=0,steps=a.steps,workers=2,device='cpu',datasets=[common.DATASETS[2]],
            arms={name:ARMS[name] for name in a.arms},data_roots={common.DATASETS[2]:str(a.data.resolve())})
        harness.run_study(a.output.resolve(),config,script_relative='scripts/run_gate_adjoint.py',protocol_relative=DOC+'/protocol.md')
