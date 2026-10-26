"""Three new radial-JVP backbones; retain the frozen direct/gradient controls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.directional_readout import build
from scripts import run_conservative_readout as harness
from scripts import run_recurrent_depth as common

DOC='docs/research-derivative-orientation-2026-10-26'
ARMS={f'{family}_directional':dict(family=family,readout='directional')
      for family in ('ode_k2','swiglu128','swiglu192')}


def train_cell(config,output,dataset,arm,steps):
    harness.train_cell(config,output,dataset,arm,steps=steps,build_model=build,arm_settings=ARMS)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('run','cell'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--dataset',choices=common.DATASETS)
    p.add_argument('--arm',choices=tuple(ARMS))
    p.add_argument('--data-v1',type=Path)
    p.add_argument('--data-product',type=Path)
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--datasets',nargs='+',choices=common.DATASETS,default=list(common.DATASETS))
    p.add_argument('--arms',nargs='+',choices=tuple(ARMS),default=list(ARMS))
    args=p.parse_args()
    if args.action=='cell':
        train_cell(args.config.resolve(),args.output.resolve(),args.dataset,args.arm,
                   int(json.loads(args.config.read_text())['steps']))
    else:
        config=dict(seed=0,steps=args.steps,workers=2,device='cpu',datasets=args.datasets,
            arms={name:ARMS[name] for name in args.arms},
            data_roots={d:str((args.data_product if d==common.DATASETS[2] else args.data_v1).resolve()) for d in args.datasets})
        harness.run_study(args.output.resolve(),config,script_relative='scripts/run_derivative_orientation.py',
                          protocol_relative=DOC+'/protocol.md')
