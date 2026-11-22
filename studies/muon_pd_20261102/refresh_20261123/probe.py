"""Localize the effect of refreshing the nonlinear response on fixed checkpoints."""
import argparse
import json
from pathlib import Path
import sys
from types import MethodType

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'coefficient_20261116'))
from state_trial import base, build_model, torch
from evaluate_screen import assets, ev

@torch.no_grad()
def intervene(self, x, steps):
    y = x
    weight = self.operator()
    dt = 1 / steps
    for _ in range(steps):
        diagonal = self.step_size(x if self.mode == 'entry' else y)
        half = y + dt / 2 * torch.nn.functional.linear(y * diagonal, weight)
        next_diag = self.step_size(half) if self.mode == 'native' else diagonal
        y = y + dt * torch.nn.functional.linear(half * next_diag, weight)
    return y

def measure(prep, output):
    torch.set_num_threads(1)
    ev.build_model = build_model
    study = prep/'coefficient-20261116'
    rows = []
    for cfgfile in sorted((study/'state-output/state_flow_20261116').glob('*/*/seed000/config.json')):
        cfg = json.loads(cfgfile.read_text())
        family = cfg['cell']['family']
        if family != 'state_midpoint4':
            continue
        dataset = cfg['cell']['dataset']
        ckpt = cfgfile.parent
        model, identity = ev.restore(ckpt,cfg,Path(cfg['data_root'])/'manifest.json',20000,'cpu')
        arr, fixed, bank = assets(study/'assets',dataset,'selection','cpu')
        x = fixed['x'][:512]
        t = fixed['t'][:512]
        src = torch.from_numpy(arr['source_0'][:512])
        ref = model(x,t).detach()
        for mode in ('native','substep','entry'):
            for _, layer in model.ode_layers():
                layer.mode = mode
                layer.solve = MethodType(intervene,layer)
            field = model(x,t).detach()
            if mode=='native':
                torch.testing.assert_close(field,ref,atol=0,rtol=0)
            endpoint = base.train.generated_samples(model,src,32)
            if mode=='native':
                reference_endpoint=endpoint
            row=dict(dataset=dataset,mode=mode,**identity, assets_sha256=bank['sha256'],
                     field_delta_rms=(field-ref).square().mean().sqrt().item(),
                     endpoint_delta_rms=(endpoint-reference_endpoint).square().mean().sqrt().item(),
                     field_mse=(field-fixed['oracle'][:512]).square().mean().item())
            rows.append(row)
            print(json.dumps(row),flush=True)
    assert len(rows)==6
    base.dump(output,rows)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('prep',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    with torch.no_grad():
        measure(args.prep,args.output)
