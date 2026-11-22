"""Probe carry-over of the previous block's increment on fixed ordinary weights."""
import json
import os
from pathlib import Path
import sys
from types import MethodType
from odd_trial import base, build_model, torch
from evaluate_screen import assets, ev


def velocity(self,x,t):
    carrier=torch.nn.functional.linear(x,self.input.weight)
    state=carrier.square()+self.input.bias
    emb=self.time(t)
    previous=None
    for block in self.blocks:
        proposal=block(state,emb)
        slope=proposal-state
        state=proposal if previous is None else proposal+self.history*(slope-previous)
        previous=slope
    return self.output(carrier*self.norm(state))


@torch.no_grad()
def run(prep,out):
    os.nice(10)
    torch.set_num_threads(1)
    ev.build_model=build_model
    rows=[]
    for ds,w in [('gmm_ring8_2d','glu-euler-ring-20261123'),('rotated_product_bimodal_16d','odd-control-20261123')]:
        path=prep/w/'selection'/ds
        setting=json.loads((path/'binding.json').read_text())
        result,=json.loads((path/'summary.json').read_text())
        cfg=setting['source_config']
        assert cfg['cell']['family']=='odd_swiglu192'
        model,identity=ev.restore(Path(setting['source_folder']),cfg,Path(cfg['data_root'])/'manifest.json',result['step'],'cpu')
        arrays,fixed,bank=assets(prep/'coefficient-20261116/assets',ds,'selection','cpu')
        pred=model(fixed['x'],fixed['t'])
        src=torch.from_numpy(arrays['source_0'][:1024])
        target=torch.from_numpy(arrays['target_0'][:1024])
        native=base.train.generated_samples(model,src,32)
        model.forward=MethodType(velocity,model)
        for coefficient in (0.,.5,-.5):
            model.history=coefficient
            y=model(fixed['x'],fixed['t'])
            samples=base.train.generated_samples(model,src,32)
            if coefficient==0:
                torch.testing.assert_close(y,pred,rtol=0,atol=0)
                torch.testing.assert_close(samples,native,rtol=0,atol=0)
            mse=(y.double()-fixed['oracle']).square().mean().item()
            sw2=base.train.sliced_w2(samples.double(),target.double(),fixed['directions'].double())
            rows.append(dict(dataset=ds,family=cfg['cell']['family'],coefficient=coefficient,mse=mse,
                             pilot_sw2=sw2,examples=1024,outer_heun=32,assets_sha256=bank['sha256'],**identity))
    base.dump(out,rows)


if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]))
