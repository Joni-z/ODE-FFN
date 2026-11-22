"""Replay the most extreme Product16 source and test the signed readout's ray limit."""
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
from odd_trial import base, build_model, torch
from evaluate_screen import assets, ev


def read(path):
    return json.loads(path.read_text())


@torch.no_grad()
def run(prep, output):
    torch.set_num_threads(1)
    os.nice(10)
    ds='rotated_product_bimodal_16d'
    registered=prep/'odd-field-20261123/selection'/ds
    full,=read(registered/'summary.json')
    def saved(directory,step,i):
        return np.load(directory/f'{step}-b{i}.npy')
    maxima=[]
    for bank in range(8):
        y=saved(registered,full['step'],bank)
        r=np.linalg.norm(y.astype(float),axis=1)
        maxima.append((float(r.max()),bank,int(r.argmax())))
    _,bank,index=max(maxima)
    arr,_,bank_info=assets(prep/'coefficient-20261116/assets',ds,'selection','cpu')
    direction=torch.from_numpy(saved(registered,full['step'],bank)[index]).double().reshape(1,-1)
    direction/=direction.norm()
    rows=[]
    for tag in ('odd-field','bounded-odd','odd-control'):
        directory=prep/f'{tag}-20261123/selection'/ds
        score,=read(directory/'summary.json')
        setting=read(directory/'binding.json')
        cfg=setting['source_config']
        manifest=Path(cfg['data_root'])/'manifest.json'
        law=base.train.load_distribution(read(manifest)['datasets'][ds])
        ev.build_model=build_model
        model,identity=ev.restore(Path(setting['source_folder']),cfg,manifest,score['step'],'cpu')
        state=torch.from_numpy(arr[f'source_{bank}']).clone()
        path=[]
        dt=1/128
        for i in range(128):
            t=torch.full((len(state),1),i*dt,dtype=state.dtype)
            drift=model(state,t)
            if i in (0,32,64,96,120,127):
                x=state[index:index+1].double()
                actual=law.velocity(x.numpy(),np.array([[i*dt]]))
                unit=x/x.norm()
                path.append(dict(t=i*dt,radius=x.norm().item(),
                                 radial_speed=(drift[index].double()*unit).sum().item(),
                                 oracle_radial_speed=float(np.sum(actual*unit.numpy()))))
            state=state+.5*dt*(drift+model(state+dt*drift,torch.full_like(t,(i+1)*dt)))
        np.testing.assert_array_equal(state.numpy(),saved(directory,score['step'],bank))
        final=state[index].double().norm().item()
        # Along a nondegenerate ray the squared carrier grows as r². Each
        # normalized residual block contributes O(1), so the last norm tends
        # to the normalized squared carrier, independent of the time embedding.
        model.double()
        p=torch.nn.functional.linear(direction,model.input.weight)
        squared=p.square()
        n=(squared-squared.mean())/squared.std(unbiased=False)
        limit=model.output((p.sign() if model.bounded_carrier else p)*(model.norm.weight*n+model.norm.bias))
        errors={}
        for r in (100.,1000.,10000.):
            actual=torch.cat([model(r*direction,torch.tensor([[t]],dtype=torch.double)) for t in (.1,.5,.9)])
            scaled=actual if model.bounded_carrier else actual/r
            errors[str(int(r))]=((scaled-limit).norm(dim=1)/limit.norm()).max().item()
        bound=None
        if model.bounded_carrier:
            bound=(torch.linalg.matrix_norm(model.output.weight,ord=2)
                   *(math.sqrt(model.width)*model.norm.weight.abs().max()+model.norm.bias.norm())).item()
        rows.append(dict(family=cfg['cell']['family'],step=score['step'],trajectory=path,endpoint_radius=final,
                         radial_limit=(limit*direction).sum().item(),bounded_velocity_norm=bound,
                         ray_relative_error_across_three_times=errors,**identity))
    base.dump(output,dict(dataset=ds,stream='selection',bank=bank,index=index,bank_sha256=bank_info['sha256'],
                         selection_rule='largest unbounded-midpoint endpoint norm',rows=rows))


if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]))
