"""How much does the winning bounded-readout checkpoint use its moving gate?"""
import json
import os
from pathlib import Path
import sys
from types import MethodType
import numpy as np
from odd_trial import base, build_model, torch
from probe import assets, ev, intervene


@torch.no_grad()
def run(work, output):
    os.nice(10)
    torch.set_num_threads(1)
    ev.build_model=build_model
    scored=[]
    for endpoint in json.loads((work/'selection.json').read_text()):
        ds=endpoint['dataset']
        setting=json.loads((work/'selection'/ds/'binding.json').read_text())
        folder=Path(setting['source_folder']);cfg=setting['source_config']
        assert cfg['cell']['family']=='bounded_odd_state_midpoint4' and cfg['seeds']==[0]
        model,identity=ev.restore(folder,cfg,Path(cfg['data_root'])/'manifest.json',endpoint['step'],'cpu')
        assert endpoint['checkpoint_sha256']==identity['checkpoint_sha256']
        snapshot=ev.tensor_snapshot(model)
        arr,fixed,binding=assets(work.parent/'coefficient-20261116/assets',ds,'selection','cpu')
        assert binding['sha256']==setting['assets']['sha256']
        dirs=fixed['directions'].double()
        target=torch.from_numpy(np.concatenate([arr[f'target_{i}'] for i in range(8)])).double()
        reference=model(fixed['x'][:512],fixed['t'][:512])
        for mode in ('native','substep','entry'):
            for _,layer in model.ode_layers():
                assert layer.substeps==4
                layer.solve=MethodType(intervene,layer);layer.mode=mode
            pred=model(fixed['x'][:512],fixed['t'][:512])
            if mode=='native':
                torch.testing.assert_close(pred,reference,rtol=0,atol=0)
                pooled=endpoint['pooled_sw2']
                bank_scores=endpoint['bank_sw2']
            else:
                generated=[base.train.generated_samples(model,torch.from_numpy(arr[f'source_{i}']),128).double()
                           for i in range(8)]
                pooled=base.train.sliced_w2(torch.cat(generated),target,dirs)
                bank_scores=[base.train.sliced_w2(y,torch.from_numpy(arr[f'target_{i}']).double(),dirs)
                             for i,y in enumerate(generated)]
            scored.append(dict(dataset=ds,step=endpoint['step'],mode=mode,pooled_sw2=pooled,
                               bank_sw2=bank_scores,field_mse_512=(pred-fixed['oracle'][:512]).square().mean().item(),
                               assets_sha256=binding['sha256'],**identity))
            base.dump(output,scored)
            ev.same_tensors(model,snapshot)


if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]))
