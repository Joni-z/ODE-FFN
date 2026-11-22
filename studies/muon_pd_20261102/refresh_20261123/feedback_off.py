"""Switch off only the learned gate feedback on the trained candidate."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from odd_trial import base, build_model, torch
from evaluate_screen import assets, ev


@torch.no_grad()
def run(work,out):
    os.nice(5)
    torch.set_num_threads(1)
    ev.build_model=build_model
    rows=[]
    for trained in json.loads((work/'selection.json').read_text()):
        ds=trained['dataset']
        setting=json.loads((work/'selection'/ds/'binding.json').read_text())
        folder=Path(setting['source_folder']);cfg=setting['source_config']
        assert cfg['cell']['family']=='odd_feedback192'
        model,binding=ev.restore(folder,cfg,Path(cfg['data_root'])/'manifest.json',trained['step'],'cpu')
        assert binding['checkpoint_sha256']==trained['checkpoint_sha256']
        arr,fixed,banks=assets(work.parent/'coefficient-20261116/assets',ds,'selection','cpu')
        snapshot=ev.tensor_snapshot(model)
        fb=[block.ffn.feedback.detach().clone() for block in model.blocks]
        native=model(fixed['x'],fixed['t'])
        for block in model.blocks:
            block.ffn.feedback.zero_()
        off=model(fixed['x'],fixed['t'])
        sources=[torch.from_numpy(arr[f'source_{i}']) for i in range(8)]
        targets=[torch.from_numpy(arr[f'target_{i}']).double() for i in range(8)]
        samples=[base.train.generated_samples(model,x,128).double() for x in sources]
        dirs=fixed['directions'].double()
        sw2=base.train.sliced_w2(torch.cat(samples),torch.cat(targets),dirs)
        bank=[base.train.sliced_w2(y,z,dirs) for y,z in zip(samples,targets)]
        for block,param in zip(model.blocks,fb):
            block.ffn.feedback.copy_(param)
        ev.same_tensors(model,snapshot)
        torch.testing.assert_close(native,model(fixed['x'],fixed['t']),rtol=0,atol=0)
        rows.append(dict(dataset=ds,step=trained['step'],native_sw2=trained['pooled_sw2'],zero_feedback_sw2=sw2,
                         paired_banks_favoring_feedback=int((np.array(bank)>trained['bank_sw2']).sum()),
                         native_mse=trained['field_mse'],zero_feedback_mse=(off.double()-fixed['oracle']).square().mean().item(),
                         gain_rms_per_block=[g.square().mean().sqrt().item() for g in fb],
                         assets_sha256=banks['sha256'],**binding))
        base.dump(out,rows)


if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]))
