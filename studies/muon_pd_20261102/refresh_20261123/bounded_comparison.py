"""Evaluate the same-readout SwiGLU separately from the original locked cohort."""
from pathlib import Path
import os
import shutil
import sys
import time
import subprocess
from final_banks import HERE, DATASETS, read, run, base


def compare(prep, ds):
    work=prep/'bounded-control-20261123'
    source=work/'selection'/ds
    while not (source/'COMPLETE.json').exists():
        os.kill(int((work/'pid').read_text()),0)
        time.sleep(10)
    done=read(source/'COMPLETE.json')
    setting=read(source/'binding.json')
    score,=read(source/'summary.json')
    assert done['passed'] and not done['witness'] and done['steps']==[score['step']]
    cfg=setting['source_config'];cell=cfg['cell']
    assert cfg['seeds']==[0] and cell['family'].startswith('bounded_odd_swiglu')
    folder=Path(setting['source_folder'])
    file=folder/'checkpoints'/f"step_{score['step']:07d}.pt"
    budget,=[r for r in read(work/'budget.json') if r['dataset']==ds]
    assert score['step']==budget['steps'] and base.digest(file)==score['checkpoint_sha256']
    os.nice(6)
    confirmation=work/'archive-confirmation'/ds
    if not (confirmation/'COMPLETE.json').exists():
        subprocess.run([sys.executable,str(HERE/'evaluate.py'),'--folder',str(folder),'--data-root',str(prep/'data/synthetic'),
                        '--assets',str(prep/'coefficient-20261116/assets'),'--output',str(confirmation),
                        '--split','confirmation','--steps',str(score['step']),'--device','cpu'],check=True)
    confirm,=read(confirmation/'summary.json')
    assert read(confirmation/'COMPLETE.json')['passed'] and confirm['checkpoint_sha256']==score['checkpoint_sha256']
    locked=prep/'bounded-locked-2026112350'
    candidate=read(locked/ds/'candidate/score.json')
    extra=dict(dataset=ds,label='bounded-swiglu',family=cell['family'],folder=str(folder),
               step=score['step'],checkpoint_sha256=score['checkpoint_sha256'])
    out=work/'fresh-comparison'/ds
    dest=out/ds/'candidate'
    dest.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(locked/ds/'candidate/score.json',dest/'score.json')
    run(prep,[candidate['model'],extra],out,stream=2026112350,datasets=(ds,))
    final=read(out/'results.json')
    main,control=final['rows']
    assert main==candidate and main['bank_hashes']==control['bank_hashes']
    base.dump(out/'comparison.json',dict(dataset=ds,family=extra['family'],steps=budget['steps'],
              step_flops=budget['step_flops'],checkpoint_sha256=extra['checkpoint_sha256'],
              selection_sw2=score['pooled_sw2'],confirmation_sw2=confirm['pooled_sw2'],
              fresh_sw2=control['pooled_sw2'],bounded_midpoint_fresh_sw2=main['pooled_sw2'],
              paired_banks_favoring_midpoint=sum(x<y for x,y in zip(main['bank_sw2'],control['bank_sw2'])),
              fresh_bank_sw2=control['bank_sw2'],stream=final['stream']))


if __name__=='__main__':
    compare(Path(sys.argv[1]),sys.argv[2])
