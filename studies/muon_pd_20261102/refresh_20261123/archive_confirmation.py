"""Compare completed endpoints on Report 4's existing confirmation stream."""
from pathlib import Path
import json
import os
import subprocess
import sys
from odd_trial import HERE, base

def read(p):
    return json.loads(p.read_text())


def run(prep, work):
    os.nice(5)
    source=read(work/'selection.json')
    scores=[]
    for row in source:
        ds=row['dataset']
        loc=work/'selection'/ds
        selection=read(loc/'binding.json')
        folder=Path(selection['source_folder'])
        cp=folder/'checkpoints'/f"step_{row['step']:07d}.pt"
        assert base.digest(cp)==row['checkpoint_sha256']
        out=work/'archive-confirmation'/ds
        if not (out/'COMPLETE.json').exists():
            subprocess.run([sys.executable,str(HERE/'evaluate.py'),'--folder',str(folder),
                            '--data-root',str(prep/'data/synthetic'),'--assets',str(prep/'coefficient-20261116/assets'),
                            '--output',str(out),'--split','confirmation','--steps',str(row['step']),'--device','cpu'],check=True)
        done=read(out/'COMPLETE.json'); binding=read(out/'binding.json')
        assert done['passed'] and not done['witness'] and done['steps']==[row['step']]
        assert binding['assets']['sha256']!=selection['assets']['sha256']
        assert binding['split']=='confirmation' and binding['source_config']['seeds']==[0]
        assert (binding['banks'],binding['heun_steps'],binding['samples_per_bank'])==(8,128,4096)
        score,=read(out/'summary.json')
        assert score['checkpoint_sha256']==row['checkpoint_sha256']
        old=read(HERE.parents[2]/'docs/report4/equal-compute/results.json')
        controls=[r for r in old['rows'] if r['dataset']==ds and r['family'] in ('free_isru','swiglu64','swiglu192')]
        assert len(controls)==2 and all(r['assets_sha256']==binding['assets']['sha256'] for r in controls)
        scores.append(dict(dataset=ds,family=binding['source_config']['cell']['family'],
                           best_legacy_sw2=min(min(r['original_20k_sw2'],r['pooled_sw2']) for r in controls),
                           assets_sha256=binding['assets']['sha256'],**score))
        base.dump(work/'archive-confirmation/scores.json',scores)


if __name__=='__main__':
    work=Path(sys.argv[1])
    run(work.parent,work)
