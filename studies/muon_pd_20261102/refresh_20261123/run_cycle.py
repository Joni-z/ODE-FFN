"""Run the two registered matched-budget cells on the Mac and score selection banks."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
import subprocess
import sys

from trial import HERE, base, build_model, original, torch
sys.path.insert(0, str(HERE.parent/'efficient_solvers_20261122'))
import cycle

def run(work):
    torch.set_num_threads(1)
    cycle.build_model = build_model
    budgets = {r['dataset']: r['target_flops'] for r in cycle.read(cycle.DOC/'budget.json')['rows']
               if r['family']=='state_midpoint4'}
    cells=cycle.read(HERE/'grid.json')['cells']
    for c in cells:
        ds=c['dataset']
        dim=2 if ds=='gmm_ring8_2d' else 16
        torch.manual_seed(0)
        ref=original(dim,'state_midpoint4')
        torch.manual_seed(0)
        model=build_model(dim,c['family'])
        x=torch.randn(32,dim)
        t=torch.rand(32,1)
        torch.testing.assert_close(model(x,t),ref(x,t),rtol=0,atol=0)
        cost=cycle.step_flops(ds,c['family'])
        c.update(step_flops=cost, steps=cycle.closest(budgets[ds],cost))
    base.dump(work/'budget.json',cells)
    base.dump(work/'progress.json',dict(pid=os.getpid(), phase='training',cells=cells))

    def execute(cell):
        ds=cell['dataset']
        steps=cell['steps']
        folder=work/'train/refresh_20261123'/ds/cell['id']/'seed000'
        endpoint=folder/'checkpoints'/f'step_{steps:07d}.pt'
        def call(script, args):
            with (work/f'{ds}-{script}.log').open('a') as log:
                subprocess.run([sys.executable,'-u',str(HERE/script),*map(str,args)],
                               stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,**cycle.THREADS),check=True)
        if not (endpoint.exists() and (folder/'ADAPTER_COMPLETE').exists()):
            prior=sorted((folder/'checkpoints').glob('step_*.pt'))
            call('trial.py',['--cell',cell['id'],'--output',work/'train','--data-root',work.parent/'data/synthetic',
                             '--device','cpu','--steps',steps]+(['--resume',prior[-1]] if prior else []))
        output=work/'selection'/ds
        call('evaluate.py',['--folder',folder,'--assets',work.parent/'coefficient-20261116/assets','--output',output,
                            '--data-root',work.parent/'data/synthetic','--split','selection','--device','cpu','--steps',steps])
        score,=cycle.read(output/'summary.json')
        assert score['checkpoint_sha256']==base.digest(endpoint)
        return dict(dataset=ds,**score)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(execute,cells))
    base.dump(work/'selection.json',rows)
    base.dump(work/'progress.json',dict(pid=os.getpid(), phase='selection_complete'))

if __name__=='__main__':
    from pathlib import Path
    work=Path(sys.argv[1]).resolve()
    work.mkdir(parents=True,exist_ok=True)
    with (work/'cycle.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(work)
