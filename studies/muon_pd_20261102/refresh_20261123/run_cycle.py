"""Run the two registered matched-budget cells on the Mac and score selection banks."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
import subprocess
import time
import sys

from refresh_trial import HERE, base, original, torch
sys.path.insert(0, str(HERE.parent/'efficient_solvers_20261122'))
import cycle

STUDIES = {
    'original': ('grid.json', 'refresh_20261123'),
    'odd': ('odd-grid.json', 'odd_field_20261123'),
    'control': ('odd-control-grid.json', 'odd_control_20261123'),
    'bounded': ('bounded-grid.json', 'bounded_odd_20261123'),
    'glu': ('glu-grid.json', 'glu_midpoint_20261123'),
    'euler': ('glu-euler-grid.json', 'glu_euler_ring_20261123'),
}


def run(work, variant, workers):
    torch.set_num_threads(1)
    grid, tag = STUDIES[variant]
    odd = variant != 'original'
    trainer = 'odd_trial.py' if odd else 'refresh_trial.py'
    from odd_trial import build_model as all_models
    cycle.build_model = all_models
    budgets = {r['dataset']: r['target_flops'] for r in cycle.read(cycle.DOC/'budget.json')['rows']
               if r['family']=='state_midpoint4'}
    cells=cycle.read(HERE/grid)['cells']
    for c in cells:
        ds=c['dataset']
        dim=2 if ds=='gmm_ring8_2d' else 16
        torch.manual_seed(0)
        reference_family = c['family'].removeprefix('bounded_').removeprefix('odd_') if odd else 'state_midpoint4'
        ref=original(dim,'swiglu192' if variant=='glu' else reference_family)
        torch.manual_seed(0)
        model=all_models(dim,c['family'])
        x=torch.randn(32,dim)
        t=torch.rand(32,1)
        torch.testing.assert_close(model(x,t),ref(x,t),rtol=0,atol=0)
        if odd:
            with torch.no_grad():
                model.output.weight.normal_()
                torch.testing.assert_close(model(x,t), -model(-x,t), rtol=0, atol=0)
        cost=cycle.step_flops(ds, c['family'], c['optimizer'], c['projection_lr'])
        c.update(step_flops=cost, steps=cycle.closest(budgets[ds],cost))
    base.dump(work/'budget.json',cells)
    base.dump(work/'progress.json',dict(pid=os.getpid(), phase='training',cells=cells))

    def execute(cell):
        ds=cell['dataset']
        steps=cell['steps']
        folder=work/'train'/tag/ds/cell['id']/'seed000'
        endpoint=folder/'checkpoints'/f'step_{steps:07d}.pt'
        def call(script, args):
            with (work/f'{ds}-{script}.log').open('a') as log:
                subprocess.run([sys.executable,'-u',str(HERE/script),*map(str,args)],
                               stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,**cycle.THREADS),check=True)
        if not (endpoint.exists() and (folder/'ADAPTER_COMPLETE').exists()):
            prior=sorted((folder/'checkpoints').glob('step_*.pt'))
            call(trainer,['--cell',cell['id'],'--output',work/'train','--data-root',work.parent/'data/synthetic',
                             '--device','cpu','--steps',steps]+(['--resume',prior[-1]] if prior else []))
        output=work/'selection'/ds
        if output.exists() and not (output/'COMPLETE.json').exists():
            output.rename(output.with_name(f'{ds}-partial-{time.time_ns()}'))
        if not (output/'COMPLETE.json').exists():
            call('evaluate.py',['--folder',folder,'--assets',work.parent/'coefficient-20261116/assets','--output',output,
                            '--data-root',work.parent/'data/synthetic','--split','selection','--device','cpu','--steps',steps])
        done=cycle.read(output/'COMPLETE.json')
        binding=cycle.read(output/'binding.json')
        assert done['passed'] and not done['witness'] and done['steps']==[steps]
        assert binding['split']=='selection' and binding['source_config']['cell']['id']==cell['id']
        assert (binding['banks'],binding['samples_per_bank'],binding['heun_steps'])==(8,4096,128)
        score,=cycle.read(output/'summary.json')
        assert score['checkpoint_sha256']==base.digest(endpoint)
        return dict(dataset=ds,**score)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(execute,cells))
    base.dump(work/'selection.json',rows)
    base.dump(work/'progress.json',dict(pid=os.getpid(), phase='selection_complete'))

if __name__=='__main__':
    from pathlib import Path
    work=Path(sys.argv[1]).resolve()
    work.mkdir(parents=True,exist_ok=True)
    with (work/'cycle.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('work')
        parser.add_argument('--workers', choices=(1,2), type=int, default=2)
        variants = parser.add_mutually_exclusive_group()
        for flag in STUDIES.keys()-{'original'}:
            variants.add_argument('--'+flag, dest='variant', action='store_const', const=flag)
        args = parser.parse_args()
        run(work, args.variant or 'original', args.workers)
