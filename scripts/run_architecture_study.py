"""Execute one frozen, hypothesis-specific architecture plan with monitoring."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_clock_study import monitor,now,write


def run_queue(plan_path: Path,data_root: Path,output: Path) -> None:
    plan=json.loads(plan_path.read_text())
    output.mkdir(parents=True,exist_ok=False)
    progress=dict(state='RUNNING',completed=0,total=len(plan['stages']),active=None)
    process=None
    try:
        for stage in plan['stages']:
            child_output=output.parent/stage['output_name']
            command=[sys.executable,str(Path(__file__)),'--plan',str(ROOT/stage['plan']),
                '--data-root',stage.get('data_root',str(data_root)),'--output',str(child_output)]
            with (output/(stage['output_name']+'.log')).open('w') as handle:
                process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
                progress.update(active=stage['output_name'],pid=process.pid)
                while process.poll() is None:
                    write(output/'status.json',dict(time=now(),**progress))
                    time.sleep(5)
            if process.returncode!=0:raise RuntimeError(f'queue stage failed: {stage["output_name"]}')
            child=json.loads((child_output/'status.json').read_text())
            assert child['state']=='COMPLETE'
            progress['completed']+=1
        progress.update(state='COMPLETE',active=None)
    except BaseException as exc:
        progress.update(state='FAILED',error=repr(exc))
        if process is not None and process.poll() is None:
            process.terminate()
        raise
    finally:
        write(output/'status.json',dict(time=now(),**progress))


def macs(model,dim: int) -> int:
    import torch
    from synthetic.models import CollapsedODEFFN
    count=[0]
    def add(module,args,result):count[0]+=args[0].numel()*module.out_features
    hooks=[m.register_forward_hook(add) for m in model.modules() if isinstance(m,torch.nn.Linear)]
    try:
        with torch.no_grad():model(torch.zeros(1,dim),torch.zeros(1,1))
    finally:
        for hook in hooks:hook.remove()
    count[0]+=sum(layer.orders*layer.proj.weight.numel() for _,layer in model.ode_layers())
    count[0]+=sum(m.matrix1.numel()+m.matrix2.numel() for m in model.modules() if isinstance(m,CollapsedODEFFN))
    if hasattr(model,'covariance_layers'):
        count[0]+=sum(layer.field.linear_macs() for _,layer in model.covariance_layers())
    if hasattr(model,'state_clock_layers'):
        count[0]+=sum(layer.projection_calls*layer.proj.weight.numel() for _,layer in model.state_clock_layers())
    if hasattr(model,'lowrank_layers'):
        count[0]+=sum(layer.projection_macs() for _,layer in model.lowrank_layers())
    return count[0]


def run(plan_path: Path,data_root: Path,output: Path) -> None:
    import torch
    from synthetic.models import VelocityNet
    from synthetic.train import sha256
    torch.set_num_threads(1)
    plan=json.loads(plan_path.read_text())
    if plan['seed']!=0:raise ValueError('This project uses training seed0 only')
    if 'stages' in plan:
        run_queue(plan_path,data_root,output)
        return
    output.mkdir(parents=True,exist_ok=False)
    progress=dict(state='RUNNING',phase='preflight',completed=0,
        total=len(plan['datasets'])*len(plan['arms']),active=[])
    write(output/'status.json',dict(time=now(),**progress))
    stop=threading.Event()
    watcher=threading.Thread(target=monitor,args=(output,stop,progress),daemon=True)
    watcher.start()
    if sys.platform=='win32':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        manifest=json.loads((data_root/'manifest.json').read_text())
        for dataset in plan['datasets']:
            for name in ('train.npz','validation.npz','test.npz','probe.npz'):
                key=dataset+'/'+name
                assert sha256(data_root/dataset/name)==manifest['files'][key]['sha256']
        source=[*sorted((ROOT/'synthetic').glob('*.py')),Path(__file__),
            ROOT/'scripts/check_vector_field.py',ROOT/'scripts/check_gate_geometry.py',
            ROOT/'scripts/run_clock_study.py',plan_path,ROOT/plan['protocol']]
        frozen={str(p.relative_to(ROOT)):sha256(p) for p in source}
        def verify() -> None:
            for name,digest in frozen.items():
                assert sha256(ROOT/name)==digest,name
        write(output/'provenance.json',dict(time=now(),host=platform.node(),python=sys.version,
            torch=torch.__version__,device=plan['device'],workers=plan['workers'],plan=plan,
            source_sha256=frozen,dataset_manifest_sha256=sha256(data_root/'manifest.json')))
        configs={}
        budgets={}
        (output/'configs').mkdir()
        for arm in plan['arms']:
            label=arm['label']
            path=output/'configs'/f'{label}.json'
            if 'resume_study' in plan:
                previous=Path(plan['resume_study'])/'configs'/f'{label}.json'
                path.write_bytes(previous.read_bytes())
                cfg=json.loads(path.read_text())
                assert cfg['models']==[arm['model']]
                assert Path(cfg['data_root']).resolve()==data_root
                assert all(cfg['model_options'][k]==v for k,v in arm.get('options',{}).items())
            else:
                cfg=json.loads((ROOT/'configs/synthetic/gates_20261021/ode_original.json').read_text())
                cfg.update(experiment_id=plan['experiment_id']+'_'+label,data_root=str(data_root),
                    datasets=plan['datasets'],models=[arm['model']],seeds=[0],steps=plan['steps'])
                cfg['model_options'].update(arm.get('options',{}))
                if 'training_targets' in arm:cfg['training_targets']=arm['training_targets']
                if 'backbone_initialization' in arm:
                    cfg['backbone_initialization']=copy.deepcopy(arm['backbone_initialization'])
                write(path,cfg)
            configs[label]=(cfg,path)
            for dataset in plan['datasets']:
                dim=manifest['datasets'][dataset]['dim']
                torch.manual_seed(0)
                model=VelocityNet(dim,arm['model'],**cfg['model_options'])
                budgets[dataset+'/'+label]=dict(parameters=model.parameter_count(),forward_macs=macs(model,dim))
        write(output/'budgets.json',budgets)
        env=os.environ.copy()
        env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        if plan['device']=='cpu':env['CUDA_VISIBLE_DEVICES']=''
        results=[]
        lock=threading.Lock()
        progress['phase']='training'
        def cell(item) -> None:
            dataset,arm=item
            label,kind=arm['label'],arm['model']
            cfg,config_path=configs[label]
            key=dataset+'/'+label
            verify()
            with lock:
                progress['active'].append(key)
                write(output/'status.json',dict(time=now(),**progress))
            folder=output/'runs'/cfg['experiment_id']/dataset/kind/'seed000'
            extra=[]
            prior_wall=0.
            if 'resume_study' in plan:
                previous=Path(plan['resume_study'])/'runs'/cfg['experiment_id']/dataset/kind/'seed000'
                shutil.copytree(previous,folder)
                step=int((folder/'COMPLETE').read_text())
                checkpoint=folder/f'checkpoints/step_{step:07d}.pt'
                assert step<plan['steps']
                old=json.loads((previous/'provenance.json').read_text())
                assert old['device']==plan['device']
                prior_wall=json.loads((folder/'metrics.jsonl').read_text().splitlines()[-1])['wall_seconds']
                write(folder/'continuation-input.json',dict(previous=str(previous),
                    checkpoint_sha256=sha256(checkpoint),from_step=step,to_step=plan['steps']))
                extra=['--resume',str(checkpoint),'--steps',str(plan['steps'])]
            with (output/f'{dataset}-{label}.log').open('w') as handle:
                result=subprocess.run([sys.executable,'-m','synthetic.train','--config',str(config_path),
                    '--dataset',dataset,'--model',kind,'--seed','0','--device',plan['device'],
                    '--output-root',str(output/'runs'),*extra],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'training failed: {key}')
            diagnostic=output/'diagnostics'/dataset/label
            with (output/f'{dataset}-{label}-diagnostic.log').open('w') as handle:
                result=subprocess.run([sys.executable,str(ROOT/'scripts/check_vector_field.py'),
                    '--checkpoint',str(folder/f'checkpoints/step_{plan["steps"]:07d}.pt'),
                    '--config',str(config_path),'--data-root',str(data_root),'--output',str(diagnostic)],
                    cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'diagnostic failed: {key}')
            verify()
            metrics=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            final=metrics[-1]
            geometry=json.loads((diagnostic/'summary.json').read_text())
            record=dict(dataset=dataset,label=label,model=kind,step=final['step'],
                validation_oracle_mse=final['validation_oracle_mse'],
                validation_sw2=final['sample']['sliced_w2'],
                field_mse=geometry['field_mse'],jacobian_error=geometry['jacobian_error'],
                wall_seconds=prior_wall+final['wall_seconds'],**budgets[key])
            with lock:
                results.append(record)
                progress['active'].remove(key)
                progress['completed']+=1
                write(output/'results.json',results)
                write(output/'status.json',dict(time=now(),**progress))
        jobs=[(dataset,arm) for dataset in plan['datasets'] for arm in plan['arms']]
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:list(pool.map(cell,jobs))
        verify()
        assert len(results)==progress['total']
        progress.update(state='COMPLETE',phase='complete')
        write(output/'status.json',dict(time=now(),**progress))
    except BaseException as exc:
        progress.update(state='FAILED',error=repr(exc))
        write(output/'status.json',dict(time=now(),**progress))
        raise
    finally:
        stop.set()
        watcher.join(timeout=15)
        if sys.platform=='win32':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        path=output/'monitor.json'
        if path.exists():
            final=json.loads(path.read_text())
            final.update(time=now(),**progress)
            write(path,final)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','data-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    run(args.plan.resolve(),args.data_root.resolve(),args.output.resolve())
