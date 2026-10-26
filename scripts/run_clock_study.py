"""Windows, Linux and macOS supervisor for the seed-0 centered clock study."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def benchmark_worker(device: str, updates: int) -> dict:
    import torch
    from synthetic.models import VelocityNet
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = VelocityNet(2, 'ode_original').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
    rng = torch.Generator(device=device).manual_seed(100000)
    def update() -> None:
        y = torch.randn((512,2), generator=rng, device=device)
        x0 = torch.randn((512,2), generator=rng, device=device)
        t = torch.rand((512,1), generator=rng, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = (model((1-t)*x0+t*y,t)-(y-x0)).square().mean()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        loss.item()
        grad.item()
    for _ in range(30):
        update()
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.monotonic()
    for _ in range(updates):
        update()
    if device == 'cuda':
        torch.cuda.synchronize()
    return dict(device=device,updates=updates,seconds=time.monotonic()-start)


def benchmark(output: Path) -> dict:
    rows = []
    for device, workers in [('cpu',1),('cpu',2),('cuda',1),('cuda',2)]:
        folder = output/f'benchmark-{device}-{workers}'
        folder.mkdir()
        processes, handles = [], []
        start = time.monotonic()
        for index in range(workers):
            target = folder/f'{index}.json'
            handle = (folder/f'{index}.log').open('w')
            handles.append(handle)
            processes.append(subprocess.Popen([sys.executable,str(Path(__file__)),
                '--benchmark-worker',device,'--worker-output',str(target)],
                cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT))
        for process in processes:
            if process.wait(timeout=240) != 0:
                raise RuntimeError(f'benchmark failed; see {folder}')
        for handle in handles:
            handle.close()
        measurements = [json.loads((folder/f'{i}.json').read_text()) for i in range(workers)]
        seconds = max(item['seconds'] for item in measurements)
        rows.append(dict(device=device,workers=workers,updates_per_second=200*workers/seconds,
            measured_seconds=seconds,total_wall_seconds=time.monotonic()-start))
    # Avoid extra concurrency for marginal improvements.
    best = max(rows,key=lambda r:r['updates_per_second'])
    alternatives = [r for r in rows if r['workers']==1 and r['device']==best['device']]
    if alternatives[0]['updates_per_second'] >= .9*best['updates_per_second']:
        best = alternatives[0]
    result = dict(rows=rows,selected=best,
        note='Kernel timings exclude warmup; CUDA also benchmarked with two independent processes.')
    write(output/'benchmark.json',result)
    return best


def monitor(output: Path, stop: threading.Event, progress: dict) -> None:
    import psutil
    current = psutil.Process()
    current.cpu_percent(None)
    gpu_program=shutil.which('nvidia-smi')
    while not stop.is_set():
        children = current.children(recursive=True)
        process_rows = []
        for child in children:
            try:
                process_rows.append(dict(pid=child.pid,name=child.name(),
                    rss_mib=child.memory_info().rss/1024**2))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        row = dict(time=now(),pid=os.getpid(),cpu_percent=psutil.cpu_percent(None),
            memory_available_gib=psutil.virtual_memory().available/1024**3,
            processes=process_rows, **progress.copy())
        training = []
        for path in (output/'runs').rglob('metrics.jsonl'):
            lines = path.read_text().splitlines()
            if lines:
                try:
                    last = json.loads(lines[-1])
                    training.append(dict(run=str(path.parent.relative_to(output)),
                        step=last['step'],oracle_mse=last['validation_oracle_mse'],
                        complete=(path.parent/'COMPLETE').exists()))
                except json.JSONDecodeError:
                    pass  # A worker may be in the middle of appending its last line.
        row['training']=training
        if gpu_program:
            try:
                gpu = subprocess.run([gpu_program,'--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                row['gpu_csv'] = gpu.stdout.strip()
            except (OSError, subprocess.TimeoutExpired) as exc:
                row['gpu_monitor_error'] = repr(exc)
        else:
            row['gpu_csv']=None
            row['gpu_monitor_backend']='NVIDIA telemetry unavailable; see recorded training device'
        write(output/'monitor.json',row)
        with (output/'telemetry.jsonl').open('a') as handle:
            handle.write(json.dumps(row,allow_nan=False)+'\n')
        stop.wait(10)


def run(output: Path, data_root: Path) -> None:
    from synthetic.train import sha256
    output.mkdir(parents=True,exist_ok=False)
    progress = dict(state='RUNNING',phase='preflight',completed=0,active=[])
    write(output/'status.json',dict(time=now(),**progress))
    stop = threading.Event()
    watcher = threading.Thread(target=monitor,args=(output,stop,progress),daemon=True)
    watcher.start()
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        manifest = json.loads((data_root/'manifest.json').read_text())
        for name, record in manifest['files'].items():
            if sha256(data_root/Path(name)) != record['sha256']:
                raise ValueError(f'dataset hash mismatch: {name}')
        import torch
        write(output/'provenance.json',dict(time=now(),host=platform.node(),python=sys.version,
            torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
            dataset_manifest_sha256=sha256(data_root/'manifest.json'),
            source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in
                [*sorted((ROOT/'synthetic').glob('*.py')),Path(__file__),ROOT/'scripts/check_clock_curvature.py']},
            protocol_sha256=sha256(ROOT/'docs/research-2026-10-24-lenovo/protocol.md')))
        progress['phase']='benchmark'
        selected = benchmark(output)
        device,workers = selected['device'],selected['workers']
        config = json.loads((ROOT/'configs/synthetic/gates_20261021/ode_original.json').read_text())
        config.update(experiment_id='clock_20261024',data_root=str(data_root),
            datasets=['gaussian_aniso_2d','gmm_ring8_2d'],
            models=['ode_original','ode_gate_linear','swiglu'],milestone_steps=[1000,2000,3000,4000,5000])
        config['model_options']['gate_center']=.6
        config_path=output/'config.json'
        write(config_path,config)
        env=os.environ.copy()
        env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        if device == 'cpu':
            env['CUDA_VISIBLE_DEVICES']=''
        cells=[]
        lock=threading.Lock()
        def train(cell: tuple[str,str]) -> None:
            dataset,kind=cell
            key=dataset+'/'+kind
            with lock:
                progress['active'].append(key)
                write(output/'status.json',dict(time=now(),**progress))
            log=output/f'{dataset}-{kind}.log'
            with log.open('w') as handle:
                result=subprocess.run([sys.executable,'-m','synthetic.train','--config',str(config_path),
                    '--dataset',dataset,'--model',kind,'--seed','0','--device',device,
                    '--output-root',str(output/'runs')],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'training failed: {key}; see {log}')
            folder=output/'runs'/config['experiment_id']/dataset/kind/'seed000'
            diagnostic=output/'diagnostics'/dataset/kind
            with (output/f'{dataset}-{kind}-diagnostic.log').open('w') as handle:
                result=subprocess.run([sys.executable,str(ROOT/'scripts/check_clock_curvature.py'),
                    '--checkpoint',str(folder/'checkpoints/step_0005000.pt'),
                    '--config',str(config_path),'--data-root',str(data_root),'--output',str(diagnostic)],
                    cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'diagnostic failed: {key}')
            rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            record=dict(dataset=dataset,model=kind,final=rows[-1],
                diagnostic=json.loads((diagnostic/'summary.json').read_text()))
            with lock:
                cells.append(record)
                progress['active'].remove(key)
                progress['completed']+=1
                write(output/'results.json',cells)
                write(output/'status.json',dict(time=now(),**progress))
        progress['phase']='native_and_baseline'
        first=[(d,k) for d in config['datasets'] for k in ('ode_original','swiglu')]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(train,first))
        material=any(c['diagnostic'].get('curvature_material',False) for c in cells)
        decision=dict(time=now(),trigger_threshold=.01,run_gate_linear_control=material,
            evidence=[dict(dataset=c['dataset'],aggregate=c['diagnostic']['aggregate']['all_removed'])
                for c in cells if c['model']=='ode_original'])
        write(output/'decision.json',decision)
        if material:
            progress['phase']='curvature_control'
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(train,[(d,'ode_gate_linear') for d in config['datasets']]))
        progress.update(state='COMPLETE',phase='complete')
        write(output/'status.json',dict(time=now(),**progress))
    except BaseException as exc:
        progress.update(state='FAILED',error=repr(exc))
        write(output/'status.json',dict(time=now(),**progress))
        raise
    finally:
        stop.set()
        watcher.join(timeout=15)
        if sys.platform == 'win32':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--benchmark-worker',choices=('cpu','cuda'))
    parser.add_argument('--worker-output',type=Path)
    args=parser.parse_args()
    if args.benchmark_worker:
        write(args.worker_output,benchmark_worker(args.benchmark_worker,200))
    else:
        if args.output is None or args.data_root is None:
            parser.error('--output and --data-root are required')
        run(args.output.resolve(),args.data_root.resolve())
