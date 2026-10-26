"""Fixed-anchor analytic curvature diagnostic and monitored execution."""
import argparse
import copy
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import psutil
import torch

BASE = Path('/Users/mr.z/ode-workspace-prep/architecture-search')
PARENT = BASE/'local_ffn_teacher_20261026'
sys.path.insert(0,str(PARENT/'source'))
import local_study as prior
from teacher_support import BANKS,TEACHERS,STUDENTS,TIMES,load,write,append,sha,teacher,signature
from curvature_math import local,left,frame,reduced


def prefix(raw: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    context = raw.time(t)
    h = raw.input(x)
    values = []
    for j,block in enumerate(raw.blocks):
        scale,shift = block.condition(context).chunk(2,-1)
        z = block.norm(h)*(1+scale)+shift
        values.append(z)
        if j<2:
            h = h+block.residual_scale*block.ffn(z)
    return torch.stack(values,dim=1)


def subjects(teacher_name: str, position: int) -> dict:
    result = {'target': (teacher(teacher_name)[0].raw.blocks[position].ffn,teacher_name)}
    for family in STUDENTS:
        module = prior.cold(family,position)
        folder = PARENT/'output/runs'/f'{teacher_name}--{family}--{position}'
        state = torch.load(folder/'step_0005000.pt',map_location='cpu',weights_only=False)
        module.load_state_dict(state['model'])
        assert signature(module)==load(folder/'result.json')['final_signature']
        result[family] = (module,family)
    for key,(module,family) in result.items():
        module = copy.deepcopy(module).double().eval()
        for p in module.parameters():
            p.requires_grad_(False)
        result[key] = module,family
    return result


def prepare(folder: Path, count: int = 16, progress=None) -> dict:
    folder.mkdir(exist_ok=False)
    full,_ = teacher('swiglu128')
    initial = signature(full)
    raw = copy.deepcopy(full.raw).double()
    rows,states,jacobians,hessians,times,indices,clocks = [],[],[],[],[],[],[]
    replay = []
    for clock in TIMES:
        with np.load(BANKS/f'panel_{clock:.1f}.npz') as bank:
            xx,tt = torch.from_numpy(bank['x']),torch.from_numpy(bank['t'])
        with torch.no_grad():
            native = prefix(full.raw,xx,tt).numpy()
        for j in range(3):
            saved = np.load(PARENT/'output/cache'/prior.file_name(f'panel-{clock:.1f}',j))
            assert np.array_equal(native[:,j],saved)
        for i in range(count):
            x,t = xx[i].double(),tt[i].double()
            function = lambda state:prefix(raw,state[None],t[None])[0]
            with torch.no_grad():
                value = function(x)
            difference = float(np.max(np.abs(value.numpy()-native[i].astype(np.float64))))
            assert difference<=2e-5*max(1.,float(np.max(np.abs(native[i]))))
            jac = torch.func.jacrev(function)(x).detach()
            hessian = torch.func.jacfwd(torch.func.jacrev(function))(x).detach()
            rows.append(x.numpy());clocks.append(t.numpy());states.append(value.numpy())
            jacobians.append(jac.numpy());hessians.append(hessian.numpy());times.append(clock);indices.append(i)
            replay.append(dict(time=clock,index=i,promotion_max_abs=difference))
        if progress:
            progress(prepared_time=clock,prepared_anchors=len(rows))
    assert signature(full)==initial
    np.savez(folder/'prefix.npz',x=np.stack(rows),t=np.stack(clocks),z=np.stack(states),
             jacobian=np.stack(jacobians),hessian=np.stack(hessians),time=np.array(times),index=np.array(indices))
    meta = dict(anchors=len(rows),per_time=count,native_panel_replay='bitwise all 512 rows, all positions',
                promotion=replay,prefix_teacher_signature=initial,prefix_sha256=sha(folder/'prefix.npz'))
    write(folder/'metadata.json',meta)
    return meta


def compose(data: dict, prefix_hessian: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    embedding = left(data['jac'],prefix_hessian)
    return data['hessian']+embedding,embedding


def energy(value: torch.Tensor) -> float:
    return float(value.square().sum())


def collect(module: torch.nn.Module, family: str, bank: dict, position: int) -> dict:
    values,jacobians,local_h,total_h,geometry,grams = [],[],[],[],[],[]
    for index in range(len(bank['x'])):
        z,u,hz = (torch.from_numpy(bank[key][index,position]) for key in ('z','jacobian','hessian'))
        with torch.no_grad():
            data = local(module,z,u,family)
            total,embedding = compose(data,hz)
            values.append(data['value'].numpy());jacobians.append((data['jac']@u).numpy())
            local_h.append(data['hessian'].numpy());total_h.append(total.numpy())
            if family=='ode_k2':
                frozen = data['frozen_hessian']+left(data['frozen_jac'],hz)
                omitted = data['origin_omission']+embedding
                definition = frame(data,u)
                actual = reduced(data['hessian'],definition)
                simple = reduced(data['frozen_hessian'],definition)
                identity_abs = float((actual-simple).abs().max())
                torch.testing.assert_close(actual,simple,rtol=2e-8,atol=5e-10)
                routes = torch.stack([data['middle'],data['clock1'],data['clock2'],embedding])
                gram = routes.reshape(4,-1)@routes.reshape(4,-1).T
                torch.testing.assert_close(gram.sum(),total.square().sum(),rtol=2e-8,atol=5e-10)
                grams.append(gram.numpy())
                geometry.append(dict(anchor=index,time=float(bank['time'][index]),
                    feature_norm=float(z.norm()),prefix_tangent_rank=int(torch.linalg.matrix_rank(u,rtol=1e-10)),
                    full_curvature_energy=energy(total),local_curvature_energy=energy(data['hessian']),
                    embedding_curvature_energy=energy(embedding),frozen_difference_energy=energy(total-frozen),
                    origin_omission_difference_energy=energy(total-omitted),
                    middle_energy=energy(data['middle']),clock1_energy=energy(data['clock1']),clock2_energy=energy(data['clock2']),
                    reduced_identity_max_abs=identity_abs,
                    input_constraint_rank=definition['input_constraint_rank'],
                    retained_input_dimension=definition['v'].shape[1],
                    output_removed_rank=definition['output_removed_rank'],
                    input_constraint_residual=definition['input_constraint_residual'],
                    retained_tangent_min_singular=min(definition['retained_tangent_singular'])))
    return dict(value=np.stack(values),jacobian=np.stack(jacobians),hessian_local=np.stack(local_h),
                hessian_total=np.stack(total_h),geometry=geometry,route_grams=np.stack(grams) if grams else np.empty((0,4,4)))


def native_derivatives(raw: torch.nn.Module,module: torch.nn.Module,x: torch.Tensor,t: torch.Tensor,position: int) -> tuple:
    function = lambda state:module(prefix(raw,state[None],t[None])[0,position][None])[0]
    return (function(x).detach(),torch.func.jacrev(function)(x).detach(),
            torch.func.jacfwd(torch.func.jacrev(function))(x).detach())


def preflight(root: Path) -> None:
    torch.set_num_threads(1)
    prepare(root/'preflight-cache',count=1)
    with np.load(root/'preflight-cache/prefix.npz') as file:
        bank = {key:file[key] for key in file.files}
    raw = copy.deepcopy(teacher('swiglu128')[0].raw).double()
    rows = []
    anchor = 2
    for name in TEACHERS:
        for j in range(3):
            for key,(module,family) in subjects(name,j).items():
                z,u,hz = (torch.from_numpy(bank[k][anchor,j]) for k in ('z','jacobian','hessian'))
                with torch.no_grad():
                    data = local(module,z,u,family)
                    total,_ = compose(data,hz)
                x,t = torch.from_numpy(bank['x'][anchor]),torch.from_numpy(bank['t'][anchor])
                value,jac,hessian = native_derivatives(raw,module,x,t,j)
                for actual,expected in [(value,data['value']),(jac,data['jac']@u),(hessian,total)]:
                    torch.testing.assert_close(actual,expected,rtol=2e-8,atol=5e-10)
                row = dict(teacher=name,position=j,subject=key,
                           value_max_abs=float((value-data['value']).abs().max()),
                           jacobian_max_abs=float((jac-data['jac']@u).abs().max()),
                           hessian_max_abs=float((hessian-total).abs().max()))
                if family=='ode_k2':
                    with torch.no_grad():
                        definition = frame(data,u)
                        a,b = reduced(data['hessian'],definition),reduced(data['frozen_hessian'],definition)
                    torch.testing.assert_close(a,b,rtol=2e-8,atol=5e-10)
                    row['identity_max_abs'] = float((a-b).abs().max())
                    aa1,aa2 = data['first']['a'].detach(),data['second']['a'].detach()
                    # Native local counterfactual with constant durations, including the nonlinear prefix.
                    def frozen(state):
                        h = prefix(raw,state[None],t[None])[0,j]
                        for layer,duration,activate in [(module.ode1,aa1,True),(module.ode2,aa2,False)]:
                            first = layer.proj(h)*duration
                            second = layer.proj(first)*duration/2
                            h = h+first+second
                            if activate:
                                h = torch.nn.functional.silu(h)
                        return h
                    actual = torch.func.jacfwd(torch.func.jacrev(frozen))(x).detach()
                    expected = data['frozen_hessian']+left(data['frozen_jac'],hz)
                    torch.testing.assert_close(actual,expected,rtol=2e-8,atol=5e-10)
                    row['frozen_hessian_max_abs'] = float((actual-expected).abs().max())
                rows.append(row)
    write(root/'preflight.json',dict(status='PASS',cases=24,checks=rows))


def case(root: Path,name: str,position: int) -> None:
    torch.set_num_threads(1)
    folder = root/'output/cases'/f'{name}--{position}'
    folder.mkdir(exist_ok=False)
    started = time.monotonic()
    def status(state='RUNNING',**extra):
        write(folder/'status.json',dict(state=state,pid=os.getpid(),elapsed_seconds=time.monotonic()-started,**extra))
    status()
    try:
        with np.load(root/'output/cache/prefix.npz') as file:
            bank = {key:file[key] for key in file.files}
        models = subjects(name,position)
        signatures = {key:signature(module) for key,(module,_) in models.items()}
        data = {}
        for key,(module,family) in models.items():
            data[key] = collect(module,family,bank,position)
            np.savez(folder/(key+'.npz'),**{k:v for k,v in data[key].items() if k!='geometry'})
            write(folder/(key+'-geometry.json'),data[key]['geometry'])
            status(latest_subject=key)
        projected = []
        ode,_ = models['ode_k2']
        for anchor in range(len(bank['x'])):
            z,u = (torch.from_numpy(bank[k][anchor,position]) for k in ('z','jacobian'))
            with torch.no_grad():
                original = local(ode,z,u,'ode_k2')
                definition = frame(original,u)
                truth = reduced(torch.from_numpy(data['target']['hessian_local'][anchor]),definition)
                truth_energy = energy(truth)
                full_energy = float(np.sum(data['target']['hessian_local'][anchor]**2))
                row = dict(anchor=anchor,time=float(bank['time'][anchor]),teacher_retained_energy=truth_energy,
                           teacher_full_local_energy=full_energy,teacher_retained_fraction=truth_energy/full_energy if full_energy>1e-18 else None)
                for family in STUDENTS:
                    value = reduced(torch.from_numpy(data[family]['hessian_local'][anchor]),definition)
                    row[family+'_error_energy'] = energy(value-truth)
                projected.append(row)
        write(folder/'projected-errors.json',projected)
        assert signatures=={key:signature(module) for key,(module,_) in models.items()}
        write(folder/'provenance.json',dict(teacher=name,position=position,signatures=signatures,
              source_sha256=sha(Path(__file__)),prefix_sha256=sha(root/'output/cache/prefix.npz'),
              device='cpu',threads=torch.get_num_threads(),anchors=len(bank['x'])))
        status(state='COMPLETE')
    except BaseException:
        status(state='FAILED',error=traceback.format_exc())
        raise


def observe() -> dict:
    parent = psutil.Process()
    processes = []
    for p in [parent]+parent.children(recursive=True):
        try:
            processes.append(dict(pid=p.pid,rss_bytes=p.memory_info().rss,cpu_seconds=sum(p.cpu_times()[:2]),command=p.cmdline()))
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
    return dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                available_memory_bytes=psutil.virtual_memory().available,processes=processes)


def run(root: Path) -> None:
    torch.set_num_threads(1)
    ready = load(root/'ready.json')
    assert load(root/'preflight.json')['status']=='PASS'
    for path,digest in ready['bindings'].items():
        assert sha(path)==digest,path
    out = root/'output'
    out.mkdir(exist_ok=False)
    stop,abort,lock = threading.Event(),threading.Event(),threading.Lock()
    children = set()
    started = time.monotonic()
    progress = dict(state='RUNNING',phase='prepare',completed=0,total=6,pid=os.getpid())
    def status(**extra):
        progress.update(extra,elapsed_seconds=time.monotonic()-started)
        write(out/'status.json',progress)
        print(progress,flush=True)
    def monitor():
        while not stop.is_set():
            value = observe()
            append(out/'monitor.jsonl',value);write(out/'monitor.json',value)
            stop.wait(5)
    watcher = threading.Thread(target=monitor,daemon=True);watcher.start();status()
    try:
        prepare(out/'cache',progress=lambda **x:status(**x))
        (out/'cases').mkdir();status(phase='cases')
        env = os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        def launch(job):
            name,j = job
            command = [sys.executable,str(Path(__file__)),'case','--root',str(root),'--teacher',name,'--position',str(j)]
            with (out/f'{name}--{j}.log').open('w') as log:
                with lock:
                    if abort.is_set():raise RuntimeError('study aborted')
                    p = subprocess.Popen(command,cwd=Path(__file__).parent,env=env,stdout=log,stderr=subprocess.STDOUT)
                    children.add(p);append(out/'launches.jsonl',dict(teacher=name,position=j,pid=p.pid,command=command))
                code=p.wait()
                with lock:children.remove(p)
            if code:raise RuntimeError(f'{job} exited {code}')
            return job
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(launch,(name,j)) for name in TEACHERS for j in range(3)]
            try:
                for future in as_completed(futures):status(completed=progress['completed']+1,latest=list(future.result()))
            except BaseException:
                abort.set()
                for future in futures:future.cancel()
                with lock:
                    for p in children:p.terminate()
                raise
        for path,digest in ready['bindings'].items():assert sha(path)==digest,path
        status(state='COMPLETE',phase='complete')
    except BaseException:
        status(state='FAILED',error=traceback.format_exc())
        raise
    finally:
        stop.set();watcher.join(timeout=6)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['preflight','run','case'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--teacher',choices=TEACHERS)
    parser.add_argument('--position',type=int,choices=range(3))
    args=parser.parse_args()
    if args.action=='preflight':preflight(args.root)
    elif args.action=='run':run(args.root)
    else:case(args.root,args.teacher,args.position)
