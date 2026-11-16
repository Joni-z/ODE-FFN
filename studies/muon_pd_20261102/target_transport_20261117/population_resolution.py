"""Portable actual tensors; unchanged native Heun and SW2 at fixed resolutions."""
import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
STEPS = (128, 127, 129, 512, 2048)


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def export(args):
    import numpy as np
    from common import assets, restored_pair, base, ev, torch
    torch.set_num_threads(1)
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    cases = read(args.cases)['cases']; rows = []
    assets_seen = {}
    for case in cases:
        model, crossed, identity = restored_pair(case, str(args.data_root))
        snapshot = ev.tensor_snapshot(model)
        # Include nonpersistent buffers. Do not reconstruct frequencies on145.
        tensors = {}
        for kind, values in [('parameter', model.named_parameters()), ('buffer', model.named_buffers())]:
            for name, value in values:
                assert value.is_floating_point()
                tensors[kind+'::'+name] = value.detach().double().numpy().copy()
        file = out/(case['id']+'-tensors.npz'); np.savez_compressed(file, **tensors)
        row = dict(**case, tensor_file=file.name, tensor_sha256=sha(file),
                   tensor_arrays={k: ev.array_hash(v) for k,v in tensors.items()},
                   identity=identity, splits={})
        for split in ('selection', 'confirmation'):
            arrays, _, binding = assets(args.assets, case['dataset'], split, 'cpu')
            key = (case['dataset'], split)
            if key not in assets_seen:
                data = {k:v for k,v in arrays.items() if k=='directions' or
                        k.startswith(('source_', 'target_')) and not k.startswith('target2_')}
                filename = case['dataset']+'-'+split+'.npz'
                np.savez_compressed(out/filename, **data)
                assets_seen[key] = dict(file=filename, sha256=sha(out/filename),
                    arrays={k:ev.array_hash(v) for k,v in data.items()}, original_binding=binding)
            folder = args.transport_root/'formal'/split/case['id']
            result = read(folder/'summary.json'); assert not result['witness']
            cache = {}
            for bank, original in enumerate(result['banks']):
                bankfile = folder/original['file']; assert sha(bankfile)==original['sha256']
                with np.load(bankfile) as z:
                    cache['native_'+str(bank)] = z['native'].copy()
                    cache['crossed_'+str(bank)] = z['endpoints'][-1].copy()
            filename = case['id']+'-'+split+'-cached128.npz'
            np.savez_compressed(out/filename, **cache)
            row['splits'][split] = dict(assets=assets_seen[key], cache_file=filename,
                cache_sha256=sha(out/filename), original_summary_sha256=sha(folder/'summary.json'))
        ev.same_tensors(model, snapshot); rows.append(row)
        dump(out/'progress.json', dict(stage='export', completed=len(rows), total=len(cases)))
        print('EXPORTED_ACTUAL_TENSORS', case['id'], flush=True)
    protocol = HERE.parents[2]/'docs/research-assessments/2026-11-16-ode-mechanism/population-resolution-protocol.md'
    dump(out/'portable.json', dict(cases=rows, training_seed=0, steps=list(STEPS),
        protocol_sha256=sha(protocol), source_sha256=sha(__file__),
        nonpersistent_buffers_included=True, new_training=0, new_draws=0))


def load_model(root, row):
    import numpy as np
    from common import torch, ev, substitute_solver
    from rk4_trial import build_model
    path = root/row['tensor_file']; assert sha(path)==row['tensor_sha256']
    family = 'state_midpoint4' if row['trained_solver']=='midpoint' else 'state_rk4_2'
    dim = 2 if row['dataset']=='gmm_ring8_2d' else 16
    torch.manual_seed(0); model = build_model(dim, family).double().eval()
    named = {'parameter::'+k:v for k,v in model.named_parameters()}
    named.update({'buffer::'+k:v for k,v in model.named_buffers()})
    with np.load(path) as z, torch.no_grad():
        assert set(named)==set(z.files)
        for name, value in named.items():
            assert ev.array_hash(z[name])==row['tensor_arrays'][name]
            assert tuple(value.shape)==z[name].shape
            value.copy_(torch.from_numpy(z[name]))
    crossed = copy.deepcopy(model); substitute_solver(crossed, row['inference_solver'])
    return model, crossed


def select_gpu(requested):
    sys.path.insert(0, str(HERE.parent/'coefficient_20261116'))
    from sharing_dispatch import eligible
    query = lambda fields: subprocess.check_output(['nvidia-smi','--query-'+fields,
        '--format=csv,noheader,nounits'], text=True)
    rows = list(csv.reader(io.StringIO(query('gpu=index,uuid,memory.used'))))
    apps = list(csv.reader(io.StringIO(query('compute-apps=gpu_uuid,pid'))))
    occupied = {r[0].strip() for r in apps if len(r)>1 and r[1].strip().isdigit()}
    idle = eligible(rows, occupied)
    if requested: idle = [(i,u) for i,u in idle if u==requested]
    if not idle: print('NO_IDLE_PERMITTED_GPU', flush=True); raise SystemExit(75)
    return idle[0]


def evaluate(args):
    physical, uuid = (None, None)
    if args.device=='cuda':
        physical, uuid = select_gpu(args.gpu_uuid)
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    else: os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # Torch is imported only after the physical GPU restriction is installed.
    import numpy as np
    from common import base, ev, torch
    from gpu_resolution import exclusive_gpu
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    if args.device=='cuda': assert torch.cuda.device_count()==1
    guard = (lambda:exclusive_gpu(uuid)) if uuid else (lambda:None)
    guard(); base.verify_sources()
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    portable = read(args.root/'portable.json')
    protocol = HERE.parents[2]/'docs/research-assessments/2026-11-16-ode-mechanism/population-resolution-protocol.md'
    assert portable['protocol_sha256']==sha(protocol)
    selected = [r for r in portable['cases'] if not args.case or r['id']==args.case]
    assert len(selected)==(1 if args.case else 4)
    if not args.witness:
        assert args.case and args.readiness and read(args.readiness)['passed']
        assert read(args.readiness)['portable_sha256']==sha(args.root/'portable.json')
    rows = []; started = time.monotonic()

    def metric(value, target, directions):
        pool = base.train.sliced_w2(torch.from_numpy(np.concatenate(value)).to(args.device),
            torch.from_numpy(np.concatenate(target)).to(args.device), directions)
        banks = [base.train.sliced_w2(torch.from_numpy(v).to(args.device),
            torch.from_numpy(t).to(args.device),directions) for v,t in zip(value,target,strict=True)]
        return dict(pooled_sw2=pool, bank_sw2=banks)

    for row in selected:
        models = dict(zip(('native','crossed'), load_model(args.root,row), strict=True))
        models = {k:v.to(args.device) for k,v in models.items()}
        snapshots = {k:ev.tensor_snapshot(v) for k,v in models.items()}
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone() if uuid else None
        for split, files in row['splits'].items():
            datafile = args.root/files['assets']['file']; assert sha(datafile)==files['assets']['sha256']
            cachefile = args.root/files['cache_file']; assert sha(cachefile)==files['cache_sha256']
            with np.load(datafile) as z: data = {k:z[k].copy() for k in z.files}
            assert {k:ev.array_hash(v) for k,v in data.items()}==files['assets']['arrays']
            with np.load(cachefile) as z: cache = {k:z[k].copy() for k in z.files}
            count = 64 if args.witness else 4096
            sources = [torch.from_numpy(data[f'source_{b}'][:count]).double().to(args.device) for b in range(8)]
            targets = [data[f'target_{b}'][:count].astype(np.float64) for b in range(8)]
            directions = torch.from_numpy(data['directions']).double().to(args.device)
            for steps in ((128,) if args.witness else STEPS):
                stage_start = time.monotonic(); values = {}; replays = {}
                for kind, model in models.items():
                    banks = []
                    for bank, source in enumerate(sources):
                        guard()
                        dump(out/'progress.json',dict(pid=os.getpid(),case=row['id'],split=split,
                            steps=steps,kind=kind,bank=bank,completed_rows=len(rows),seconds=time.monotonic()-started))
                        generated = base.train.generated_samples(model,source,steps).cpu().numpy()
                        assert np.isfinite(generated).all(); banks.append(generated); guard()
                    values[kind] = banks
                    if steps==128:
                        expected = [cache[f'{kind}_{b}'][:count] for b in range(8)]
                        errors = [np.abs(a-b) for a,b in zip(banks,expected,strict=True)]
                        qualified = all(np.all(e<=1e-7*(1+np.abs(b))) for e,b in zip(errors,expected,strict=True))
                        actual_metrics = metric(banks,targets,directions); ref_metrics = metric(expected,targets,directions)
                        metric_errors = np.abs(np.array([actual_metrics['pooled_sw2']]+actual_metrics['bank_sw2'])-
                                               np.array([ref_metrics['pooled_sw2']]+ref_metrics['bank_sw2']))
                        refs = np.array([ref_metrics['pooled_sw2']]+ref_metrics['bank_sw2'])
                        qualified &= bool(np.all(metric_errors<=2e-8+1e-7*np.abs(refs)))
                        replays[kind] = dict(qualified=bool(qualified),max_component_error=float(max(e.max() for e in errors)),
                            max_metric_error=float(metric_errors.max()))
                        dump(out/(row['id']+'-'+split+'-replay.json'),replays)
                        assert qualified, ('Original128replay failed',row['id'],split,kind,replays[kind])
                filename = row['id']+'-'+split+f'-heun{steps}.npz'
                np.savez_compressed(out/filename, **{k:np.stack(v) for k,v in values.items()})
                result = dict(case=row['id'],split=split,steps=steps,points_per_bank=count,
                    native=metric(values['native'],targets,directions),crossed=metric(values['crossed'],targets,directions),
                    file=filename,sha256=sha(out/filename),seconds=time.monotonic()-stage_start,replay=replays)
                rows.append(result); dump(out/'rows.json',rows)
                if uuid:
                    status = subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True)
                    with (out/'gpu-history.jsonl').open('a') as f:f.write(json.dumps(dict(unix_time=time.time(),case=row['id'],split=split,steps=steps,resource=status.strip()))+'\n')
                print('RESOLUTION_ROW',row['id'],split,steps,result['seconds'],flush=True)
        for kind,model in models.items():ev.same_tensors(model,snapshots[kind])
        assert torch.equal(rng,torch.get_rng_state())
        if uuid: assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
        del models; guard()
    dump(out/'COMPLETE.json',dict(passed=True,witness=args.witness,rows=len(rows),
        physical_gpu=physical,gpu_uuid=uuid,device=args.device,portable_sha256=sha(args.root/'portable.json'),
        source_sha256=sha(__file__),parameters_buffers_rng_unchanged=True,seconds=time.monotonic()-started))
    print('POPULATION_RESOLUTION_COMPLETE',len(rows),flush=True)


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('export')
    for name in ('cases','data-root','assets','transport-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p = sub.add_parser('evaluate')
    for name in ('root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case');p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--gpu-uuid');p.add_argument('--readiness',type=Path);p.add_argument('--witness',action='store_true')
    args = parser.parse_args(); (export if args.command=='export' else evaluate)(args)


if __name__=='__main__':main()
