"""Fixed-checkpoint mean/radial/tangent intervention with full downstream replay."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import psutil
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.recurrent_depth import build
from synthetic.normalized_geometry import GeometryNet
from synthetic.train import restore_model

DOC='docs/research-normalized-geometry-2026-10-26'
ARMS={'ode_k2_native_final':'ode_k2','swiglu192_native_final':'swiglu192','swiglu128_native_deep':'swiglu128'}
TIMES=(.1,.3,.5,.7,.9)
GLOBALS=('native','drop_uniform','drop_radial','drop_tangent')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def csv_write(path,rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def derivative(model,x,t,capture=None):
    xx=x.detach().clone().requires_grad_(True)
    prediction=model(xx,t,capture=capture)
    jac=torch.stack([torch.autograd.grad(prediction[:,j].sum(),xx,retain_graph=j+1<prediction.shape[1])[0]
                     for j in range(prediction.shape[1])],dim=1)
    return prediction.detach(),jac.detach()


@torch.no_grad()
def trace_rows(trace,dataset,arm,variant,time):
    rows=[]
    for record in trace:
        c,f,u,r,q=[record[key].detach() for key in ('centered','branch','uniform','radial','tangent')]
        beta=record['beta'].detach()[:,0];h=record['hidden'].detach();step=record['step'];eps=record['epsilon']
        next_h=record['next_hidden'].detach();next_c=next_h-next_h.mean(-1,keepdim=True)
        ef=f.square().mean(1);eu=u.square().mean(1);er=r.square().mean(1);eq=q.square().mean(1)
        torch.testing.assert_close(f,u+r+q,rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(ef,eu+er+eq,rtol=1e-10,atol=1e-11)
        assert float(q.mean(1).abs().max())<1e-10
        np.testing.assert_allclose((q*c).sum(1).numpy(),0.,rtol=0,atol=1e-8)
        var=c.square().mean(1);next_var=next_c.square().mean(1)
        denominator=c.norm(dim=1)*next_c.norm(dim=1)
        safe=torch.where(denominator>0,denominator,torch.ones_like(denominator))
        cosine=((c*next_c).sum(1)/safe).clamp(-1,1)
        angle=torch.acos(cosine)*180/torch.pi
        for i in range(len(h)):
            rows.append(dict(dataset=dataset,arm=arm,variant=variant,time=time,block=record['block'],input_index=i,
                hidden_energy=float(h[i].square().mean()),centered_energy=float(var[i]),
                next_centered_energy=float(next_var[i]),branch_energy=float(ef[i]),uniform_energy=float(eu[i]),
                radial_energy=float(er[i]),tangent_energy=float(eq[i]),beta=float(beta[i]),
                radial_multiplier=float(1+step*beta[i]),epsilon_fraction=float(eps/(var[i]+eps)),
                angle_defined=bool(denominator[i]>0),angle_degrees=float(angle[i]) if denominator[i]>0 else 0.))
    return rows


def run(inputs,source,output):
    torch.set_num_threads(1)
    hashes=load(source/'source-sha256.json')
    for name,value in hashes.items():assert sha(source/name)==value
    for name in ('synthetic/models.py','synthetic/recurrent_depth.py','synthetic/train.py'):
        assert sha(ROOT/name)==hashes[name]
    output.mkdir(parents=True,exist_ok=False)
    progress=dict(state='RUNNING',pid=os.getpid(),completed=0,total=9);write(output/'status.json',progress)
    bindings=dict(source_sha256=hashes,script_sha256=sha(Path(__file__)),
        geometry_module_sha256=sha(ROOT/'synthetic/normalized_geometry.py'),protocol_sha256=sha(ROOT/DOC/'protocol.md'),inputs={})
    metrics,panels_rows,traces=[],[],[]
    for group in inputs:
        config=load(group/'study.json')
        assert load(group/'status.json')['state']=='COMPLETE'
        assert load(group/'provenance.json')['source_sha256']==hashes
        for dataset in config['datasets']:
            banks=group/'banks'/dataset
            with np.load(banks/'field.npz') as data:field={k:data[k] for k in ('x','t','oracle')}
            bindings['inputs'][str(banks/'field.npz')]=sha(banks/'field.npz')
            saved_panels=[]
            for time in TIMES:
                path=banks/f'panel_{time:.1f}.npz'
                with np.load(path) as data:panel={k:data[k] for k in ('x','t','oracle','oracle_jacobian')}
                bindings['inputs'][str(path)]=sha(path);saved_panels.append(panel)
            for arm,family in ARMS.items():
                folder=output/dataset/arm;folder.mkdir(parents=True,exist_ok=False)
                native_folder=group/'runs'/dataset/arm
                path=native_folder/'step_0005000.pt';bindings['inputs'][str(path)]=sha(path)
                state=torch.load(path,map_location='cpu',weights_only=False)
                parent=build(field['x'].shape[1],family,'native');restore_model(parent,state)
                parent.eval();model=GeometryNet(parent)
                for parameter in model.parameters():parameter.requires_grad_(False)
                # Validate actual native float32 fields and complete derivatives first.
                for time,panel in zip(TIMES,saved_panels):
                    prediction,jac=derivative(model,torch.from_numpy(panel['x']),torch.from_numpy(panel['t']))
                    path=native_folder/'diagnostics'/f'panel_{time:.1f}.npz'
                    with np.load(path) as data:
                        np.testing.assert_array_equal(prediction.numpy(),data['prediction'])
                        np.testing.assert_array_equal(jac.numpy(),data['jacobian'])
                    bindings['inputs'][str(path)]=sha(path)
                model=model.double();native_field=None;native_panels={}
                for variant in GLOBALS:
                    model.variant=variant;model.block_index=None
                    predictions=[]
                    with torch.no_grad():
                        for start in range(0,len(field['x']),1024):
                            predictions.append(model(torch.from_numpy(field['x'][start:start+1024]).double(),
                                torch.from_numpy(field['t'][start:start+1024]).double()).numpy())
                    prediction=np.concatenate(predictions);assert np.isfinite(prediction).all()
                    if variant=='native':native_field=prediction
                    if variant=='drop_uniform':np.testing.assert_allclose(prediction,native_field,rtol=1e-10,atol=1e-10)
                    errors=(prediction-field['oracle'].astype(np.float64))**2
                    primary=float(errors.mean())
                    np.savez_compressed(folder/(variant+'-field.npz'),prediction=prediction)
                    jac_errors=[];panel_errors=[]
                    for time,panel in zip(TIMES,saved_panels):
                        capture=[]
                        value,jac=derivative(model,torch.from_numpy(panel['x']).double(),torch.from_numpy(panel['t']).double(),capture)
                        value,jac=value.numpy(),jac.numpy();assert np.isfinite(value).all() and np.isfinite(jac).all()
                        if variant=='native':native_panels[time]=(value,jac)
                        if variant=='drop_uniform':
                            np.testing.assert_allclose(value,native_panels[time][0],rtol=1e-10,atol=1e-10)
                            np.testing.assert_allclose(jac,native_panels[time][1],rtol=1e-9,atol=1e-9)
                        error=float(np.mean((value-panel['oracle'])**2))
                        je=float(np.mean(np.sum((jac-panel['oracle_jacobian'])**2,axis=(1,2))))
                        jac_errors.append(je);panel_errors.append(error)
                        panels_rows.append(dict(dataset=dataset,arm=arm,variant=variant,block='all',time=time,field_mse=error,jacobian_error=je))
                        traces.extend(trace_rows(capture,dataset,arm,variant,time))
                        np.savez_compressed(folder/f'{variant}-panel-{time:.1f}.npz',prediction=value,jacobian=jac)
                    metrics.append(dict(dataset=dataset,arm=arm,variant=variant,primary_mse=primary,
                        panel_mse=float(np.mean(panel_errors)),jacobian_error=float(np.mean(jac_errors))))
                for variant in ('drop_radial','drop_tangent'):
                    model.variant=variant
                    for block in (0,1,2):
                        model.block_index=block
                        for time,panel in zip(TIMES,saved_panels):
                            with torch.no_grad():value=model(torch.from_numpy(panel['x']).double(),torch.from_numpy(panel['t']).double()).numpy()
                            assert np.isfinite(value).all()
                            error=float(np.mean((value-panel['oracle'])**2))
                            panels_rows.append(dict(dataset=dataset,arm=arm,variant=variant,block=str(block),time=time,field_mse=error,jacobian_error=None))
                            np.savez_compressed(folder/f'{variant}-block-{block}-panel-{time:.1f}.npz',prediction=value)
                progress['completed']+=1;progress['latest']=[dataset,arm];write(output/'status.json',progress)
                process=psutil.Process();cpu=process.cpu_times()
                with (output/'monitor.jsonl').open('a') as handle:handle.write(json.dumps(dict(completed=progress['completed'],pid=os.getpid(),cpu_seconds=cpu.user+cpu.system,rss_bytes=process.memory_info().rss,available_memory_bytes=psutil.virtual_memory().available))+'\n')
                print(json.dumps(progress),flush=True)
    assert progress['completed']==9
    csv_write(output/'metrics.csv',metrics);csv_write(output/'panels.csv',panels_rows);csv_write(output/'traces.csv',traces)
    for path,value in bindings['inputs'].items():assert sha(path)==value
    assert sha(Path(__file__))==bindings['script_sha256']
    assert sha(ROOT/'synthetic/normalized_geometry.py')==bindings['geometry_module_sha256']
    assert sha(ROOT/DOC/'protocol.md')==bindings['protocol_sha256']
    write(output/'provenance.json',bindings)
    write(output/'status.json',dict(progress,state='COMPLETE',metric_rows=len(metrics),panel_rows=len(panels_rows),trace_rows=len(traces)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,nargs='+',required=True);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:run([p.resolve() for p in args.inputs],args.source.resolve(),args.output.resolve())
    except BaseException as error:
        path=args.output.resolve()/'status.json'
        if path.exists():
            progress=load(path)
            if progress.get('pid')==os.getpid() and progress.get('state')=='RUNNING':write(path,dict(progress,state='FAILED',error=repr(error)))
        raise
