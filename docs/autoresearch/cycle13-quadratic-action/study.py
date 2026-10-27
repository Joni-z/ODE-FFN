"""Registered frozen K2 action envelopes and one whole-model intervention."""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil
import torch

FAMILIES=('ode_k2','frame_k2','global_k2')
CHECKPOINTS=(0,1000,2500,5000)


def load(path): return json.loads(Path(path).read_text())
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def table(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def setup(root):
    settings=load(root/'settings.json');parent=Path(settings['parent'])
    if 'path_study' not in sys.modules:
        spec=importlib.util.spec_from_file_location('path_study',parent/'source/path_study.py')
        module=importlib.util.module_from_spec(spec);sys.modules['path_study']=module;spec.loader.exec_module(module)
    prior=sys.modules['path_study'];assert Path(prior.__file__).resolve()==parent/'source/path_study.py'
    _,frame,frame_settings,geometry,native,core=prior.setup(parent)
    return settings,prior,frame,frame_settings,geometry,native,core


def verify_inputs(root):
    settings,prior,_,_,_,native,_=setup(root)
    for name,digest in load(root/'bindings.json').items(): assert native.sha(Path(name))==digest,name
    parent=Path(settings['parent']);prior.verify_inputs(parent)
    for name,digest in load(parent/'sealed-runtime.json').items(): assert native.sha(parent/name)==digest,name


def prepare(root,parent):
    root.mkdir(exist_ok=False);source=root/'source';source.mkdir()
    names=('study.py','action_core.py','verify_action.py','protocol.md','theory.md','research-question-card.md')
    for name in names: shutil.copyfile(Path(__file__).parent/name,source/name)
    repo=Path(__file__).resolve().parents[3]
    write(root/'settings.json',dict(parent=str(parent.resolve()),
        registration_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        families=FAMILIES,checkpoints=CHECKPOINTS,field_rows=20000,native_batch=1024,centre=.6))
    _,_,_,_,_,native,_=setup(root)
    paths=[p for p in source.iterdir() if p.is_file()]+[root/'settings.json',parent/'sealed-runtime.json']
    write(root/'bindings.json',{str(p):native.sha(p) for p in paths});verify_inputs(root)
    print(dict(prepared=str(root),bindings=len(paths),new_training=False),flush=True)


def restore(root,law,family,step):
    settings,prior,_,_,_,native,_=setup(root)
    model,folder,_=prior.model_and_folder(Path(settings['parent']),law,family)
    state=torch.load(folder/f'step_{step:07d}.pt',weights_only=False)
    native.restore_model(model,state);model.eval()
    assert len(list(model.ode_layers()))==6
    return model,folder


@torch.no_grad()
def predict(model,field):
    return np.concatenate([model(torch.from_numpy(field['x'][i:i+1024]),
        torch.from_numpy(field['t'][i:i+1024])).numpy() for i in range(0,len(field['x']),1024)])


def preflight(root):
    from verify_action import verify
    verify_inputs(root);settings,_,_,old,_,_,_=setup(root)
    math=verify();started=time.monotonic();checks=[]
    for entry in old['laws']:
        with np.load(Path(entry['banks'])/'field.npz') as data: field={k:data[k] for k in ('x','t')}
        assert field['x'].shape==(20000,128)
        for family in FAMILIES:
            for step in CHECKPOINTS:
                model,folder=restore(root,entry['law'],family,step);prediction=predict(model,field)
                with np.load(folder/f'field_{step:07d}.npz') as data: np.testing.assert_array_equal(prediction,data['prediction'])
                checks.append(dict(law=entry['law'],family=family,step=step,native_field_bitwise=True))
                assert time.monotonic()-started<600
        print(dict(preflight_law=entry['law'],checkpoints=len(checks)),flush=True)
    verify_inputs(root);write(root/'preflight.json',dict(status='PASS',mathematics=math,checks=checks,
        pid=os.getpid(),elapsed_seconds=time.monotonic()-started))
    _,_,_,_,_,native,_=setup(root)
    write(root/'ready.json',dict(bindings=native.sha(root/'bindings.json'),preflight=native.sha(root/'preflight.json')))
    print(dict(status='PREFLIGHT_PASS',checkpoints=len(checks),mathematics=math),flush=True)


def controls(layer,x):
    if hasattr(layer,'frame_mode'):
        s,g,_=layer.controls(x);return s,g
    return layer.duration_from_logit(layer.t_func(x)),None


@torch.no_grad()
def local(layer,inputs):
    from action_core import actions
    double=copy.deepcopy(layer).double();x=torch.from_numpy(inputs).double()
    s,g=controls(double,x)
    samples,output=actions(double.proj.weight.numpy(),x.numpy(),s.numpy(),None if g is None else g.numpy())
    reference=double(x).numpy()
    np.testing.assert_allclose(output,reference,atol=1e-10,rtol=1e-10)
    return samples,output


def attach_intervention(model):
    # All original controls are recomputed at the intervention's actual input.
    handles=[]
    def substitute(layer,args,output):
        x=args[0];s,g=controls(layer,x)
        if g is None: z=x
        else:
            operator=sys.modules[type(layer).__module__];z=operator.rotate(x,g,transpose=True)
        u=torch.nn.functional.linear(z,layer.proj.weight)
        w=torch.nn.functional.linear(u,layer.proj.weight);value=s*(u+.3*w)
        if g is not None: value=operator.rotate(value,g)
        return x+value
    for _,layer in model.ode_layers(): handles.append(layer.register_forward_hook(substitute))
    return handles


@torch.no_grad()
def run(root):
    from action_core import summarize,KEYS
    started=time.monotonic();verify_inputs(root)
    settings,_,_,old,_,native,_=setup(root)
    assert load(root/'ready.json')==dict(bindings=native.sha(root/'bindings.json'),preflight=native.sha(root/'preflight.json'))
    out=root/'output';out.mkdir(exist_ok=False);raw=out/'raw';raw.mkdir()
    rows=[];complete=[];spectra=[];telemetry=[]
    write(out/'invocation.json',dict(executable=sys.executable,argv=sys.argv,pid=os.getpid(),settings=settings))
    def status(label='',done=False):
        row=dict(state='COMPLETE' if done else 'RUNNING',completed=len(complete),total=24,latest=label,
                 pid=os.getpid(),elapsed_seconds=time.monotonic()-started,rss_bytes=psutil.Process().memory_info().rss)
        assert row['elapsed_seconds']<600
        write(out/'status.json',row);telemetry.append(row)
    status()
    for entry in old['laws']:
        law=entry['law'];banks=Path(entry['banks'])
        with np.load(banks/'field.npz') as data: field={k:data[k] for k in ('x','t','oracle')}
        for family in FAMILIES:
            for step in CHECKPOINTS:
                model,folder=restore(root,law,family,step);layers=list(model.ode_layers())
                captured={name:[] for name,_ in layers};outputs={name:[] for name,_ in layers};handles=[]
                for name,layer in layers:
                    def hook(module,args,output,key=name):
                        captured[key].append(args[0].numpy().copy());outputs[key].append(output.numpy().copy())
                    handles.append(layer.register_forward_hook(hook))
                prediction=predict(model,field)
                for handle in handles: handle.remove()
                with np.load(folder/f'field_{step:07d}.npz') as data: np.testing.assert_array_equal(prediction,data['prediction'])
                saved={r['layer']:r for r in load(folder/f'generator_{step:07d}.json')}
                clocks=native.legacy.visit_diagnostics(model,banks);stored=load(folder/f'clocks_{step:07d}.json')
                assert len(clocks)==len(stored)==6
                for a,b in zip(clocks,stored):
                    assert a['layer']==b['layer'] and a['visit']==b['visit']
                    for k,v in a['activation'].items():
                        if isinstance(v,(float,int)): np.testing.assert_allclose(v,b['activation'][k],atol=1e-9,rtol=1e-8)
                        else: assert v==b['activation'][k]
                arrays={}
                for index,(name,layer) in enumerate(layers):
                    x=np.concatenate(captured[name]);native_output=np.concatenate(outputs[name])
                    assert x.shape==(20000,64)
                    samples,promoted=local(layer,x)
                    np.testing.assert_allclose(promoted,native_output,atol=1e-5,rtol=1e-5)
                    difference=float(np.max(np.abs(promoted-native_output)))
                    singular=np.linalg.svd(layer.proj.weight.double().numpy(),compute_uv=False)
                    np.testing.assert_allclose(singular,saved[name]['P_singular_values'],rtol=1e-8,atol=1e-8)
                    spectra.append(dict(law=law,family=family,step=step,layer=name,
                        sigma_max=float(singular[0]),sigma_min=float(singular[-1]),singular_cv=float(singular.std()/singular.mean())))
                    arrays[f'input_{index}']=x;arrays[f'samples_{index}']=samples
                    for group in range(-1,4):
                        mask=np.ones(len(x),dtype=bool) if group<0 else (field['t'][:,0]>=group/4)&(field['t'][:,0]<(group+1)/4)
                        assert mask.any()
                        rows.append(dict(law=law,family=family,step=step,layer=name,
                            time_bin='pooled' if group<0 else f'{group/4:.2f}-{(group+1)/4:.2f}',
                            promoted_vs_native_output_max=difference,**summarize(samples[mask])))
                np.savez_compressed(raw/f'{law}-{family}-{step:07d}.npz',**arrays)
                handles=attach_intervention(model);changed=predict(model,field)
                for handle in handles: handle.remove()
                delta=changed.astype(float)-prediction;error=prediction.astype(float)-field['oracle']
                displacement=float(np.mean(delta*delta));cross=float(2*np.mean(error*delta))
                risk=float(np.mean(error*error));newrisk=float(np.mean((changed.astype(float)-field['oracle'])**2))
                np.testing.assert_allclose(newrisk-risk,displacement+cross,atol=1e-10,rtol=1e-8)
                denominator=float(np.mean(prediction.astype(float)**2))
                relative=float(np.sqrt(displacement/denominator)) if denominator>0 else None
                complete.append(dict(law=law,family=family,step=step,original_field_mse=risk,changed_field_mse=newrisk,
                    displacement_mse=displacement,cross_term=cross,relative_displacement_rms=relative,
                    displacement_below_one_percent=relative<=.01 if relative is not None else None,native_field_bitwise=True))
                np.savez_compressed(raw/f'{law}-{family}-{step:07d}-predictions.npz',native=prediction,changed=changed)
                status(f'{law}/{family}/{step}')
                print(dict(completed=len(complete),law=law,family=family,step=step,elapsed=time.monotonic()-started),flush=True)
    table(out/'local.csv',rows);table(out/'complete.csv',complete);table(out/'P-history.csv',spectra)
    write(out/'sample-columns.json',list(KEYS));verify_inputs(root);status(done=True);table(out/'telemetry.csv',telemetry)
    write(out/'receipt.json',dict(status='PASS',pid=os.getpid(),elapsed_seconds=time.monotonic()-started,
        peak_sampled_rss_gib=max(r['rss_bytes'] for r in telemetry)/2**30,checkpoints=len(complete),layers=len(spectra),
        local_rows=len(rows),source_rows_per_checkpoint=20000,registration=settings['registration_commit'],new_training=False))
    print(dict(status='COMPLETE',checkpoints=len(complete),elapsed=time.monotonic()-started),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','preflight','run'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--parent',type=Path)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.root.resolve(),args.parent.resolve())
    else: globals()[args.command](args.root.resolve())
