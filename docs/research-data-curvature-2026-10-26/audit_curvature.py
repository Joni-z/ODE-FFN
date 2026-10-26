"""Independent native-composition checks of every fixed curvature panel."""
import copy
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(sys.argv[1])
sys.path.insert(0,str(ROOT/'source'))
import curvature_study as study
from curvature_math import local,left,frame,reduced
from teacher_support import TEACHERS,STUDENTS,load,write,append,sha,teacher,signature


def captured_prefix(raw: torch.nn.Module,x: torch.Tensor,t: torch.Tensor) -> torch.Tensor:
    """Use the actual complete backbone and pre-hooks, preserving derivatives."""
    records,handles = [],[]
    def hook(module,args):
        records.append(args[0][0])
    try:
        for block in raw.blocks:
            handles.append(block.ffn.register_forward_pre_hook(hook))
        raw(x[None],t[None])
        assert len(records)==3
        return torch.stack(records)
    finally:
        for handle in handles:
            handle.remove()


def close(actual,expected) -> None:
    np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),rtol=2e-8,atol=5e-10)


def audit() -> None:
    torch.set_num_threads(1)
    assert load(ROOT/'output/status.json')['state']=='COMPLETE'
    ready = load(ROOT/'ready.json')
    for path,digest in {**ready['bindings'],**ready['prior_documents']}.items():
        assert sha(path)==digest,path
    started = time.monotonic()
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            value=study.observe();append(ROOT/'audit-monitor.jsonl',value)
            stop.wait(5)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        with np.load(ROOT/'output/cache/prefix.npz') as file:
            bank={key:file[key] for key in file.files}
        assert len(bank['x'])==80
        metadata=load(ROOT/'output/cache/metadata.json')
        assert metadata['prefix_sha256']==sha(ROOT/'output/cache/prefix.npz')
        raw=copy.deepcopy(teacher('swiglu128')[0].raw).double()
        raw_signature=signature(raw)
        maxima=dict(prefix_value=0.,prefix_jacobian=0.,prefix_hessian=0.,value=0.,jacobian=0.,hessian=0.,local_hessian=0.,frozen_hessian=0.,reduced_identity=0.)
        for index in range(80):
            x,t=torch.from_numpy(bank['x'][index]),torch.from_numpy(bank['t'][index])
            function=lambda state:captured_prefix(raw,state,t)
            value=function(x).detach();jac=torch.func.jacrev(function)(x).detach()
            hessian=torch.func.jacfwd(torch.func.jacrev(function))(x).detach()
            for key,actual,expected in [('prefix_value',value,bank['z'][index]),('prefix_jacobian',jac,bank['jacobian'][index]),('prefix_hessian',hessian,bank['hessian'][index])]:
                close(actual.numpy(),expected);maxima[key]=max(maxima[key],float(np.max(np.abs(actual.numpy()-expected))))
        print(dict(stage='native_prefix_replay',anchors=80),flush=True)
        counts=dict(functions=0,complete_hessians=0,local_hessians=0,frozen_hessians=0,reduced_identities=0)
        for name in TEACHERS:
            for position in range(3):
                folder=ROOT/'output/cases'/f'{name}--{position}'
                assert load(folder/'status.json')['state']=='COMPLETE'
                models=study.subjects(name,position)
                before={key:signature(module) for key,(module,_) in models.items()}
                provenance=load(folder/'provenance.json')
                assert before==provenance['signatures']
                assert provenance['threads']==1 and provenance['device']=='cpu' and provenance['anchors']==80
                native_locals={}
                for key,(module,family) in models.items():
                    with np.load(folder/(key+'.npz')) as file:
                        saved={k:file[k] for k in file.files}
                    geometry=load(folder/(key+'-geometry.json'))
                    native_locals[key]=[]
                    for index in range(80):
                        x,t=torch.from_numpy(bank['x'][index]),torch.from_numpy(bank['t'][index])
                        z,u,hz=(torch.from_numpy(bank[k][index,position]) for k in ('z','jacobian','hessian'))
                        function=lambda state:module(captured_prefix(raw,state,t)[position][None])[0]
                        value=function(x).detach();jac=torch.func.jacrev(function)(x).detach()
                        hessian=torch.func.jacfwd(torch.func.jacrev(function))(x).detach()
                        local_function=lambda delta:module((z+u@delta)[None])[0]
                        hlocal=torch.func.jacfwd(torch.func.jacrev(local_function))(torch.zeros(16,dtype=torch.float64)).detach()
                        for field,actual,expected in [('value',value,saved['value'][index]),('jacobian',jac,saved['jacobian'][index]),
                                                      ('hessian',hessian,saved['hessian_total'][index]),('local_hessian',hlocal,saved['hessian_local'][index])]:
                            close(actual.numpy(),expected)
                            maxima[field]=max(maxima[field],float(np.max(np.abs(actual.numpy()-expected))))
                        native_locals[key].append(hlocal)
                        counts['complete_hessians']+=1;counts['local_hessians']+=1
                        if family=='ode_k2':
                            with torch.no_grad():
                                data=local(module,z,u,family);definition=frame(data,u)
                                first,second=data['first'],data['second']
                                ah=reduced(hlocal,definition);bh=reduced(data['frozen_hessian'],definition)
                                difference=float((ah-bh).abs().max());maxima['reduced_identity']=max(maxima['reduced_identity'],difference)
                                close(ah.numpy(),bh.numpy());counts['reduced_identities']+=1
                                row=geometry[index]
                                close(row['full_curvature_energy'],hessian.square().sum().item())
                                close(row['local_curvature_energy'],hlocal.square().sum().item())
                                close(saved['route_grams'][index].sum(),hessian.square().sum().item())
                                expected_frozen=data['frozen_hessian']+left(data['frozen_jac'],hz)
                            a1,a2=first['a'].detach(),second['a'].detach()
                            def frozen(state):
                                h=captured_prefix(raw,state,t)[position]
                                for layer,a,activate in [(module.ode1,a1,True),(module.ode2,a2,False)]:
                                    p1=layer.proj(h)*a;p2=layer.proj(p1)*a/2
                                    h=h+p1+p2
                                    if activate:h=torch.nn.functional.silu(h)
                                return h
                            h0=torch.func.jacfwd(torch.func.jacrev(frozen))(x).detach()
                            close(h0.numpy(),expected_frozen.numpy())
                            maxima['frozen_hessian']=max(maxima['frozen_hessian'],float((h0-expected_frozen).abs().max()))
                            close(row['frozen_difference_energy'],(hessian-h0).square().sum().item())
                            counts['frozen_hessians']+=1
                    counts['functions']+=1
                    print(dict(stage='native_function_replay',teacher=name,position=position,subject=key,functions=counts['functions']),flush=True)
                projected=load(folder/'projected-errors.json')
                for index in range(80):
                    z,u=(torch.from_numpy(bank[k][index,position]) for k in ('z','jacobian'))
                    with torch.no_grad():
                        data=local(models['ode_k2'][0],z,u,'ode_k2');definition=frame(data,u)
                        truth=reduced(native_locals['target'][index],definition)
                        row=projected[index];close(row['teacher_retained_energy'],truth.square().sum().item())
                        for family in STUDENTS:
                            difference=reduced(native_locals[family][index],definition)-truth
                            close(row[family+'_error_energy'],difference.square().sum().item())
                assert before=={key:signature(module) for key,(module,_) in models.items()}
        assert signature(raw)==raw_signature
        for path,digest in {**ready['bindings'],**ready['prior_documents']}.items():assert sha(path)==digest,path
        receipt=dict(status='PASS',**counts,prefix_anchors=80,maxima=maxima,
                     input_bindings=len(ready['bindings']),preserved_prior_files=len(ready['prior_documents']),
                     elapsed_seconds=time.monotonic()-started,pid=os.getpid(),ready_sha256=sha(ROOT/'ready.json'),
                     source_sha256=sha(__file__))
        write(ROOT/'audit.json',receipt);print(receipt,flush=True)
    finally:
        stop.set();watcher.join(timeout=6)


if __name__=='__main__':
    audit()
