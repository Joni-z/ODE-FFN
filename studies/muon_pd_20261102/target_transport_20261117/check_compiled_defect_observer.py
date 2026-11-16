"""Validate and time the same FP64 observer with one compiled tensor step."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from population_resolution import read,dump,sha,load_model,select_gpu

OWNED_OUTPUT=None


def main():
    global OWNED_OUTPUT
    p=argparse.ArgumentParser()
    for name in ('root','reference','output','cache'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--witness',action='store_true');p.add_argument('--profile',action='store_true')
    p.add_argument('--case');p.add_argument('--readiness',type=Path)
    a=p.parse_args();assert not a.output.exists()
    physical,uuid=select_gpu(None);os.environ['CUDA_VISIBLE_DEVICES']=uuid
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(a.cache/'inductor')
    os.environ['TRITON_CACHE_DIR']=str(a.cache/'triton')
    os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
    import numpy as np
    import torch
    from common import ev,base
    from gpu_resolution import exclusive_gpu
    import hidden_defect_transport as original
    from compiled_defect_observer import make_compiled_step,response_with_step
    from check_hidden_defect_transport import close
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    base.verify_sources()
    assert torch.cuda.device_count()==1
    a.output.mkdir(parents=True);OWNED_OUTPUT=a.output;torch.cuda.reset_peak_memory_stats()
    assert not torch._dynamo.config.suppress_errors
    guard=lambda:exclusive_gpu(uuid)
    if not a.witness:
        assert a.readiness and read(a.readiness)['passed']
        assert read(a.readiness)['fields_sha256']==sha(a.root/'fields.json')
    canonical=read(a.reference/'COMPLETE.json');assert canonical['passed'] and not canonical['witness']
    assert canonical['all_numerically_qualified'] and canonical['refined_reference']
    reference_rows=read(a.reference/'rows.json');portable=read(a.root/'portable.json')
    rows=[];first_calls=[];timings=[];began=time.monotonic();step=make_compiled_step()
    native_function=original.variation_solve
    selected=[c for c in portable['cases'] if not a.case or c['id']==a.case]
    assert len(selected)==(1 if a.case else 4)

    def progress(case,split,stage):
        dump(a.output/'progress.json',dict(pid=os.getpid(),case=case,split=split,stage=stage,
            completed=len(rows),witness=a.witness,seconds=time.monotonic()-began))
        status=subprocess.check_output(['nvidia-smi','-i',uuid,
            '--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True)
        with (a.output/'gpu-history.jsonl').open('a') as f:f.write(json.dumps(dict(unix_time=time.time(),stage=stage,resource=status.strip()))+'\n')

    for case in selected:
        model,_=load_model(a.root,case);model=model.cuda()
        snapshot=ev.tensor_snapshot(model);methods=[layer.solve.__func__ for _,layer in model.ode_layers()]
        rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
        n,c=case['trained_solver'],case['inference_solver']
        for split in ('selection','confirmation'):
            ref=next(v for v in reference_rows if (v['case'],v['split'])==(case['id'],split))
            file=a.reference/ref['file'];assert sha(file)==ref['sha256']
            count=8 if a.witness else 4096
            with np.load(file) as z:
                x=torch.from_numpy(z['x'][:count].copy()).cuda()
                t=torch.from_numpy(z['time'][:count].copy()).cuda()
                expected=[torch.from_numpy(z[k][:count].copy()).cuda() for k in ('primal256','tangent256')]
                profile_inputs=[torch.from_numpy(z[k].copy()).cuda() for k in ('x','time')]
            with torch.no_grad():native_before=model(x,t).clone()
            if a.profile and not (a.output/'eager-profile.txt').exists():
                guard();progress(case['id'],split,'profile-eager-full-batch-reference16')
                with torch.random.fork_rng(devices=[0]), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]) as profile:
                    original.model_response(model,*profile_inputs,16,n,c)
                    torch.cuda.synchronize()
                (a.output/'eager-profile.txt').write_text(profile.key_averages().table(sort_by='self_cpu_time_total',row_limit=30))
                dump(a.output/'eager-profile.json',dict(reference_steps=16,batch=4096,
                    events=[dict(key=v.key,count=v.count,self_cpu_us=v.self_cpu_time_total,
                        self_device_us=getattr(v,'self_device_time_total',None)) for v in profile.key_averages()]))
            guard();progress(case['id'],split,'compiled-reference256')
            with torch.random.fork_rng(devices=[0]):
                torch.cuda.synchronize();start=time.perf_counter()
                value=response_with_step(model,x,t,256,n,c,step)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
            first_calls.append(dict(case=case['id'],split=split,points=count,first_call_includes_trace_compile_and_execution_seconds=elapsed))
            errors=[close(v,r,1e-10,1e-9 if i==0 else 1e-8) for i,(v,r) in enumerate(zip(value,expected,strict=True))]
            assert all(v.dtype==torch.float64 for v in value)
            progress(case['id'],split,'independent-prefix-controls')
            xx,tt=x[:8],t[:8]
            # New shape/step/solver-order compilation is isolated from experimental RNG.
            with torch.random.fork_rng(devices=[0]):
                primal,tangent=response_with_step(model,xx,tt,16,n,c,step)
                e=torch.tensor(0.,device='cuda',dtype=torch.float64)
                with torch.no_grad():
                    jp,jz=torch.func.jvp(lambda eps:original.forced_model(model,xx,tt,16,eps,n,c),(e,),(torch.ones_like(e),))
                    jvp=[close(primal,jp,1e-11,1e-11),close(tangent,jz,1e-11,1e-11)]
                    finite=[]
                    for epsilon in (1e-4,1e-5):
                        e=torch.tensor(epsilon,device='cuda',dtype=torch.float64)
                        plus=original.forced_model(model,xx,tt,16,e,n,c)
                        minus=original.forced_model(model,xx,tt,16,-e,n,c)
                        finite.append(dict(epsilon=epsilon,error=close(tangent,(plus-minus)/(2*e),2e-7,2e-4)))
                _,zero=response_with_step(model,xx,tt,16,n,n,step);assert not torch.count_nonzero(zero)
                reverse_p,reverse_z=response_with_step(model,xx,tt,16,c,n,step)
                reverse=[close(reverse_p,primal,1e-12,0),close(reverse_z,-tangent,1e-12,0)]
            if not a.witness and case['id']=='product_rk4_trained' and split=='confirmation':
                for repetition in range(3):
                    progress(case['id'],split,'paired-timing-'+str(repetition));guard()
                    order=('eager','compiled') if repetition%2==0 else ('compiled','eager')
                    record=dict(repetition=repetition,order=order)
                    for mode in order:
                        torch.cuda.synchronize();start=time.perf_counter()
                        current=(original.model_response(model,x,t,256,n,c) if mode=='eager' else
                                 response_with_step(model,x,t,256,n,c,step))
                        torch.cuda.synchronize();record[mode+'_seconds']=time.perf_counter()-start
                        for i,(v,r) in enumerate(zip(current,expected,strict=True)):
                            close(v,r,1e-10,1e-9 if i==0 else 1e-8)
                    timings.append(record);dump(a.output/'timings.json',timings)
            assert original.variation_solve is native_function
            assert methods==[layer.solve.__func__ for _,layer in model.ode_layers()]
            with torch.no_grad():assert torch.equal(native_before,model(x,t))
            name=case['id']+'-'+split+'.npz'
            np.savez_compressed(a.output/name,primal=value[0].cpu().numpy(),tangent=value[1].cpu().numpy())
            rows.append(dict(case=case['id'],split=split,points=count,primal_error=errors[0],tangent_error=errors[1],
                jvp_errors=jvp,finite_differences=finite,reverse_errors=reverse,zero_exact=True,native_replay_exact=True,
                file=name,sha256=sha(a.output/name),reference_sha256=ref['sha256']))
            dump(a.output/'rows.json',rows);print('COMPILED_OBSERVER_CELL',case['id'],split,count,errors,flush=True)
        ev.same_tensors(model,snapshot)
        assert torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    counters={k:{str(x):int(y) for x,y in v.items()} for k,v in torch._dynamo.utils.counters.items()}
    dump(a.output/'compiler-counters.json',counters)
    (a.output/'compiler-times.txt').write_text(torch._dynamo.utils.compile_times())
    assert not counters.get('graph_break') and not counters.get('unimplemented'),counters
    assert counters.get('stats',{}).get('unique_graphs',0)>0,'No compiled graph was observed'
    speed=None
    if timings:
        eager=statistics.median(v['eager_seconds'] for v in timings)
        fast=statistics.median(v['compiled_seconds'] for v in timings);speed=eager/fast
    result=dict(passed=True,witness=a.witness,rows=len(rows),physical_gpu=physical,gpu_uuid=uuid,
        fields_sha256=sha(a.root/'fields.json'),step_source_sha256=sha(Path(__file__).with_name('compiled_defect_observer.py')),
        source_sha256=sha(__file__),reference_complete_sha256=sha(a.reference/'COMPLETE.json'),
        first_calls=first_calls,timings=timings,median_speedup=speed,
        peak_memory_bytes=torch.cuda.max_memory_allocated(),tensors_methods_rng_unchanged=True,
        source_observer_unchanged=True,seconds=time.monotonic()-began)
    dump(a.output/'COMPLETE.json',result);print('COMPILED_OBSERVER_PASS',len(rows),speed,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        import sys,traceback
        if OWNED_OUTPUT is not None:
            dump(OWNED_OUTPUT/'FAILED.json',dict(passed=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),source_sha256=sha(__file__)))
        raise
