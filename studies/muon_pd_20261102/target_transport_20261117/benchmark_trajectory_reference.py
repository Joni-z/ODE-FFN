"""One actual full bank entrance: CPU timing only, not target-quality evidence."""
import argparse
import functools
import os
from pathlib import Path
import resource
import time
from population_resolution import read, dump, sha, load_model


def main():
    p=argparse.ArgumentParser()
    for name in ('root','input','output','execution-proof'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    import numpy as np
    import torch
    from common import base,ev
    from compiled_defect_observer import response_with_step
    from factored_defect_drift import coupled_step
    torch.set_num_threads(1);base.verify_sources()
    done=read(a.input/'COMPLETE.json');assert done['passed'] and not done['witness'] and done['points']==4096
    assert read(a.input/'independent.json')['complete_sha256']==sha(a.input/'COMPLETE.json')
    file=a.input/done['file'];assert sha(file)==done['sha256']
    with np.load(file) as z:x=z['augmented_path'][0,:,:z['native'].shape[1]].copy()
    case=next(c for c in read(a.root/'portable.json')['cases'] if c['id']==done['case'])
    model,_=load_model(a.root,case);snapshot=ev.tensor_snapshot(model);rng=torch.get_rng_state().clone()
    proof=read(a.execution_proof)
    assert proof['passed'] and proof['variant']=='power-reuse' and not proof['witness']
    assert proof['step_source_sha256']==sha(Path(__file__).with_name('factored_defect_drift.py'))
    step=functools.partial(coupled_step,reuse_powers=True)
    x=torch.from_numpy(x);t=torch.zeros((len(x),1),dtype=x.dtype);timings=[];outputs={}
    for count in (128,256):
        before=time.perf_counter()
        p,z=response_with_step(model,x,t,count,case['trained_solver'],case['inference_solver'],step)
        timings.append(dict(reference=count,points=len(x),seconds=time.perf_counter()-before))
        outputs[str(count)]=dict(primal_max=float(p.abs().max()),tangent_max=float(z.abs().max()))
    ev.same_tensors(model,snapshot);assert torch.equal(rng,torch.get_rng_state())
    dump(a.output,dict(passed=True,timing_only=True,device='cpu',threads=1,points=4096,
        case=done['case'],source_main_complete_sha256=sha(a.input/'COMPLETE.json'),
        source_sha256=sha(__file__),timings=timings,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        output_checks=outputs,tensors_rng_unchanged=True))
    print('CPU_REFERENCE_FULL_BATCH_TIMING',timings,flush=True)


if __name__=='__main__':main()
