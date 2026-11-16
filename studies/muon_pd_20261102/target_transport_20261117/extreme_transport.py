"""Small-perturbation and outer-resolution checks on prebound cached sources."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import pickle
import time

import numpy as np
from common import base,ev,torch,read,dump,restored_pair
from learned_transport import BlendedField,SensitivityField

STEPS = (127,128,129,256,512)
EPSILONS = (1e-4,1e-5,1e-6,1e-7)


def check(actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> dict:
    error = np.abs(actual-expected)
    finite = np.isfinite(actual)&np.isfinite(expected)
    ok = finite&(error<=atol+rtol*np.abs(expected))
    return dict(qualified=bool(ok.all()),failed_components=int((~ok).sum()),
        nonfinite_components=int((~finite).sum()),
        max_error=float(error.max()) if np.isfinite(error).all() else None)


def rng_hashes() -> dict:
    return dict(torch=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest())


def solve_pair(native,crossed,source,steps):
    primal = base.train.generated_samples(native,source,steps)
    alternate = base.train.generated_samples(crossed,source,steps)
    field = SensitivityField(native,crossed)
    augmented = base.train.generated_samples(field,
        torch.cat((source,torch.zeros_like(source)),dim=1),steps)
    width = source.shape[1]
    return primal,alternate,augmented[:,:width],augmented[:,width:],field


class TimeAffine(torch.nn.Module):
    def __init__(self,a,b,c):
        super().__init__();self.a,self.b,self.c = map(torch.tensor,(a,b,c))

    def forward(self,x,t):
        return x@self.a.T+torch.sin(7*t)*(x@self.b.T)+self.c


def math_controls() -> dict:
    rng = np.random.default_rng(0)
    a,b = rng.normal(size=(2,2,2,2))*.07
    c = rng.normal(size=(2,2))*.03
    x = .5*np.cos(np.arange(12).reshape(6,2))
    native,crossed = (TimeAffine(a[i],b[i],c[i]) for i in (0,1))
    rows = []
    for steps in STEPS:
        primal,_,replay,tangent,field = solve_pair(native,crossed,torch.tensor(x),steps)
        y,z = x.copy(),np.zeros_like(x);h=1/steps
        def drift(point,variation,t):
            matrix=a[0]+np.sin(7*t)*b[0]
            difference=a[1]-a[0]+np.sin(7*t)*(b[1]-b[0])
            return point@matrix.T+c[0],variation@matrix.T+point@difference.T+c[1]-c[0]
        for i in range(steps):
            v,dv=drift(y,z,i*h)
            w,dw=drift(y+h*v,z+h*dv,(i+1)*h)
            y,z=y+h/2*(v+w),z+h/2*(dv+dw)
        p=check(primal.numpy(),y,1e-11,1e-11)
        u=check(tangent.numpy(),z,1e-11,1e-11)
        assert p['qualified'] and u['qualified'] and torch.equal(primal,replay)
        assert field.primal_qualified
        rows.append(dict(steps=steps,primal=p,tangent=u,augmented_primal_exact=True))
    finite=[]
    expected=solve_pair(native,crossed,torch.tensor(x),128)[3].numpy()
    for epsilon in EPSILONS:
        plus=base.train.generated_samples(BlendedField(native,crossed,epsilon),torch.tensor(x),128)
        minus=base.train.generated_samples(BlendedField(native,crossed,-epsilon),torch.tensor(x),128)
        observed=((plus-minus)/(2*epsilon)).numpy()
        row=check(observed,expected,1e-8,1e-8);assert row['qualified']
        finite.append(dict(epsilon=epsilon,**row))
    return dict(passed=True,scope='independent time-varying affine recurrence only',
        seed=0,steps=rows,finite_differences=finite)


def evaluate(args) -> None:
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    selection=read(Path(args.input_cases))
    assert base.digest(args.source_cases)==selection['source_cases_sha256']
    assert base.digest(Path(args.input_cases).parent/'extreme-transport-protocol.md')==selection['protocol_sha256']
    case=next(v for v in selection['cases'] if v['case']==args.case and v['split']==args.split)
    assert base.digest(case['file'])==case['sha256']
    original=next(v for v in read(Path(args.source_cases))['cases'] if v['id']==args.case)
    base.verify_sources()
    native32,crossed32,identity=restored_pair(original,args.data_root)
    native,crossed=copy.deepcopy(native32).double(),copy.deepcopy(crossed32).double()
    snapshots=[(model,ev.tensor_snapshot(model)) for model in (native32,crossed32,native,crossed)]
    before_rng=rng_hashes()
    with np.load(case['file']) as z:cached={k:z[k] for k in z.files}
    identities=[(v['bank'],v['index']) for v in case['roles']]
    unique=list(dict.fromkeys(identities));indices=[identities.index(v) for v in unique]
    roles=[dict(**v,unique_row=unique.index(identities[i])) for i,v in enumerate(case['roles'])]
    source=torch.from_numpy(cached['source32'][indices]).double()
    levels=(128,) if args.witness else STEPS
    saved=dict(source32=cached['source32'][indices],steps=np.array(levels),epsilons=np.array(EPSILONS))
    rows=[]
    for steps in levels:
        dump(out/'progress.json',dict(pid=os.getpid(),case=args.case,split=args.split,
            stage='resolution',steps=steps,completed=len(rows),total=len(levels),seconds=time.monotonic()-start))
        primal,alternate,replay,tangent,field=solve_pair(native,crossed,source,steps)
        primal,alternate,replay,tangent=(v.numpy() for v in (primal,alternate,replay,tangent))
        for name,value in [('native',primal),('crossed',alternate),('augmented_native',replay),('tangent',tangent)]:
            saved.setdefault(name,[]).append(value)
        rows.append(dict(steps=steps,augmented_primal_exact=bool(np.array_equal(primal,replay)),
            jvp_primal_qualified=field.primal_qualified,jvp_primal_error=field.primal_max_error,
            all_outputs_finite=bool(all(np.isfinite(v).all() for v in (primal,alternate,tangent)))))
        print('EXTREME_RESOLUTION',args.case,args.split,steps,flush=True)
    for name in ('native','crossed','augmented_native','tangent'):saved[name]=np.stack(saved[name])
    i=list(levels).index(128)
    cache_checks={}
    for name,atol,rtol in [('native',2e-8,2e-8),('crossed',2e-8,2e-8),('tangent',2e-6,1e-5)]:
        cache_checks[name]=check(saved[name][i],cached['cached_'+name][indices],atol,rtol)
    finite=[];plus_rows=[];minus_rows=[]
    for epsilon in EPSILONS:
        dump(out/'progress.json',dict(pid=os.getpid(),case=args.case,split=args.split,
            stage='finite-difference',epsilon=epsilon,completed=len(finite),total=4,seconds=time.monotonic()-start))
        plus=base.train.generated_samples(BlendedField(native,crossed,epsilon),source,128).numpy()
        minus=base.train.generated_samples(BlendedField(native,crossed,-epsilon),source,128).numpy()
        observed=(plus-minus)/(2*epsilon)
        comparisons=[check(saved['tangent'][i,j],observed[j],2e-5,1e-3) for j in range(len(unique))]
        plus_rows.append(plus);minus_rows.append(minus)
        finite.append(dict(epsilon=epsilon,unique_source_comparisons=comparisons))
        print('EXTREME_FINITE_DIFFERENCE',args.case,args.split,epsilon,flush=True)
    saved['fd_plus'],saved['fd_minus']=np.stack(plus_rows),np.stack(minus_rows)
    np.savez_compressed(out/'arrays.npz',**saved)
    for model,snapshot in snapshots:ev.same_tensors(model,snapshot)
    assert before_rng==rng_hashes()
    qualified=all(v['qualified'] for v in cache_checks.values()) and all(
        v['augmented_primal_exact'] and v['jvp_primal_qualified'] and v['all_outputs_finite'] for v in rows)
    result=dict(case=args.case,split=args.split,roles=roles,unique_sources=len(unique),
        witness=args.witness,steps=rows,cache_checks=cache_checks,finite_differences=finite,
        replay_qualified=qualified,parameters_buffers_rng_unchanged=True,
        binding=dict(input_cases_sha256=base.digest(args.input_cases),source_cases_sha256=base.digest(args.source_cases),
            selected_arrays_file=case['file'],selected_arrays_sha256=case['sha256'],
            arrays_sha256=base.digest(out/'arrays.npz'),source_sha256=base.digest(__file__),**identity),
        seconds=time.monotonic()-start)
    dump(out/'summary.json',result)
    dump(out/'COMPLETE.json',dict(passed=True,replay_qualified=qualified,witness=args.witness,
        summary_sha256=base.digest(out/'summary.json')))
    print('EXTREME_TRANSPORT_COMPLETE',args.case,args.split,qualified,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--math',action='store_true')
    for name in ('input-cases','source-cases','data-root','case','split'):parser.add_argument('--'+name)
    parser.add_argument('--output',required=True);parser.add_argument('--witness',action='store_true')
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(1)
    if args.math:
        out=Path(args.output);assert not out.exists()
        result=math_controls();result['source_sha256']=base.digest(__file__)
        dump(out,result);print('EXTREME_TRANSPORT_MATH_PASS',flush=True)
    else:
        assert all(getattr(args,k) for k in ('input_cases','source_cases','data_root','case','split'))
        assert args.split in ('selection','confirmation')
        evaluate(args)


if __name__=='__main__':main()
