"""All prebound learned states: original drift, NumPy jets and six-product form."""
import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
from population_resolution import read,dump,sha,load_model


def main():
    p=argparse.ArgumentParser()
    for name in ('root','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    import torch
    import torch.nn.functional as F
    import hidden_defect_transport as old
    from factored_defect_drift import drift
    from rk4_defect import local_defects
    torch.set_num_threads(1);rows=[];maximum=0.;jet_maximum=0.
    linear=F.linear
    def counted(fn):
        calls=[]
        def wrapped(*args,**kwargs):calls.append(1);return linear(*args,**kwargs)
        with patch.object(F,'linear',new=wrapped):value=fn()
        return value,len(calls)
    def original(y,z,matrix,n,c):
        return (F.linear(old.response(y),matrix),F.linear((1+1.2*y.square()).pow(-1.5)*z,matrix)+old.forcing(y,matrix,n,c))
    with torch.no_grad():
        for case in read(a.root/'portable.json')['cases']:
            model,_=load_model(a.root,case)
            for split in ('selection','confirmation'):
                with np.load(a.reference/(case['id']+'-'+split)/'continuous_tight.npz') as data:entrances=data['entrances'].copy()
                for i,(_,layer) in enumerate(model.ode_layers()):
                    y=torch.from_numpy(entrances[i,:16]);z=.01*torch.cos(torch.arange(y.numel(),dtype=y.dtype)).reshape_as(y)
                    matrix=layer.operator();jets=local_defects(y.numpy(),matrix.numpy());counts={}
                    for n,c in [('midpoint','rk4'),('rk4','midpoint'),('midpoint','midpoint'),('rk4','rk4')]:
                        expected,old_count=counted(lambda:original(y,z,matrix,n,c))
                        jet=(.0625*(jets[c]-jets[n]))
                        ref=(expected[0].numpy(),((1+1.2*y.numpy()**2)**(-1.5)*z.numpy())@matrix.numpy().T+jet)
                        for v,w in zip(expected,ref,strict=True):
                            error=np.abs(v.numpy()-w);assert np.all(error<=1e-12+1e-10*np.abs(w));jet_maximum=max(jet_maximum,float(error.max()))
                        for reuse in (False,True):
                            result,count=counted(lambda:drift(y,z,matrix,n,c,reuse))
                            assert old_count==(2 if n==c else 19) and count==(2 if n==c else 6)
                            for value,reference in zip(result,expected,strict=True):
                                error=(value-reference).abs();assert bool(torch.all(error<=1e-12+1e-10*reference.abs()))
                                maximum=max(maximum,float(error.max()))
                            counts[n+'->'+c+('/reuse' if reuse else '/pow')]=dict(original=old_count,factored=count)
                    rows.append(dict(case=case['id'],split=split,layer=i,points=16,linear_calls=counts))
    dump(a.output,dict(passed=True,states=768,layer_cells=48,variants=2,solver_pairs=4,
        max_factorization_error=maximum,max_independent_jet_error=jet_maximum,rows=rows,
        source_sha256=sha(__file__),drift_source_sha256=sha(Path(__file__).with_name('factored_defect_drift.py')),
        equation_changed=False,new_training=0,new_draws=0))
    print('SIX_PRODUCT_DRIFT_PASS',768,maximum,jet_maximum,flush=True)


if __name__=='__main__':main()
