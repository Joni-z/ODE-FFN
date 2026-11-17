"""Read-only complete-group audit; never pool fewer than the eight original banks."""
import argparse
import json
from pathlib import Path
import numpy as np
from population_resolution import read, sha
from verify_analytic_target import reconstruct


def audit(root, case, split, steps):
    values={k:[] for k in ('native','native_reference','crossed_reference','analytic_tangent','measured_tangent','target')}
    directions=None;rows=[];bindings=[]
    for bank in range(8):
        folder=root/f'{case}-{split}-bank{bank}-heun{steps}'
        done=read(folder/'COMPLETE.json');ind=read(folder/'independent.json')
        assert done['passed'] and not done['witness'] and done['points']==4096
        assert (done['case'],done['split'],done['bank'],done['steps'])==(case,split,bank,steps)
        assert done['hidden_reference_steps']==256 and done['variant']=='power-reuse'
        assert ind['passed'] and ind['complete_sha256']==sha(folder/'COMPLETE.json')
        file=folder/done['file'];assert sha(file)==done['sha256']
        with np.load(file) as z:
            for key in values:values[key].append(z[key].copy())
            if directions is None:directions=z['directions'].copy()
            else:assert np.array_equal(directions,z['directions'])
        rows.append(done)
        bindings.append(dict(bank=bank,complete_sha256=sha(folder/'COMPLETE.json'),array_sha256=sha(file),
                             independent_sha256=sha(folder/'independent.json')))
    for key in ('source_sha256','kernel_sha256','execution_proof_sha256','portable_sha256'):
        assert len({row[key] for row in rows})==1,key
    values={k:np.concatenate(v) for k,v in values.items()}
    x,y=values['native'],values['target']
    loss0,_=reconstruct(values['native_reference'],y,directions)
    loss1,_=reconstruct(values['crossed_reference'],y,directions)
    actual=loss1-loss0
    ties=sum(int(np.count_nonzero(np.diff(np.sort(x@d))==0)) for d in directions)
    result={}
    for name in ('analytic','measured'):
        u=values[name+'_tangent'];loss,predicted=reconstruct(x,y,directions,u)
        finite=[]
        for epsilon in (1e-4,1e-5):
            plus,_=reconstruct(x+epsilon*u,y,directions)
            minus,_=reconstruct(x-epsilon*u,y,directions)
            estimate=(plus-loss)/epsilon if ties else (plus-minus)/(2*epsilon)
            error=abs(estimate-predicted)
            finite.append(dict(epsilon=epsilon,estimate=estimate,error=error,
                               qualified=bool(error<=1e-6+1e-3*abs(predicted))))
        endpoint_loss,_=reconstruct(x+u,y,directions)
        displacement=values['crossed_reference']-values['native_reference']
        signal=float(np.mean(displacement**2));residual=float(np.mean((u-displacement)**2))
        result[name]=dict(predicted_squared_loss_change=predicted,
            score=None if actual==0 else 1-(predicted-actual)**2/actual**2,
            sign_agrees=bool(np.sign(predicted)==np.sign(actual)),finite_differences=finite,
            linearized_endpoint_loss=endpoint_loss,
            endpoint_score=None if signal==0 else 1-residual/signal,
            loss_from_endpoint_remainder=loss1-endpoint_loss,
            functional_remainder=endpoint_loss-loss0-predicted,
            total_linear_loss_error=actual-predicted)
    return dict(case=case,split=split,steps=steps,banks=8,points=32768,directions=len(directions),
        native_sw2=float(np.sqrt(loss0)),crossed_sw2=float(np.sqrt(loss1)),actual_squared_loss_change=actual,
        predictions=result,all_main_prefix_checks_qualify=all(r['numerically_qualified'] for r in rows),
        all_source_reference_status='separate required follow-up; not inferred from prefix checks',
        bank_bindings=bindings,source_sha256=sha(Path(__file__)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--case',required=True)
    p.add_argument('--split',choices=('selection','confirmation'),required=True)
    p.add_argument('--steps',type=int,choices=(128,2048),required=True)
    a=p.parse_args();print(json.dumps(audit(a.root,a.case,a.split,a.steps),indent=2,allow_nan=False))


if __name__=='__main__':main()
