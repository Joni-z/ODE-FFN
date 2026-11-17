"""Cached-only diagnosis of a retained analytic-target metric-check failure."""
import argparse
from pathlib import Path
import numpy as np
from population_resolution import read, dump, sha
from diagnose_metric_rank import diagnose, torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(1)
    done=read(a.input/'COMPLETE.json');assert done['passed']
    independent=read(a.input/'independent.json')
    assert independent['passed'] and independent['complete_sha256']==sha(a.input/'COMPLETE.json')
    file=a.input/done['file'];assert sha(file)==done['sha256']
    with np.load(file) as z:
        values={k:z[k].copy() for k in ('native','analytic_tangent','measured_tangent','target','directions')}
    rows={}
    for name in ('analytic','measured'):
        original=done['predictions'][name]['derivative']
        d=diagnose(values['native'],values[name+'_tangent'],values['target'],values['directions'])
        assert abs(d['loss']-original['loss'])<=1e-12*(1+abs(original['loss']))
        assert abs(d['derivative']-original['loss_derivative'])<=1e-12*(1+abs(original['loss_derivative']))
        for recorded in original['finite_differences']:
            scan=next(v for v in d['scan'] if v['epsilon']==recorded['epsilon'])
            # Same cross-host replay tolerance as the original independent bank checker.
            assert abs(scan['finite_difference']-recorded['finite_difference'])<=1e-10*(1+abs(recorded['finite_difference']))
            assert scan['original_gate_at_this_epsilon']==recorded['qualified']
        rows[name]=dict(original_qualified=original['qualified'],original_checks=original['finite_differences'],**d)
    assert sha(file)==done['sha256']
    dump(a.output,dict(passed=True,scope='post-observation single-bank diagnosis; original gates unchanged',
        case=done['case'],split=done['split'],bank=done['bank'],steps=done['steps'],rows=rows,
        complete_sha256=sha(a.input/'COMPLETE.json'),array_sha256=sha(file),source_sha256=sha(__file__),
        diagnostic_source_sha256=sha(Path(__file__).with_name('diagnose_metric_rank.py')),
        new_model_calls=0,new_samples=0,original_failure_preserved=True))
    print('CACHED_ANALYTIC_TARGET_RANK_DIAGNOSIS',done['case'],done['split'],done['bank'],flush=True)


if __name__=='__main__':main()
