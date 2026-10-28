"""Exact scalar interpretation fixture, not a trained experiment or image score."""
from pathlib import Path
import argparse,json,math
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
preflight=json.loads((args.root/'preflight.json').read_text());assert preflight['status']=='PASS'
epsilon=.05;a=epsilon/math.e;rows=[]
for record in preflight['fixtures']:
    n=record['N'];factor=1.
    for i in range(n):
        l=max(1-i/n,epsilon);r=max(1-(i+1)/n,epsilon)
        factor*=1-(1/l+(1-1/(n*l))/r)/(2*n)
    assert math.isclose(factor,record['zero_clean_factor'],rel_tol=1e-13,abs_tol=1e-15)
    rows.append(dict(N=n,discrete_factor=factor,continuous_factor=a,absolute_gap=abs(factor-a)))
assert 0<a**4<a*a
result=dict(status='PASS',scope='Analytic zero-data scalar identity; no trained output or image FID',epsilon=epsilon,
    perfect_predictor_terminal_std=a,perfect_predictor_terminal_variance=a*a,
    negative_identity_predictor_terminal_variance=a**4,original_registered_fixture_replay=rows,
    cap_or_sampler_changed=False)
assert not args.output.exists();args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
