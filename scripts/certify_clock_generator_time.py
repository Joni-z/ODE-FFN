"""A fixed-generator, time-coupled refinement of the scalar flow certificate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from certify_clock_generator import quotient


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def certify(source,output):
    previous=json.loads((source/'certificate.json').read_text())
    assert previous['state']=='PASS' and sha(source/'example.npz')==previous['example_sha256']
    center=.6;delta=previous['delta'];records=[];maximum=0.
    with np.load(source/'example.npz') as data:example={k:data[k] for k in data.files}
    selected=previous['example']['axis'];selected_time=previous['example']['time']
    for axis in range(16):
        rows=[r for r in previous['rows'] if r['axis']==axis]
        assert len(rows)==5
        minimum=min(r['coefficient_min'] for r in rows);peak=max(r['coefficient_max'] for r in rows)
        midpoint=delta*(minimum+peak)/2
        exact_midpoint=(np.log1p(delta*minimum)+np.log1p(delta*peak))/2
        lo,hi=center+delta*(minimum-peak)/2,center+delta*(peak-minimum)/2
        assert .2<lo<=hi<1
        expanding=sum(r['expanding_increment_energy'] for r in rows)
        contracting=sum(r['contracting_increment_energy'] for r in rows)
        for row in rows:
            # A common interval covers every panel, with tails handled by the
            # analytic coefficient-range bounds rather than this finite grid.
            x=np.linspace(-8.,8.,2001)
            q=quotient(x,row['alpha'],row['beta'],row['kappa'])
            target=x*(1+delta*q)
            duration=center+delta*q-midpoint
            exact_duration=center+np.log1p(delta*q)-exact_midpoint
            assert .2<duration.min()<=duration.max()<1
            assert .2<exact_duration.min()<=exact_duration.max()<1
            euler=x+midpoint*x+(duration-center)*x
            exact=np.exp(exact_midpoint+exact_duration-center)*x
            error=max(float(np.abs(target-euler).max()),float(np.abs(target-exact).max()))
            maximum=max(maximum,error);assert error<1e-13
            if axis==selected and row['time']==selected_time:
                example=dict(x=x,velocity=q*x,target=target,additive=euler,exact=exact,
                             duration=duration,exact_duration=exact_duration)
        records.append(dict(axis=axis,mu=rows[0]['mu'],fixed_euler_P=midpoint/center,
            fixed_exact_P=exact_midpoint/center,euler_gate_min=lo,euler_gate_max=hi,
            expanding_increment_energy=expanding,contracting_increment_energy=contracting,
            time_coupled_positive_clock_floor=min(expanding,contracting),
            relative_floor=min(expanding,contracting)/(expanding+contracting)))
    numerator=sum(r['time_coupled_positive_clock_floor'] for r in records)
    denominator=sum(r['expanding_increment_energy']+r['contracting_increment_energy'] for r in records)
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'example.npz',**example)
    result=dict(state='PASS',axes=records,delta=delta,center=center,
        panel_relaxed_relative_floor=sum(r['positive_clock_risk_floor'] for r in previous['rows'])/denominator,
        time_coupled_relative_floor=numerator/denominator,
        time_coupled_velocity_mse_floor=numerator/(80*delta**2),
        target_velocity_energy=denominator/(80*delta**2),
        maximum_fixed_generator_witness_difference=maximum,
        original_certificate_sha256=sha(source/'certificate.json'),source_sha256=sha(Path(__file__)),
        dependency_sha256=sha(Path(__file__).with_name('certify_clock_generator.py')),
        scope='One fixed scalar generator per latent coordinate across the five declared times, arbitrary positive clocks. Additive witnesses use fixed generators too. Bounds do not apply to the full hidden multilayer velocity network.')
    (output/'certificate.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=certify(args.source,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='axes'}))
