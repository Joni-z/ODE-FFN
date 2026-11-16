"""NumPy reconstruction of saved field predictions and array-derived gates."""
import argparse
import numpy as np
from pathlib import Path
from population_resolution import read,dump,sha


def comparison(x,y,atol,rtol):
    error=np.abs(x-y);ok=error<=atol+rtol*np.abs(y)
    return dict(qualified=bool(ok.all()),max_error=float(error.max()),failed_components=int((~ok).sum()))


def main():
    p=argparse.ArgumentParser()
    for name in ('root','input','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cpu-witness',type=Path);a=p.parse_args();assert not a.output.exists()
    done=read(a.input/'COMPLETE.json');assert done['passed'] and done['tensors_methods_rng_unchanged']
    fields=read(a.root/'fields.json');portable=read(a.root/'portable.json')
    assert done['fields_sha256']==sha(a.root/'fields.json') and done['portable_sha256']==sha(a.root/'portable.json')
    rows=read(a.input/'rows.json');assert len(rows)==done['rows'];errors=[];gates=0
    for row in rows:
        case=next(c for c in portable['cases'] if c['id']==row['case'])
        assert sha(a.root/case['tensor_file'])==case['tensor_sha256']
        binding=next(f for f in fields['rows'] if (f['dataset'],f['split'])==(case['dataset'],row['split']))
        assert sha(a.root/binding['file'])==binding['sha256']
        file=a.input/row['file'];assert sha(file)==row['sha256']
        with np.load(file) as z:values={k:z[k].copy() for k in z.files}
        count=row['points'];assert count==(16 if done['witness'] else 4096)
        with np.load(a.root/binding['file']) as z:
            for name,key in [('x','field_x'),('time','field_t'),('oracle','field_oracle')]:
                assert np.array_equal(values[name],z[key][:count].astype(np.float64))
        expected={};refined={}
        for kind,atol,rtol in [('primal',1e-7,1e-7),('tangent',1e-8,1e-5)]:
            expected[kind+'_refinement']=comparison(values[kind+'32'],values[kind+'64'],atol,rtol)
            before=float(np.max(np.abs(values[kind+'32']-values[kind+'16'])))
            after=float(np.max(np.abs(values[kind+'64']-values[kind+'32'])))
            assert expected[kind+'_refinement']==row['checks'][kind+'_refinement']
            assert row['checks'][kind+'_decreasing']==dict(qualified=after<before or max(before,after)<=1e-12,
                difference16to32=before,difference32to64=after)
            if done['refined_reference']:
                refined[kind+'_refinement']=comparison(values[kind+'128'],values[kind+'256'],atol,rtol)
                before=float(np.max(np.abs(values[kind+'128']-values[kind+'64'])))
                after=float(np.max(np.abs(values[kind+'256']-values[kind+'128'])))
                assert row['refined_checks'][kind+'_decreasing']==dict(qualified=after<before or max(before,after)<=1e-12,
                    difference64to128=before,difference128to256=after)
        with np.load(file.parent/'continuous_loose.npz') as z:loose=z['predictions']
        with np.load(file.parent/'continuous_tight.npz') as z:tight=z['predictions']
        expected['adaptive_reference']=comparison(loose,tight,1e-9,1e-9)
        expected['primal64_reference']=comparison(values['primal64'][:16],tight,1e-7,1e-7)
        for epsilon in (1e-4,1e-5):
            key='finite_difference'+str(epsilon)
            expected[key]=comparison(values['tangent16'][:8],values[key],2e-7,2e-4)
        if done['refined_reference']:
            refined['primal256_reference']=comparison(values['primal256'][:16],tight,1e-7,1e-7)
        if a.cpu_witness:
            with np.load(a.cpu_witness/file.parent.name/'arrays.npz') as z:
                for kind in ('primal','tangent'):
                    expected['cpu_'+kind]=comparison(values[kind+'16'][:16],z[kind+'16'],1e-9,1e-8)
                    if done['refined_reference']:refined['cpu_'+kind+'256']=comparison(values[kind+'256'][:16],z[kind+'256'],1e-9,1e-8)
        assert all(v==row['checks'][k] for k,v in expected.items())
        assert all(v==row['refined_checks'][k] for k,v in refined.items());gates+=len(expected)+len(refined)
        original=all(v['qualified'] for v in row['checks'].values());assert original==row['original_numerically_qualified']
        keep={k:v for k,v in row['checks'].items() if k not in
            ('primal_refinement','tangent_refinement','primal_decreasing','tangent_decreasing','primal64_reference')}
        qualified=all(v['qualified'] for v in keep.values()) and all(v['qualified'] for v in row['refined_checks'].values()) if done['refined_reference'] else original
        assert qualified==row['numerically_qualified']
        delta=values['crossed']-values['native'];pred=values['tangent'+str(row['final_reference_steps'])]
        signal=float(np.mean(delta**2));error=float(np.mean((pred-delta)**2));base=values['native']-values['oracle']
        actual=float(2*np.mean(base*delta)+np.mean(delta**2));forecast=float(2*np.mean(base*pred)+np.mean(pred**2))
        errors.extend([abs(signal-row['signal_mse']),abs(error-row['prediction_error_mse']),
            abs(actual-row['actual_oracle_risk_change']),abs(forecast-row['predicted_oracle_risk_change'])])
        if signal:errors.append(abs((1-error/signal)-row['score']))
        else:assert row['score'] is None
    assert max(errors)<1e-12
    dump(a.output,dict(passed=True,witness=done['witness'],rows=len(rows),saved_array_gates_reconstructed=gates,
        max_scalar_error=max(errors),all_numerically_qualified=done['all_numerically_qualified'],
        refined_reference=done['refined_reference'],fields_sha256=done['fields_sha256'],
        complete_sha256=sha(a.input/'COMPLETE.json'),rows_sha256=sha(a.input/'rows.json'),
        source_sha256=sha(__file__),native_executed_checks='whole-network JVP, zero/sign, unchanged tensors/methods/RNG'))
    print('INDEPENDENT_DEFECT_FIELD_PASS',len(rows),max(errors),flush=True)


if __name__=='__main__':main()
