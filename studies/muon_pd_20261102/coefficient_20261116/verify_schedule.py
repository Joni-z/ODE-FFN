"""NumPy reconstruction of the fixed-mean four-stage map diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q(x: np.ndarray) -> np.ndarray:
    return x/np.sqrt(1+1.2*x*x)


def solve(x: np.ndarray, weights: np.ndarray, delta: float) -> np.ndarray:
    y=x.copy()
    for w in weights:
        drift=q(y)@w.T
        y=y+delta*(q(y+.5*delta*drift)@w.T)
    return y


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x*x)))


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--probe',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    probe,out=Path(a.probe),Path(a.output)
    assert not out.exists()
    observed=read(probe)
    assert observed['passed'] and len(observed['rows'])==6
    assert observed['self_test']['four_stage_expansion']['passed']
    cases=[];vector_max=0.;scalar_max=0.
    def close_scalar(actual: float, expected: float) -> None:
        nonlocal scalar_max
        error=abs(actual-expected);scalar_max=max(scalar_max,error)
        assert error<1e-10*max(1.,abs(expected)),(actual,expected,error)
    for case in observed['rows']:
        folder=Path(case['source_folder']);identity=case['checkpoint']
        assert sha(folder/'checkpoints/step_0020000.pt')==identity['checkpoint_sha256']
        assert sha(folder/'config.json')==identity['config_sha256']
        assert sha(Path(case['dataset_manifest']))==identity['manifest_sha256']
        file=Path(case['arrays']['file']);assert sha(file)==case['arrays']['sha256']
        entries=[]
        with np.load(file) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files)
            oracle,prediction=z['field_oracle'],z['field_original']
            close_scalar(float(np.mean((prediction.astype(float)-oracle.astype(float))**2)),case['field_mse'])
            for changed in case['interventions']:
                value=z['field_'+changed['mode']]
                close_scalar(float(np.mean((value.astype(float)-oracle.astype(float))**2)),changed['field_mse'])
                close_scalar(float(np.sqrt(np.mean((value.astype(float)-prediction.astype(float))**2))),changed['prediction_shift_rms'])
            assert len(case['layers'])==30 and len({(e['layer'],e['time']) for e in case['layers']})==30
            for e in case['layers']:
                prefix=e['capture_prefix'];x,w=z[prefix+'entrance'],z[prefix+'weights']
                assert x.shape==(16,64) and w.shape==(4,64,64) and x.dtype==w.dtype==np.float64
                derivative=(1+1.2*x*x)**(-1.5)
                f=[q(x)@v.T for v in w]
                leading=sum((derivative*f[i])@w[j].T-(derivative*f[j])@w[i].T for i in range(4) for j in range(i+1,4))/32
                err=float(np.max(np.abs(leading-z[prefix+'leading'])));vector_max=max(vector_max,err);assert err<1e-10
                close_scalar(rms(z[prefix+'leading']),e['mean_flow_expansion']['leading_coefficient_rms'])
                horizons=e['mean_flow_expansion']['rows'];assert [r['tau'] for r in horizons]==[1.,.5,.25,.125,.0625]
                for i,row in enumerate(horizons):
                    tau=row['tau'];native=solve(x,w,tau/4)
                    mean_diff=native-solve(x,np.repeat(w.mean(0)[None],4,axis=0),tau/4)
                    reverse=native-solve(x,w[::-1],tau/4)
                    actual_mean,actual_reverse=z[prefix+f'tau{i}_mean_difference'],z[prefix+f'tau{i}_reverse_difference']
                    err=max(float(np.max(np.abs(mean_diff-actual_mean))),float(np.max(np.abs(reverse-actual_reverse))));vector_max=max(vector_max,err);assert err<1e-10
                    # Scalar diagnostics use the fixed captured arrays after the
                    # independent state/matrix reconstruction passed above.
                    pred=tau*tau*z[prefix+'leading'];scale=rms(actual_mean)
                    for key,value in [('mean_difference_rms',scale),('reverse_difference_rms',rms(actual_reverse)),
                                      ('predicted_mean_rms',rms(pred)),('mean_remainder_rms',rms(actual_mean-pred)),
                                      ('reverse_remainder_rms',rms(actual_reverse-2*pred)),
                                      ('relative_mean_prediction_error',rms(actual_mean-pred)/max(scale,1e-14))]:close_scalar(value,row[key])
                    if row['prediction_cosine'] is not None:
                        cosine=float(np.sum(actual_mean*pred)/(np.linalg.norm(actual_mean)*np.linalg.norm(pred)))
                        close_scalar(cosine,row['prediction_cosine'])
                unit,last,previous=horizons[0],horizons[-1],horizons[-2]
                order=float(np.log2(previous['mean_remainder_rms']/last['mean_remainder_rms'])) if min(previous['mean_remainder_rms'],last['mean_remainder_rms'])>1e-13 else None
                entries.append(dict(layer=e['layer'],time=e['time'],unit_relative_prediction_error=unit['relative_mean_prediction_error'],
                    unit_cosine=unit['prediction_cosine'],unit_mean_gap_rms=unit['mean_difference_rms'],
                    unit_gap_over_increment=unit['mean_difference_rms']/e['increment_rms'],
                    small_horizon_relative_prediction_error=last['relative_mean_prediction_error'],small_remainder_order=order))
        cases.append(dict(id=case['id'],dataset=case['dataset'],family=case['family'],checkpoint=identity,
            field_mse=case['field_mse'],interventions=case['interventions'],entries=entries,arrays_sha256=case['arrays']['sha256']))
    assert len({c['id'] for c in cases})==6
    result=dict(passed=True,cases=cases,models=6,layer_clock_records=180,horizons_per_record=5,
        max_independent_vector_error=vector_max,max_independent_scalar_error=scalar_max,
        source_probe_sha256=sha(probe),verification_source_sha256=sha(Path(__file__)))
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['passed','models','layer_clock_records','max_independent_vector_error','max_independent_scalar_error']}))


if __name__=='__main__':
    main()
