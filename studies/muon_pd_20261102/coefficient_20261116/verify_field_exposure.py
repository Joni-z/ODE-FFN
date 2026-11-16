"""Independent complete-cohort local solver and field-risk reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import solve_ivp


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integrate(initial: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    shape = initial.shape
    def f(time: float, value: np.ndarray) -> np.ndarray:
        y = value.reshape(shape)
        return ((y/np.sqrt(1+1.2*y*y))@matrix.T).ravel()
    solution = solve_ivp(f, (0.,1.), initial.ravel(), method='RK45', rtol=1e-11, atol=1e-13)
    assert solution.success and np.isfinite(solution.y).all(), solution.message
    return solution.y[:,-1].reshape(shape)


def discrete(initial: np.ndarray, matrix: np.ndarray, method: str) -> np.ndarray:
    value = initial.copy()
    def f(y: np.ndarray) -> np.ndarray:
        return (y/np.sqrt(1+1.2*y*y))@matrix.T
    steps = 4 if method=='midpoint' else 2
    dt = 1/steps
    for _ in range(steps):
        a = f(value)
        b = f(value+dt*a/2)
        if method=='midpoint':
            value += dt*b
        else:
            c = f(value+dt*b/2)
            d = f(value+dt*c)
            value += (dt/6)*(a+2*b+2*c+d)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    complete = read(root/'COMPLETE.json')
    assert complete['passed'] and complete['rows_sha256']==sha(root/'rows.json')
    assert complete['inputs_sha256']==sha(root/'inputs.npz')
    with np.load(root/'inputs.npz') as z:
        oracle, inputs, clock = z['oracle'].copy(), z['x'].copy(), z['time'].copy()
        assert np.array_equal(z['original_indices'], np.arange(len(inputs)))
    assert len(inputs)==(8 if complete['witness'] else 4096)
    rows, before = [], time.monotonic()
    for row in read(root/'rows.json')['rows']:
        case = root/row['id']
        qualified = np.ones(len(inputs), dtype=bool)
        mode_records, predictions = [], {}
        for mode in row['modes']:
            file = case/mode['file']
            assert mode['sha256']==sha(file)
            with np.load(file) as z:
                h, endpoints, matrices, field = [z[k].copy() for k in
                    ['entrances','endpoints','matrices','predictions']]
            assert h.shape==endpoints.shape==(6,len(inputs),64)
            assert matrices.shape==(6,64,64) and field.shape==oracle.shape
            predictions[mode['mode']] = field
            layer_records = []
            for layer in range(6):
                independent = np.empty_like(endpoints[layer])
                continuous = mode['mode'].startswith('continuous')
                if continuous:
                    for start in range(0,len(inputs),64):
                        stop = min(len(inputs),start+64)
                        independent[start:stop] = integrate(h[layer,start:stop],matrices[layer])
                    local = np.all(np.abs(independent-endpoints[layer]) <=
                                   2e-8+2e-8*np.abs(endpoints[layer]),axis=1)
                    qualified &= local
                else:
                    method = mode['mode'] if mode['mode'] in ('midpoint','rk4') else (
                        'midpoint' if row['family']=='state_midpoint4' else 'rk4')
                    independent = discrete(h[layer],matrices[layer],method)
                    assert np.allclose(independent,endpoints[layer],atol=1e-11,rtol=1e-11)
                    local = np.ones(len(inputs),dtype=bool)
                layer_records.append(dict(layer=layer,points=len(inputs),qualified_points=int(local.sum()),
                    reconstruction_max_error=float(np.max(np.abs(independent-endpoints[layer])))))
                (out/'progress.json').write_text(json.dumps(dict(id=row['id'],mode=mode['mode'],
                    layers_completed=layer+1,layers=6,seconds=time.monotonic()-before))+'\n')
            mode_records.append(dict(mode=mode['mode'],layers=layer_records))
            print(row['id'],mode['mode'],'verified',flush=True)
        assert np.array_equal(predictions['original'],predictions['identity'])
        own = 'midpoint' if row['family']=='state_midpoint4' else 'rk4'
        assert np.array_equal(predictions['original'],predictions[own])
        continuous = predictions['continuous_tight']
        error = continuous-oracle
        maximum = 0.
        risks = {}
        for method in ('midpoint','rk4'):
            e = predictions[method]-oracle
            delta = continuous-predictions[method]
            risk = float(np.einsum('ij,ij->',e,e)/len(inputs))
            shift = float(np.einsum('ij,ij->',delta,delta)/len(inputs))
            cross = float(np.einsum('ij,ij->',e,delta)/len(inputs))
            continuous_risk = float(np.einsum('ij,ij->',error,error)/len(inputs))
            expected = row['risk_accounting'][method]
            comparisons = dict(R=risk,S=shift,C=cross,continuous_R=continuous_risk,
                               risk_change=continuous_risk-risk)
            for key,value in comparisons.items():
                maximum = max(maximum,abs(value-expected[key]))
            maximum = max(maximum,abs(continuous_risk-risk-2*cross-shift))
            assert maximum < 1e-10
            risks[method] = comparisons
        rows.append(dict(id=row['id'],dataset=row['dataset'],family=row['family'],
            points=len(inputs),qualified_points=int(qualified.sum()),modes=mode_records,
            risk_accounting=risks,risk_reconstruction_max_error=maximum))
    result = dict(passed=True,witness=complete['witness'],cases=len(rows),rows=rows,
        all_points_qualified=all(r['points']==r['qualified_points'] for r in rows),
        input_complete_sha256=sha(root/'COMPLETE.json'),source_sha256=sha(Path(__file__)),
        seconds=time.monotonic()-before)
    (out/'PASS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('independent full-field verification complete',result['all_points_qualified'],flush=True)


if __name__ == '__main__':
    main()
