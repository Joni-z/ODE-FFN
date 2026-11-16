"""Independently check every saved hidden endpoint and continuous reference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    complete = json.loads((root/'COMPLETE.json').read_text())
    assert complete['passed'] and complete['rows_sha256'] == digest(root/'rows.json')
    row = json.loads((root/'rows.json').read_text())
    assert row['inputs_sha256'] == digest(root/'inputs.npz')
    with np.load(root/'inputs.npz') as bundle:
        count = len(bundle['bank'])
        assert set(bundle['bank']) == set(range(8))
        assert count == (8 if row['witness'] else 80)
    records = []
    for layer in row['layers']:
        file = root/layer['file']
        assert layer['sha256'] == digest(file)
        with np.load(file) as bundle:
            z = {key:bundle[key].copy() for key in bundle.files}
        h, matrix = z['entrance'], z['matrix']
        assert len(h) == count
        def f(time: float, state: np.ndarray) -> np.ndarray:
            return matrix@(state/np.sqrt(1+1.2*state*state))
        endpoints = []
        for steps in z['levels']:
            y, dt = h.copy(), 1/int(steps)
            for _ in range(int(steps)):
                a = np.array([f(0, v) for v in y])
                b = np.array([f(0, v) for v in y+.5*dt*a])
                if row['family'] == 'state_midpoint4':
                    y += dt*b
                else:
                    assert row['family'] == 'state_rk4_2'
                    c = np.array([f(0, v) for v in y+.5*dt*b])
                    d = np.array([f(0, v) for v in y+dt*c])
                    y += dt*(a+2*b+2*c+d)/6
            endpoints.append(y)
        endpoints = np.stack(endpoints)
        assert np.allclose(endpoints, z['endpoints'], atol=1e-11, rtol=1e-11)
        independent = []
        for initial in h:
            result = solve_ivp(f, (0., 1.), initial, method='RK45', rtol=1e-11, atol=1e-13)
            assert result.success and np.isfinite(result.y).all()
            independent.append(result.y[:, -1])
        independent = np.stack(independent)
        reference = z['references'][-1]
        qualified = np.all(np.abs(independent-reference) <= 2e-8+2e-8*np.abs(reference), axis=1)
        actual_error2 = np.sum((z['endpoints']-reference)**2, axis=2)
        actual_displacement2 = np.sum((reference-h)**2, axis=1)
        assert np.allclose(actual_error2, z['error_squared'], atol=1e-15, rtol=1e-12)
        assert np.allclose(actual_displacement2, z['displacement_squared'], atol=1e-15, rtol=1e-12)
        records.append(dict(layer=layer['layer'], points=count,
            qualified_points=int(qualified.sum()),
            native_reconstruction_max_error=float(np.max(np.abs(endpoints-z['endpoints']))),
            independent_reference_max_error=float(np.max(np.abs(independent-reference)))))
    result = dict(passed=True, id=row['id'], witness=row['witness'], layers=records,
        all_points_qualified=all(r['points'] == r['qualified_points'] for r in records),
        input_complete_sha256=digest(root/'COMPLETE.json'), source_sha256=digest(Path(__file__)))
    (out/'PASS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,id=row['id'],all_points_qualified=result['all_points_qualified'])),flush=True)


if __name__ == '__main__':
    main()
