"""Deterministic mathematical controls before any learned hidden-energy audit."""
import argparse
from pathlib import Path
import numpy as np
from geometry import dump, sha


def potential(value: np.ndarray) -> np.ndarray:
    # Rationalization avoids cancellation near zero.
    return np.sum(value**2/(np.sqrt(1+1.2*value**2)+1), axis=-1)


def response(value: np.ndarray) -> np.ndarray:
    return value/np.sqrt(1+1.2*value**2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    points = np.vstack([np.zeros(4), np.linspace(-.2, .2, 4), np.arange(1., 5.),
                        np.array([-10., -1., 2., 8.])])
    matrix = np.sin(np.arange(16).reshape(4, 4)+1)/3
    symmetric, skew = (matrix+matrix.T)/2, (matrix-matrix.T)/2
    q = response(points)
    epsilon = 1e-6
    gradient = np.zeros_like(points)
    for i in range(4):
        delta = np.zeros_like(points)
        delta[:, i] = epsilon
        gradient[:, i] = (potential(points+delta)-potential(points-delta))/(2*epsilon)
    gradient_error = float(np.max(np.abs(gradient-q)))
    assert gradient_error < 2e-8
    work = np.einsum('bi,ij,bj->b', q, matrix, q)
    symmetric_work = np.einsum('bi,ij,bj->b', q, symmetric, q)
    balance_error = float(np.max(np.abs(work-symmetric_work)))
    skew_work = float(np.max(np.abs(np.einsum('bi,ij,bj->b', q, skew, q))))
    assert max(balance_error, skew_work) < 1e-14
    change = .05*np.cos(np.arange(points.size).reshape(points.shape)+1)
    remainder = potential(points+change)-potential(points)-(q*change).sum(1)
    bound = .5*np.square(change).sum(1)
    assert np.all(np.abs(remainder) <= bound+1e-13)
    linear_skew = np.array([[0., -.7, 0., 0.], [.7, 0., 0., 0.],
                            [0., 0., 0., -1.3], [0., 0., 1.3, 0.]])
    rows = []
    for step in [.001, .01, .1, .25]:
        midpoint = points+.5*step*(points@linear_skew.T)
        endpoint = points+step*(midpoint@linear_skew.T)
        observed = .5*(np.square(endpoint).sum(1)-np.square(points).sum(1))
        expected = step**4/8*np.square(points@(linear_skew@linear_skew).T).sum(1)
        maximum = float(np.max(np.abs(observed-expected)))
        assert maximum < 1e-12
        rows.append(dict(step=step, quadratic_midpoint_defect_max_error=maximum,
                         expected_defect=expected.tolist(), observed_defect=observed.tolist()))
    result = dict(passed=True, learned_models=0, gradient_finite_difference_max_error=gradient_error,
        symmetric_work_identity_max_error=balance_error, skew_work_max_error=skew_work,
        smooth_potential_remainder=remainder.tolist(), smooth_potential_bound=bound.tolist(),
        quadratic_controls=rows, source_sha256=sha(Path(__file__)),
        scope='mathematical identities only; no learned-state mechanism claim')
    dump(out/'PASS.json', result)
    print('PASS hidden-energy mathematical controls', gradient_error, balance_error, skew_work)


if __name__ == '__main__':
    main()
