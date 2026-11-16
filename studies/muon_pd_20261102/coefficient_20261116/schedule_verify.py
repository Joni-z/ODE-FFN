"""Reconstruct captured native order maps in NumPy without importing Torch."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--witness', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, out = Path(a.witness), Path(a.output)
    assert not out.exists()
    receipt = json.loads((root/'PASS.json').read_text())
    assert receipt['passed'] and receipt['native_continuations'] == receipt['real_evaluator_witnesses'] == 4
    data = json.loads((root/'math.json').read_text())
    h, operator, offset = [np.asarray(data[name]) for name in ['entrance', 'P', 'B']]
    q = lambda v: v/np.sqrt(1+1.2*v*v)
    bracket = offset@((1+1.2*h*h)**(-1.5)*(operator@q(h)))-operator@((1+1.2*h*h)**(-1.5)*(offset@q(h)))
    maximum = 0.
    for row in data['nonlinear_native']:
        tau = row['tau']
        values = []
        for label, signs in [('palindrome', [1, -1, -1, 1]), ('alternating', [1, -1, 1, -1])]:
            y, dt = h.copy(), tau/4
            for sign in signs:
                matrix = operator+sign*offset
                middle = y+.5*dt*(matrix@q(y))
                y = y+dt*(matrix@q(middle))
            recorded = np.asarray(data['nonlinear_arrays'][str(tau)][label])
            maximum = max(maximum, float(np.max(np.abs(y-recorded))))
            values.append(y)
        remainder = float(np.linalg.norm(values[1]-values[0]+tau*tau*bracket/8))
        assert abs(remainder-row['remainder']) < 1e-12
    assert maximum < 1e-12
    result = dict(passed=True, native_maps=10, max_numpy_map_error=maximum,
        native_continuations=4, actual_evaluator_witnesses=4,
        math_sha256=hashlib.sha256((root/'math.json').read_bytes()).hexdigest(),
        readiness_sha256=hashlib.sha256((root/'PASS.json').read_bytes()).hexdigest())
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
