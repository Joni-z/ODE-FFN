"""Independent count-vector verification of the all-bank bootstrap statistics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def counts(total: int, slots: int):
    if slots == 1:
        yield (total,)
    else:
        for first in range(total+1):
            for tail in counts(total-first, slots-1):
                yield (first,)+tail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    assert not output.exists()
    data = json.loads(source.read_text())
    assert data['passed'] and len(data['pairs']) == 2
    checks = []
    for pair in data['pairs']:
        delta = np.asarray(pair['bank_differences'])
        assert delta.shape == (8,) and np.isfinite(delta).all()
        assert abs(float(delta.mean())-pair['primary_gap']) < 1e-12
        assert abs(float(delta.std(ddof=1))-pair['bootstrap_unit_sd']) < 1e-12
        values, multiplicity = [], []
        for c in counts(8, 8):
            values.append(sum(float(v)*int(n) for v, n in zip(delta, c))/8)
            multiplicity.append(math.factorial(8)//math.prod(math.factorial(n) for n in c))
        assert len(values) == 6435 and sum(multiplicity) == 8**8
        order = np.argsort(values)
        cumulative = np.cumsum(np.asarray(multiplicity, dtype=np.int64)[order])
        values = np.asarray(values)[order]
        at = lambda q: float(values[np.searchsorted(cumulative, math.ceil(q*8**8))])
        actual = [at(.025), at(.975), at(.0125), at(.9875)]
        expected = pair['exact_bootstrap']['bootstrap95']+pair['exact_bootstrap']['two_contrast_bonferroni']
        error = float(np.max(np.abs(np.asarray(actual)-expected)))
        assert error < 1e-14
        assert pair['expected_direction_banks'] == int(np.count_nonzero(delta*pair['expected_gap_sign'] > 0))
        checks.append(dict(dataset=pair['dataset'], max_interval_error=error, count_vectors=len(values),
                           total_multiplicity=int(cumulative[-1])))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    output.write_text(json.dumps(dict(passed=True, checks=checks, input_sha256=digest(source),
        source_sha256=digest(Path(__file__))), indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True, checks=checks)), flush=True)


if __name__ == '__main__':
    main()
