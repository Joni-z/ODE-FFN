"""Regenerate paired field bootstrap with different packing and percentile math."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    file, out = Path(args.analysis), Path(args.output)
    assert not out.exists()
    data = json.loads(file.read_text())
    assert data['passed'] and len(data['cases'])==len(data['contrasts'])==4
    maximum, draws = 0., 0
    for binding in data['bindings']:
        source = Path(binding['file'])
        assert hashlib.sha256(source.read_bytes()).hexdigest()==binding['sha256']
        with np.load(source) as z:
            values, saved = z['values'].copy(), z['bootstrap'].copy()
        assert values.shape==(4096,4) and saved.shape==(10000,4)
        rng = np.random.default_rng(0)
        reproduced = np.empty_like(saved)
        for start in range(0,10000,25):
            index = rng.integers(0,4096,size=(25,4096))
            for column in range(4):
                reproduced[start:start+25,column] = np.take(values[:,column],index).sum(axis=1)/4096
        maximum = max(maximum,float(np.max(np.abs(reproduced-saved))))
        rows = [r for r in data['contrasts'] if r['task']==binding['task']]
        assert [r['trained_solver'] for r in rows]==['midpoint','rk4']
        for i,row in enumerate(rows):
            point = math.fsum(values[:,i])/4096
            maximum = max(maximum,abs(point-row['own_minus_other_vector_risk']))
            ordered = sorted(reproduced[:,i])
            def percentile(q: float) -> float:
                position = 9999*q
                low,high = math.floor(position),math.ceil(position)
                return ordered[low]+(position-low)*(ordered[high]-ordered[low])
            for q,expected in zip([.025,.975,.00625,.99375],row['marginal95']+row['four_comparison_adjusted']):
                maximum = max(maximum,abs(percentile(q)-expected))
        draws += 10000
    assert maximum<1e-12
    receipt = dict(passed=True,contrasts=4,bootstrap_draws=draws,
        maximum_independent_error=maximum,
        analysis_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
