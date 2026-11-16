"""Separate fsum/integer-CDF reconstruction of the six bank comparisons."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    file, out = Path(args.analysis), Path(args.output)
    assert not out.exists()
    result = json.loads(file.read_text())
    assert result['passed'] and len(result['contrasts']) == 6
    maximum = 0.
    for row in result['contrasts']:
        distribution = []
        for sample in itertools.combinations_with_replacement(range(8), 8):
            weight = math.factorial(8)//math.prod(math.factorial(v) for v in Counter(sample).values())
            if row['metric'] == 'bank_sw2':
                value = math.fsum(row['bank_differences'][i] for i in sample)/8
            else:
                value = math.fsum(row['candidate_error_bank'][i] for i in sample)/math.fsum(row['candidate_displacement_bank'][i] for i in sample)
                value -= math.fsum(row['reference_error_bank'][i] for i in sample)/math.fsum(row['reference_displacement_bank'][i] for i in sample)
            distribution.append((value, weight))
        distribution.sort()
        assert len(distribution) == 6435 and sum(w for _, w in distribution) == 8**8
        for name, q in [('conditional95', .025), ('nominal_four_adjusted', .05/8), ('six_contrast_sensitivity', .05/12)]:
            for probability, expected in zip([q, 1-q], row['intervals'][name]):
                threshold, seen = math.ceil(probability*8**8), 0
                for value, weight in distribution:
                    seen += weight
                    if seen >= threshold:
                        maximum = max(maximum, abs(value-expected))
                        break
        if row['metric'] == 'bank_sw2':
            maximum = max(maximum, abs(math.fsum(row['bank_differences'])/8-row['point_difference']),
                          abs(statistics.stdev(row['bank_differences'])-row['sd_difference']))
        else:
            point = math.fsum(row['candidate_error_bank'])/math.fsum(row['candidate_displacement_bank'])
            point -= math.fsum(row['reference_error_bank'])/math.fsum(row['reference_displacement_bank'])
            maximum = max(maximum, abs(point-row['point_difference']))
    assert maximum < 1e-12
    for dimension in [2, 16]:
        rows = [r for r in result['cost']['rows'] if r['dimension'] == dimension]
        assert len(rows) == 2
        assert len({r['parameters'] for r in rows}) == len({r['dense_macs_per_example'] for r in rows}) == 1
        for row in rows:
            assert all(len(v) == 100 for v in row['timings'].values())
            for key, values in row['timings'].items():
                assert statistics.median(values) == row['median_seconds'][key]
    receipt = dict(passed=True, contrasts=6, exact_groups=6435,
        maximum_independent_error=maximum,
        analysis_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
