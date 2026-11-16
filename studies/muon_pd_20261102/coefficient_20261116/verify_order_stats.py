"""Independent fsum/integer-CDF check of all conditional bank contrasts."""
import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--analysis', required=True)
    p.add_argument('--banks', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    analysis, banks_file, out = map(Path, [a.analysis, a.banks, a.output])
    assert not out.exists()
    result = json.loads(analysis.read_text())
    assert result['passed'] and len(result['contrasts']) == 10
    with banks_file.open() as file:
        banks = list(csv.DictReader(file))
    patterns = []
    for sample in itertools.combinations_with_replacement(range(8), 8):
        multiplicity = math.factorial(8)//math.prod(math.factorial(v) for v in Counter(sample).values())
        patterns.append((sample, multiplicity))
    assert len(patterns) == 6435 and sum(w for _, w in patterns) == 8**8
    maximum, pvalues = 0., []
    for row in result['contrasts']:
        mode = 'full-outer' if row['inner'] == 4 else 'full-internal'
        values = {}
        for family in ['palindrome_operator_midpoint4', 'alternating_operator_midpoint4']:
            id = row['dataset']+'__'+family+'__adamw_0.0003'
            selected = sorted([r for r in banks if r['id'] == id and r['mode'] == mode and
                int(r['inner']) == row['inner'] and int(r['outer']) == row['outer']], key=lambda r: int(r['bank']))
            assert [int(r['bank']) for r in selected] == list(range(8))
            values[family] = [float(r['score']) for r in selected]
        left, right = values.values()
        delta = [x-y for x, y in zip(left, right)]
        assert delta == row['bank_differences']
        errors = [abs(math.fsum(delta)/8-row['paired_mean']), abs(statistics.stdev(delta)-row['paired_sd'])]
        distribution = sorted((math.fsum(delta[i] for i in sample)/8, weight) for sample, weight in patterns)
        for q, expected in zip([.025, .975, .0025, .9975], row['marginal95']+row['nominal_ten_contrast_bonferroni']):
            threshold, count = math.ceil(q*8**8), 0
            for value, weight in distribution:
                count += weight
                if count >= threshold:
                    errors.append(abs(value-expected))
                    break
        n, k = sum(v != 0 for v in delta), sum(v < 0 for v in delta)
        probability = min(1., 2*sum(math.comb(n, j) for j in range(min(k, n-k)+1))/2**n) if n else 1.
        errors.append(abs(probability-row['paired_sign_p']))
        maximum = max(maximum, *errors)
        pvalues.append(probability)
    adjusted = [0.]*10
    previous = 0.
    for rank, index in enumerate(sorted(range(10), key=lambda i: pvalues[i])):
        previous = min(1., max(previous, (10-rank)*pvalues[index]))
        adjusted[index] = previous
    maximum = max(maximum, *(abs(v-r['paired_sign_holm_ten']) for v, r in zip(adjusted, result['contrasts'])))
    assert maximum < 1e-12
    receipt = dict(passed=True, contrasts=10, exact_bootstrap_groups=6435,
        maximum_independent_error=maximum, analysis_sha256=sha(analysis), banks_sha256=sha(banks_file),
        source_sha256=sha(Path(__file__)), scope='conditional sampling-bank statistics at fixed weights')
    out.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
