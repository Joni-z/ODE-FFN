"""Independent source-row joins and recursive weighted paired statistics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def multisets(total, bins):
    if bins == 1:
        yield (total,)
    else:
        for head in range(total+1):
            for rest in multisets(total-head, bins-1):
                yield (head,)+rest


def main():
    parser = argparse.ArgumentParser()
    for name in ('input', 'binding', 'ordinary-input', 'ode-input', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    assert not a.output.exists()
    result = read(a.input/'summary.json')
    assert len(result['rows']) == 48 and len(result['contrasts']) == 32
    assert result['primary_contrasts'] == 32 and result['primary_steps'] == 2048
    assert result['adjusted_tail'] == .05/64
    for file, digest in result['source_bindings'].items():
        assert sha(file) == digest, file
    binding = read(a.binding)
    originals, errors = {}, []

    def close(actual, expected):
        error = abs(actual-expected)
        assert error < 1e-12*(1+abs(expected)), (actual, expected, error)
        errors.append(error)

    for row in result['rows']:
        if row['ordinary']:
            entry = next(c for c in binding['cases'] if c['case']['id'] == row['model'])
            data = read(a.ordinary_input/f"index{entry['index']}"/'rows.json')
            original = next(r for r in data if (r['split'], r['steps']) == (row['split'], row['steps']))
            pooled, banks = original['pooled_sw2'], original['bank_sw2']
            assert row['checkpoint_sha256'] == entry['case']['checkpoint_sha256']
        else:
            case, kind = row['model'].split('::')
            data = read(a.ode_input/case/'rows.json')
            original = next(r for r in data if (r['split'], r['steps']) == (row['split'], row['steps']))
            pooled, banks = original[kind]['pooled_sw2'], original[kind]['bank_sw2']
        assert banks == row['bank_sw2'] and pooled == row['pooled_sw2'] and len(banks) == 8
        key = row['model'], row['split'], row['steps']
        assert key not in originals
        originals[key] = row
    counts = list(multisets(8, 8))
    assert len(counts) == 6435
    weights = [math.factorial(8)//math.prod(math.factorial(k) for k in c) for c in counts]
    assert sum(weights) == 8**8
    pvalues, keys = [], set()
    for row in result['contrasts']:
        ode = originals[(row['ode_model'], row['split'], 2048)]
        control = originals[(row['control_model'], row['split'], 2048)]
        assert not ode['ordinary'] and control['ordinary']
        assert ode['dataset'] == control['dataset'] == row['dataset']
        key = row['ode_model'], row['control_model'], row['split']
        assert key not in keys; keys.add(key)
        differences = [x*x-y*y for x, y in zip(ode['bank_sw2'], control['bank_sw2'], strict=True)]
        for x, y in zip(differences, row['bank_squared_loss_difference'], strict=True):
            close(x, y)
        close(row['mean'], statistics.mean(differences))
        close(row['median'], statistics.median(differences))
        close(row['sd'], statistics.stdev(differences))
        close(row['pooled_squared_loss_difference'], ode['pooled_sw2']**2-control['pooled_sw2']**2)
        distribution = sorted((sum(c*x for c, x in zip(count, differences))/8, weight)
                              for count, weight in zip(counts, weights))
        for key, tail in [('ci95', .025), ('ci_adjusted', .05/64)]:
            wanted = []
            for q in (tail, 1-tail):
                threshold, cumulative = math.ceil(q*8**8), 0
                for value, weight in distribution:
                    cumulative += weight
                    if cumulative >= threshold:
                        wanted.append(value); break
            for actual, expected in zip(row[key], wanted, strict=True):
                close(actual, expected)
        positive = sum(x > 0 for x in differences)
        negative = sum(x < 0 for x in differences)
        n = positive+negative
        probability = min(1., 2*sum(math.comb(n, k) for k in range(min(positive, negative)+1))/2**n) if n else 1.
        close(row['exact_sign_p'], probability)
        pvalues.append(probability)
    order = sorted(range(32), key=pvalues.__getitem__)
    for rank, original in enumerate(order):
        corrected = min(1., max((32-j)*pvalues[order[j]] for j in range(rank+1)))
        close(result['contrasts'][original]['holm_sign_p'], corrected)
    proof = dict(passed=True, source_rows=48, primary_contrasts=32,
        scalar_checks=len(errors), max_error=max(errors),
        bootstrap='independent recursive 6435-multiset enumeration',
        summary_sha256=sha(a.input/'summary.json'), source_sha256=sha(__file__))
    a.output.write_text(json.dumps(proof, indent=2)+'\n')
    print('INDEPENDENT_ORDINARY_COMPARISON_PASS', len(errors), max(errors), flush=True)


if __name__ == '__main__':
    main()
