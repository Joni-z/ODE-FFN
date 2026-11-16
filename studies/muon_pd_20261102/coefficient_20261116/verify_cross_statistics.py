"""Separate integer-CDF/fsum reconstruction of all crossed bank forecasts."""
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
    parser.add_argument('--analysis',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    file,out = Path(args.analysis),Path(args.output)
    assert not out.exists()
    data = json.loads(file.read_text())
    assert data['passed'] and len(data['contrasts'])==8
    maximum = 0.
    for row in data['contrasts']:
        values = row['bank_differences']
        distribution = []
        for sample in itertools.combinations_with_replacement(range(8),8):
            weight = math.factorial(8)//math.prod(math.factorial(v) for v in Counter(sample).values())
            distribution.append((math.fsum(values[i] for i in sample)/8,weight))
        distribution.sort()
        assert len(distribution)==6435 and sum(w for _,w in distribution)==8**8
        for name,q in [('marginal95',.025),('four_comparison_adjusted',.05/8),('eight_comparison_sensitivity',.05/16)]:
            for probability,expected in zip([q,1-q],row[name]):
                threshold,count = math.ceil(probability*8**8),0
                for value,weight in distribution:
                    count += weight
                    if count>=threshold:
                        maximum = max(maximum,abs(value-expected))
                        break
        maximum = max(maximum,abs(math.fsum(values)/8-row['mean_difference']),
                      abs(statistics.stdev(values)-row['sd_difference']))
        measured = next(r for r in data['rows'] if r['id']==row['id'] and r['split']==row['split'])
        differences = [a-b for a,b in zip(measured['crossed']['bank_sw2'],measured['native']['bank_sw2'])]
        assert differences==values
        maximum = max(maximum,abs(measured['crossed']['pooled_sw2']-measured['native']['pooled_sw2']-row['pooled_difference']))
    assert maximum<1e-12
    result = dict(passed=True,contrasts=8,exact_multisets=6435,maximum_independent_error=maximum,
        analysis_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
