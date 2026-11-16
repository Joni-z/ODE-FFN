"""Independent recursive multinomial enumeration and full-cohort decision audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recursive_counts(remaining=8, slots=8, prefix=()):
    if slots==1:
        yield prefix+(remaining,)
    else:
        for number in range(remaining+1):
            yield from recursive_counts(remaining-number,slots-1,prefix+(number,))


def bootstrap(values,tail):
    counts=np.array(list(recursive_counts()),dtype=np.int64)
    frequencies=np.array([math.factorial(8)//math.prod(math.factorial(int(k)) for k in row)
                          for row in counts],dtype=np.int64)
    assert counts.shape==(6435,8) and np.all(counts.sum(1)==8) and frequencies.sum()==8**8
    means=np.sum(counts*np.asarray(values)[None],axis=1)/8
    order=np.argsort(means);probability=np.cumsum(frequencies[order])/(8**8)
    return [float(means[order[np.searchsorted(probability,q,side='left')]]) for q in (tail,1-tail)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input');parser.add_argument('--output',required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();output=Path(args.output);assert not output.exists()
    output.parent.mkdir(parents=True,exist_ok=True)
    assert bootstrap(np.ones(8),.025)==[1.,1.]
    symmetric=bootstrap(np.arange(8),.025)
    assert symmetric[0]+symmetric[1]==7
    if args.check_only:
        output.write_text(json.dumps(dict(passed=True,recursive_count_vectors=6435,
            ordered_resamples=8**8,symmetric_interval=symmetric,check_only=True),indent=2)+'\n')
        print('TARGET_STATISTICS_CONTROLS_PASS');return
    assert args.input
    root=Path(args.input);summary=read(root/'summary.json')
    assert len(summary['rows'])==32 and len(summary['contrasts'])==8
    errors=[]
    for contrast in summary['contrasts']:
        actual,predicted=np.asarray(contrast['actual']),np.asarray(contrast['predicted'])
        assert actual.shape==predicted.shape==(8,)
        expected=dict(actual_mean=float(np.mean(actual)),actual_sd=float(np.std(actual,ddof=1)),
            predicted_mean=float(np.mean(predicted)),residual_mean=float(np.mean(predicted-actual)))
        for key,value in expected.items():errors.append(abs(value-contrast[key]))
        for key,values,tail in [('actual_ci95',actual,.025),('actual_ci_adjusted',actual,.05/16),
            ('residual_ci95',predicted-actual,.025),('residual_ci_adjusted',predicted-actual,.05/16)]:
            expected=bootstrap(values,tail)
            errors.append(float(np.max(np.abs(np.array(expected)-contrast[key]))))
    for filename,digest in summary['source_bindings'].items():
        file=Path(filename);assert sha(file)==digest
        data=read(file);verified=read(file.parent/'independent.json')
        assert verified['passed'] and verified['summary_sha256']==digest and not verified['witness']
        assert len(data['banks'])==8 and all(row['points']==4096 for row in data['banks'])
    primary=[row for row in summary['rows'] if row['case']=='product_rk4_trained' and row['coefficient']==1]
    assert len(primary)==2
    expected=all(row['qualified'] and row['precision_direction'] and
                 row['predicted_loss_change']*row['actual_loss_change']>0 and
                 (row['predicted_loss_change']-row['actual_loss_change'])**2<row['actual_loss_change']**2
                 for row in primary)
    assert expected==summary['primary_full_scale_explained']
    assert max(errors)<1e-12
    output.write_text(json.dumps(dict(passed=True,check_only=False,contrasts=8,
        independent_max_error=max(errors),primary_full_scale_explained=expected,
        summary_sha256=sha(root/'summary.json'),source_sha256=sha(__file__)),indent=2)+'\n')
    print('TARGET_TRANSPORT_STATISTICS_PASS',max(errors))


if __name__=='__main__':main()
