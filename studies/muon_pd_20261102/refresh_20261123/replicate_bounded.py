"""Frozen replication of the first ODE endpoint to beat both archived baselines."""
import fcntl
import os
from pathlib import Path
import sys
from final_banks import HERE, entries, read, run


if __name__=='__main__':
    prep=Path(sys.argv[1])
    spec=read(HERE.parents[2]/'docs/report4/midpoint-refresh/bounded-replication-lock.json')
    if spec['stream']==2026112300:
        raise ValueError('Reserved finalist stream cannot be reused')
    models=entries(prep,prep/'bounded-odd-20261123',require_symmetry_win=False)
    if [{k:v for k,v in m.items() if k!='folder'} for m in models]!=spec['weights']:
        raise ValueError('Checkpoint cohort differs from the pre-sampling lock')
    out=prep/f"bounded-locked-{spec['stream']}"
    out.mkdir(exist_ok=True)
    os.nice(8)
    with (out/'score.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(prep,models,out,spec['stream'])
