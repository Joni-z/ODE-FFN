"""Monitor the two prospective designs and use the locked banks for at most one."""
import os
from pathlib import Path
import sys
import time
import subprocess
from final_banks import DATASETS, HERE, base, entries, read


def collect(prep, out, feedback=False):
    designs=['bounded-odd-20261123','glu-midpoint-20261123']
    if feedback:
        designs=['gate-feedback-20261123']
    workdirs=designs+['odd-control-20261123','glu-euler-ring-20261123']
    while True:
        unfinished=[]
        for name in workdirs:
            work=prep/name
            if (work/'selection.json').is_file():
                continue
            pid=int((work/'pid').read_text())
            try:
                os.kill(pid,0)
            except ProcessLookupError:
                raise RuntimeError(f'{name}: owner {pid} exited without selection results')
            unfinished.append(name)
        base.dump(out/'progress.json',dict(waiting=unfinished,checked_unix=time.time()))
        if not unfinished:
            break
        time.sleep(20)
    ordinary=read(HERE.parents[2]/'docs/report4/midpoint-refresh/ordinary-selection.json')
    limits={ds:min(min(r['original_20k_sw2'],r['matched_sw2']) for r in ordinary if r['dataset']==ds)
            for ds in DATASETS}
    for row in read(prep/'odd-control-20261123/selection.json'):
        ds=row['dataset']; limits[ds]=min(limits[ds],row['pooled_sw2'])
    passed=[];scores={}
    for name in designs:
        work=prep/name
        table=read(work/'selection.json')
        scores[name]={r['dataset']:r['pooled_sw2'] for r in table}
        try:
            entries(prep,work)
        except ValueError as err:
            scores[name]['decision']=str(err)
            continue
        # Resolve multiple eligible designs using the larger of the two task ratios.
        passed.append((max(r['pooled_sw2']/limits[r['dataset']] for r in table),name))
    selected=min(passed)[1] if passed else None
    base.dump(out/'selection-decision.json',dict(scores=scores,selected=selected,limits=limits))
    if selected and feedback:
        prior=read(prep/'refresh-screen-monitor-20261123/selection-decision.json')
        if prior['selected'] is not None:
            base.dump(out/'progress.json',dict(phase='existing_finalist',selected=selected,prior=prior['selected']))
            return
    if selected:
        subprocess.run([sys.executable,str(HERE/'final_banks.py'),str(prep),str(prep/selected)],check=True)
    base.dump(out/'progress.json',dict(phase='complete',selected=selected,checked_unix=time.time()))


if __name__=='__main__':
    prep=Path(sys.argv[1])
    feedback=sys.argv[2:]==['--feedback']
    if sys.argv[2:] and not feedback:
        raise ValueError('Expected --feedback')
    out=prep/('feedback-screen-monitor-20261123' if feedback else 'refresh-screen-monitor-20261123')
    out.mkdir(exist_ok=True)
    import fcntl
    with (out/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        collect(prep,out,feedback)
