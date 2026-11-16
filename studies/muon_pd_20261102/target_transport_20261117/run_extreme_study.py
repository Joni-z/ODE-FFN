"""Fixed eight-case CPU queue, following the already used monitored runner."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_study import dump

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    parser=argparse.ArgumentParser()
    for name in ('input-cases','source-cases','data-root','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--workers',type=int,choices=(1,2,4),default=4)
    parser.add_argument('--witness',action='store_true')
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cases=json.loads(Path(args.input_cases).read_text())['cases']
    assert len(cases)==8 and len({(v['case'],v['split']) for v in cases})==8
    commands=[];before=time.monotonic()
    for case in cases:
        folder=out/case['split']/case['case']
        native=[sys.executable,str(HERE/'extreme_transport.py'),'--input-cases',args.input_cases,
            '--source-cases',args.source_cases,'--data-root',args.data_root,'--case',case['case'],
            '--split',case['split'],'--output',str(folder)]
        if args.witness:native.append('--witness')
        verifier=[sys.executable,str(HERE/'verify_extreme_transport.py'),'--input',str(folder),
            '--input-cases',args.input_cases,'--output',str(folder/'independent.json')]
        commands.append(dict(case=case['case'],split=case['split'],folder=str(folder),
                              native=native,verifier=verifier))
    dump(out/'commands.json',commands)
    def run(job):
        logdir=out/'logs'/job['split'];logdir.mkdir(parents=True,exist_ok=True)
        result=dict(case=job['case'],split=job['split'],folder=job['folder'],commands=[])
        for kind in ('native','verifier'):
            if kind=='verifier' and result['commands'][0]['actual_terminal_exit']:break
            with (logdir/(job['case']+'-'+kind+'.log')).open('w') as log:
                child=subprocess.Popen(job[kind],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                dump(logdir/(job['case']+'-'+kind+'-pid.json'),dict(pid=child.pid,command=job[kind]))
                actual=child.wait()
            result['commands'].append(dict(kind=kind,command=job[kind],actual_terminal_exit=actual))
        result['passed']=len(result['commands'])==2 and all(v['actual_terminal_exit']==0 for v in result['commands'])
        if result['passed']:result['replay_qualified']=json.loads((Path(job['folder'])/'COMPLETE.json').read_text())['replay_qualified']
        dump(logdir/(job['case']+'-terminal.json'),result)
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,job) for job in commands]
        while not all(f.done() for f in futures):
            finished=[f.result() for f in futures if f.done()]
            dump(out/'progress.json',dict(pid=os.getpid(),total=8,completed=len(finished),
                passed=sum(v['passed'] for v in finished),seconds=time.monotonic()-before))
            time.sleep(5)
        results=[f.result() for f in futures]
    passed=all(v['passed'] for v in results)
    dump(out/'TERMINAL.json',dict(passed=passed,cases=results,witness=args.witness,
        all_replay_qualified=passed and all(v.get('replay_qualified',False) for v in results),
        workers=args.workers,seconds=time.monotonic()-before))
    print('EXTREME_STUDY_TERMINAL',passed,flush=True)
    if not passed:sys.exit(1)


if __name__=='__main__':main()
