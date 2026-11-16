"""Monitored fixed-cohort CPU queue; preserve actual native/verifier exits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ('cases','data-root','assets','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--witness',action='store_true')
    parser.add_argument('--workers',type=int,default=4,choices=(1,2,4))
    args = parser.parse_args()
    assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    out = Path(args.output);out.mkdir(parents=True,exist_ok=False)
    before = time.monotonic()
    cases = json.loads(Path(args.cases).read_text())['cases']
    jobs = [(case,split) for case in cases for split in ('selection','confirmation')]
    commands, results = [], []
    for case,split in jobs:
        folder = out/split/case['id']
        native = [sys.executable,str(HERE/'evaluate_transport.py'),'--cases',args.cases,
            '--case',case['id'],'--data-root',args.data_root,'--assets',args.assets,
            '--split',split,'--output',str(folder)]
        if args.witness:native.append('--witness')
        verifier = [sys.executable,str(HERE/'verify_transport.py'),'--input',str(folder),
                    '--output',str(folder/'independent.json')]
        commands.append(dict(id=case['id'],split=split,folder=str(folder),native=native,verifier=verifier))
    dump(out/'commands.json',commands)

    def run(job: dict) -> dict:
        logdir = out/'logs'/job['split'];logdir.mkdir(parents=True,exist_ok=True)
        row = dict(id=job['id'],split=job['split'],folder=job['folder'],commands=[])
        for kind in ('native','verifier'):
            if kind == 'verifier' and row['commands'][0]['actual_terminal_exit']:
                break
            with (logdir/(job['id']+'-'+kind+'.log')).open('w') as log:
                process = subprocess.Popen(job[kind],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                dump(logdir/(job['id']+'-'+kind+'-pid.json'),dict(pid=process.pid,command=job[kind]))
                actual = process.wait()
            row['commands'].append(dict(kind=kind,command=job[kind],actual_terminal_exit=actual))
        row['passed'] = len(row['commands'])==2 and all(c['actual_terminal_exit']==0 for c in row['commands'])
        if row['passed']:
            result = json.loads((Path(job['folder'])/'COMPLETE.json').read_text())
            row['numerically_qualified'] = result['all_numerically_qualified']
        dump(logdir/(job['id']+'-terminal.json'),row)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run,job) for job in commands]
        while not all(f.done() for f in futures):
            completed = [f.result() for f in futures if f.done()]
            dump(out/'progress.json',dict(pid=os.getpid(),total=len(jobs),completed=len(completed),
                passed=sum(r['passed'] for r in completed),seconds=time.monotonic()-before,witness=args.witness))
            time.sleep(5)
        results = [f.result() for f in futures]
    passed = all(row['passed'] for row in results)
    qualified = passed and all(row.get('numerically_qualified',False) for row in results)
    dump(out/'TERMINAL.json',dict(passed=passed,all_numerically_qualified=qualified,
        witness=args.witness,cases=results,seconds=time.monotonic()-before,workers=args.workers))
    print(json.dumps(dict(passed=passed,all_numerically_qualified=qualified,output=str(out))))
    if not passed:sys.exit(1)


if __name__ == '__main__':main()
