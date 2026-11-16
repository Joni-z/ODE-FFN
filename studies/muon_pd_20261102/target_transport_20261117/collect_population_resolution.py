"""Read the original remote queue, copy once terminal, analyze frozen source."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from population_resolution import read, dump, sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--host',default='145');p.add_argument('--remote',required=True)
    p.add_argument('--local',type=Path,required=True);p.add_argument('--analysis-python',required=True)
    a=p.parse_args();state=a.local/'collector2-progress.json';assert not state.exists()
    probe=("import json;from pathlib import Path;p=Path("+repr(a.remote)+
           ");t=p/'TERMINAL.json';print(json.dumps({'terminal':json.loads(t.read_text()) if t.exists() else None,"
           "'progress':json.loads((p/'progress.json').read_text())}))")
    remote_command=shlex.join(['python3','-c',probe])
    started=time.time();print('POPULATION_COLLECTOR_PID',os.getpid(),flush=True)
    while True:
        try:
            result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',a.host,
                remote_command],capture_output=True,text=True,timeout=30)
            if result.returncode:raise RuntimeError(result.stderr[-1000:])
            data=json.loads(result.stdout)
            record=dict(pid=os.getpid(),unix_time=time.time(),stage='waiting-original-queue',
                source_sha256=sha(__file__),remote_queue=data['progress']['pid'],
                remote_progress=data['progress'],terminal=data['terminal'])
            dump(state,record)
            with (a.local/'collector2-history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if data['terminal'] is not None:break
        except (OSError,ValueError,subprocess.SubprocessError,RuntimeError) as error:
            dump(state,dict(pid=os.getpid(),unix_time=time.time(),stage='read-failure-no-compute-restart',error=str(error)))
        time.sleep(30)
    assert not (a.local/'formal').exists()
    copied=subprocess.run(['scp','-q','-r',a.host+':'+a.remote,str(a.local)],capture_output=True,text=True)
    dump(a.local/'collector2-copy.json',dict(actual_terminal_exit=copied.returncode,stderr=copied.stderr))
    assert copied.returncode==0
    if not data['terminal']['passed']:
        dump(state,dict(pid=os.getpid(),stage='failed-cohort-preserved'));raise SystemExit(1)
    freeze=read(a.local/'analysis-source-freeze.json');repo=Path(freeze['source_root'])
    assert all(sha(repo/f)==h for f,h in freeze['files'].items())
    command=[a.analysis_python,str(repo/'studies/muon_pd_20261102/target_transport_20261117/analyze_population_resolution.py'),
        '--input',str(a.local/'formal'),'--output',str(a.local/'analysis')]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (a.local/'collector2-analysis.log').open('x') as log:result=subprocess.run(command,stdout=log,stderr=log,env=env)
    assert all(sha(repo/f)==h for f,h in freeze['files'].items())
    dump(a.local/'collector2-terminal.json',dict(passed=result.returncode==0,
        analysis_actual_terminal_exit=result.returncode,command=command,source_commit=freeze['commit'],
        source_files_unchanged=len(freeze['files']),collector_sha256=sha(__file__),seconds=time.time()-started))
    dump(state,dict(pid=os.getpid(),stage='analysis-terminal',passed=result.returncode==0))
    print('POPULATION_COLLECTOR_TERMINAL',result.returncode,flush=True);raise SystemExit(result.returncode)


if __name__=='__main__':main()
