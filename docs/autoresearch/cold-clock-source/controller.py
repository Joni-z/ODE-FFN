"""Two owned training children; all subsequent audited phases are sequential."""
from __future__ import annotations
import json,subprocess,time
import psutil
import clock_source_study as study

def gate(root):
    study.verify(root);value=study.load(root/'documented-pass.json');assert value['status']=='PASS' and value['mismatches']==[] and len(value['records'])==2
    for action,interpreter,sentinel in (('preflight',study.TRAIN,'COLD_CLOCK_SOURCE_NATIVE_PASS'),('metric-preflight',study.METRICS,'COLD_CLOCK_SOURCE_METRIC_PASS')):
        row=next(v for v in value['records'] if v['action']==action);expected=f"{interpreter} -B {root/'source/study.py'} {action} --root {root}";assert row['documented_command']==row['executed_command']==expected;assert row['invocation_count']==1 and row['exit_code']==row['terminal_tool_result']['exit_code']==0;assert row['mismatches']==[] and row['sentinel']==sentinel and row['sentinel_observed'];assert value['preflight_hashes'][action]==study.sha(root/(action+'.json')) and study.load(root/(action+'.json'))['status']=='PASS'
def run(root):
    gate(root);out=root/'orchestrator';out.mkdir();(root/'output').mkdir();me=psutil.Process();study.write(out/'identity.json',dict(pid=me.pid,create_time=me.create_time()));source=root/'source/study.py'
    command=lambda executable,action,arm=None:[executable,'-B',str(source),action,'--root',str(root)]+(['--arm',arm] if arm else [])
    phases=[('training',[(arm,command(study.TRAIN,'cell',arm)) for arm in study.ARMS],7500),('audit-training',[('audit',command(study.TRAIN,'audit-training'))],1200),('sample',[('sample',command(study.TRAIN,'sample'))],3600),('audit-sampling',[('audit',command(study.TRAIN,'audit-sampling'))],3600),('evaluate',[('metrics',command(study.METRICS,'evaluate'))],1200),('audit-metrics',[('audit',command(study.METRICS,'audit-metrics'))],1200)]
    records=[];live=[]
    try:
        for phase,commands,ceiling in phases:
            tick=time.monotonic();live=[]
            for name,args in commands:
                with (out/f'{phase}-{name}.log').open('w') as f:child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT)
                process=psutil.Process(child.pid);process.cpu_percent();row=dict(phase=phase,arm=name,command=args,pid=child.pid,create_time=process.create_time());records.append(row);live.append((child,process,row));print(json.dumps(dict(state='STARTED',**row)),flush=True)
            study.write(out/'launches.json',records)
            while any(child.poll() is None for child,_,_ in live):
                assert time.monotonic()-tick<ceiling,(phase,'wall-time ceiling');resources=[]
                for child,process,row in live:
                    code=child.poll()
                    if code is not None:
                        row['exit_code']=code;assert code==0,row
                    else:
                        try:resources.append(dict(arm=row['arm'],pid=child.pid,CPU_percent=process.cpu_percent(),RSS=process.memory_info().rss))
                        except psutil.NoSuchProcess:pass
                state=dict(state='RUNNING',phase=phase,seconds=time.monotonic()-tick,resources=resources);study.append(out/'resources.jsonl',state);study.write(out/'status.json',state)
                if phase=='training':study.write(root/'output/status.json',dict(state='RUNNING',active=[r['arm'] for p,_,r in live if p.poll() is None]))
                time.sleep(5)
            for child,_,row in live:row.update(exit_code=child.wait(),phase_seconds=time.monotonic()-tick);assert row['exit_code']==0,row
            study.write(out/'launches.json',records)
            if phase=='training':study.write(root/'output/status.json',dict(state='COMPLETE',arms=study.ARMS))
            print(json.dumps(dict(state='PHASE_COMPLETE',phase=phase,seconds=time.monotonic()-tick)),flush=True)
        study.verify(root);study.write(out/'status.json',dict(state='COMPLETE',phases=6,new_training_trajectories=2,quality_conclusion_pending=True))
    except BaseException as error:
        for child,_,row in live:
            if child.poll() is None:child.terminate()
        for child,_,row in live:
            try:row['exit_code']=child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();row['exit_code']=child.wait()
        study.write(out/'launches.json',records);study.write(out/'status.json',dict(state='FAILED',error=repr(error),records=records));raise
