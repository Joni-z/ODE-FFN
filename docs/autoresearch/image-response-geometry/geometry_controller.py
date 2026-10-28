"""One owned CPU child per phase with actual exits and primed resource sampling."""
import json,subprocess,time,traceback
import psutil
import geometry_study as g

def run(root):
    g.setup();g.verify(root);receipt=g.load(root/'documented-pass.json')
    assert receipt['status']=='PASS' and receipt['doc_vs_reality_mismatches']==[]
    assert receipt['preflight_sha256']==g.sha(root/'preflight.json') and all(v['exit_code']==0 for v in receipt['records'])
    out=root/'controller';out.mkdir();me=psutil.Process();g.write(out/'identity.json',dict(pid=me.pid,create_time=me.create_time()));phases=[];p=None
    try:
        for phase in ('measure','audit'):
            g.write(out/'status.json',dict(state='RUNNING',phase=phase));command=[g.PYTHON,'-B',str(root/'source/study.py'),phase,'--root',str(root)]
            with (out/f'{phase}.log').open('w') as f:p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
            proc=psutil.Process(p.pid);proc.cpu_percent();tick=time.monotonic();record=dict(phase=phase,command=command,pid=p.pid,create_time=proc.create_time());phases.append(record);g.write(out/'launches.json',phases)
            while p.poll() is None:
                if time.monotonic()-tick>1800:p.terminate();p.wait(timeout=30);record['exit_code']=p.returncode;raise TimeoutError(phase)
                try:
                    with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(phase=phase,seconds=time.monotonic()-tick,pid=p.pid,create_time=proc.create_time(),CPU_percent=proc.cpu_percent(),RSS=proc.memory_info().rss))+'\n')
                except psutil.NoSuchProcess:pass
                time.sleep(5)
            record.update(exit_code=p.returncode,seconds=time.monotonic()-tick);g.write(out/'launches.json',phases);assert p.returncode==0,(phase,p.returncode)
        g.verify(root);g.write(out/'status.json',dict(state='COMPLETE',phases=2));print('IMAGE_RESPONSE_CONTROLLER_COMPLETE',flush=True)
    except BaseException:
        if p is not None and p.poll() is None:p.terminate();p.wait(timeout=30)
        g.write(out/'launches.json',phases);g.write(out/'status.json',dict(state='FAILED',error=traceback.format_exc()));raise
