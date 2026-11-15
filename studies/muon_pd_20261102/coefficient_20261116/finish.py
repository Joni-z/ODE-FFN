"""Follow the native queue with all registered common-bank evaluations."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from coefficient_trial import HERE,TAG,base

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--anchors',required=True);p.add_argument('--numpy-python',required=True)
    a=p.parse_args();root=Path(a.root);out=root/'analysis';out.mkdir(exist_ok=False)
    def status(phase,**kw):base.dump(out/'status.json',dict(phase=phase,pid=os.getpid(),unix_time=time.time(),**kw))
    try:
        terminal=root/'output/monitor/queue-finished.json'
        while not terminal.exists():
            q=json.loads((root/'output/monitor/status.json').read_text())
            os.kill(q['queue_pid'],0)
            status('waiting_for_training',training_queue_pid=q['queue_pid']);time.sleep(20)
        q=json.loads(terminal.read_text());assert not q['failed'] and not q['unstarted'] and len(q['completed'])==2
        cases=json.loads(Path(a.anchors).read_text())['cases']
        count=0
        for split,steps in [('selection',['15000','20000']),('confirmation',['20000'])]:
            for case in cases:
                folder=Path(case['folder']);target=out/split/case['id']
                cmd=[sys.executable,str(HERE/'coefficient_evaluate.py'),'--folder',str(folder),'--data-root',a.data_root,
                     '--assets',str(root/'assets'),'--output',str(target),'--device','cpu','--split',split,'--steps',*steps]
                status('evaluating',split=split,case=case['id'],completed=count,total=20)
                with (out/(split+'-'+case['id']+'.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
                count+=1
        status('independent_recomputation',completed=count,total=20)
        code=('import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);'
              'from analyze_screen import collect;root=Path(sys.argv[2]);'
              'collect(root,root,split="selection",expected_cases=10);'
              'collect(root,root,split="confirmation",expected_cases=10)')
        subprocess.run([a.numpy_python,'-c',code,str(HERE.parent/'response_screen_20261115'),str(out)],check=True)
        base.dump(out/'COMPLETE.json',dict(passed=True,training_cases=2,selection_cases=10,confirmation_cases=10,
                  assets_sha256=base.digest(root/'assets/manifest.json'),anchors_sha256=base.digest(a.anchors),
                  source_sha256=base.digest(__file__)))
        status('complete',completed=count,total=20)
    except Exception as e:
        status('failed',error=repr(e));raise

if __name__=='__main__':main()
