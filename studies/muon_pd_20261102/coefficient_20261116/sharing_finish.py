"""Two-worker evaluation of the operator-sharing control and fixed references."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
from sharing_trial import HERE,base

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--data-root',required=True);p.add_argument('--numpy-python',required=True)
    a=p.parse_args();root=Path(a.root);out=root/'sharing-analysis';out.mkdir(exist_ok=False)
    def phase(name,**kw):base.dump(out/'status.json',dict(phase=name,pid=os.getpid(),unix_time=time.time(),**kw))
    try:
        anchors=json.loads((root/'sharing-anchors.json').read_text());assert len(anchors['cases'])==8
        assert base.digest(root/'sharing-assets/manifest.json')==anchors['assets_manifest_sha256']
        terminal=root/'sharing-output/monitor/queue-finished.json'
        while not terminal.exists():
            status=json.loads((root/'sharing-output/monitor/status.json').read_text())
            os.kill(status['queue_pid'],0);phase('waiting_for_training',queue_pid=status['queue_pid']);time.sleep(20)
        done=json.loads(terminal.read_text());assert not done['failed'] and not done['unstarted'] and len(done['completed'])==2
        for row in anchors['cases']:
            folder=Path(row['folder']);assert (folder/'ADAPTER_COMPLETE').exists()
            for relative,digest in row.get('bound_files',{}).items():assert base.digest(folder/relative)==digest
            cfg=json.loads((folder/'config.json').read_text());assert cfg['cell']['id']==row['id'] and cfg['steps']==20000 and cfg['seeds']==[0]
        tasks=[(split,steps,row) for split,steps in [('selection',['15000','20000']),('confirmation',['20000'])] for row in anchors['cases']]
        def evaluate(task):
            split,steps,row=task;cmd=[sys.executable,str(HERE/'sharing_evaluate.py'),'--folder',row['folder'],'--data-root',a.data_root,
                '--assets',str(root/'sharing-assets'),'--output',str(out/split/row['id']),'--device','cpu','--split',split,'--steps',*steps]
            with (out/(split+'-'+row['id']+'.log')).open('x') as log:
                proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            complete=out/split/row['id']/'COMPLETE.json'
            assert proc.returncode==0 and complete.exists(),(row['id'],split,proc.returncode)
            assert json.loads(complete.read_text())['passed']
            return dict(id=row['id'],split=split,exit_code=proc.returncode,completion_sha256=base.digest(complete),command=cmd)
        receipts=[];phase('evaluating',completed=0,total=len(tasks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(evaluate,task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                receipts.append(future.result());base.dump(out/'receipts.json',receipts);phase('evaluating',completed=len(receipts),total=len(tasks))
        phase('independent_recomputation')
        code=('import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from analyze_screen import collect;'
              'r=Path(sys.argv[2]);collect(r,r,split="selection",expected_cases=8);collect(r,r,split="confirmation",expected_cases=8)')
        subprocess.run([a.numpy_python,'-c',code,str(HERE.parent/'response_screen_20261115'),str(out)],check=True)
        base.dump(out/'COMPLETE.json',dict(passed=True,training_cases=2,reference_cases=6,selection_cases=8,confirmation_cases=8,
            anchors_sha256=base.digest(root/'sharing-anchors.json'),source_sha256=base.digest(__file__)))
        phase('complete',completed=len(tasks),total=len(tasks))
    except Exception as e:phase('failed',error=repr(e));raise

if __name__=='__main__':main()
