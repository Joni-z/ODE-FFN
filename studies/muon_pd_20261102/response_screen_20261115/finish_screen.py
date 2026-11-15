"""Watch one bounded CPU queue, then freeze selection and run confirmation."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from screen_trial import HERE,base

def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--data-root',required=True);p.add_argument('--assets',required=True)
 p.add_argument('--numpy-python',required=True);p.add_argument('--output',required=True)
 a=p.parse_args();root=Path(a.runtime);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 def phase(name,**kw):base.dump(out/'status.json',dict(phase=name,unix_time=time.time(),**kw))
 try:
  terminal=root/'monitor/queue-finished.json'
  while not terminal.exists():
   phase('waiting_for_training',monitor=str(root/'monitor/status.json'));time.sleep(30)
  done=json.loads(terminal.read_text());assert not done['failed'] and not done['unstarted'] and len(done['completed'])==36
  phase('forward_timings')
  subprocess.run([sys.executable,str(HERE/'latency_screen.py'),'--runtime',str(root),'--data-root',a.data_root,'--output',str(out/'latency.json')],check=True)
  phase('selection_recomputation')
  subprocess.run([a.numpy_python,str(HERE/'analyze_screen.py'),'--runtime',str(root),'--output',str(out),'--latency',str(out/'latency.json')],check=True)
  shortlist=out/'shortlist.json';frozen=base.digest(shortlist);names=json.loads(shortlist.read_text())['selected_responses']
  phase('confirmation',selected=names,shortlist_sha256=frozen)
  count=0
  for file in sorted((root/'selection').glob('*/binding.json')):
   b=json.loads(file.read_text());c=b['source_config']['cell']
   if c['response'] not in names and c['role']!='ordinary_control_cpu':continue
   target=out/'confirmation'/c['id']
   subprocess.run([sys.executable,str(HERE/'evaluate_screen.py'),'--folder',b['source_folder'],'--data-root',a.data_root,
        '--assets',a.assets,'--output',str(target),'--device','cpu','--split','confirmation','--steps','20000'],check=True)
   assert base.digest(shortlist)==frozen;count+=1
   phase('confirmation',selected=names,completed=count,shortlist_sha256=frozen)
  phase('confirmation_recomputation',confirmation_cases=count)
  subprocess.run([a.numpy_python,str(HERE/'analyze_screen.py'),'--runtime',str(out),'--output',str(out),'--confirmation-only','--expected-cases',str(count)],check=True)
  base.dump(out/'COMPLETE.json',dict(passed=True,selected=names,confirmation_cases=count,shortlist_sha256=frozen,source_sha256=base.digest(__file__)))
  phase('complete',selected=names,confirmation_cases=count)
 except Exception as e:
  phase('failed',error=repr(e));raise

if __name__=='__main__':main()
