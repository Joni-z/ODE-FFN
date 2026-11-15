"""End-to-end native200step throughput: one CPU worker versus two."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from screen_trial import HERE,TAG,base

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);results=[]
 grid=json.loads((HERE/'grid.json').read_text());cell=next(c for c in grid['cells'] if c['dataset']=='rotated_product_bimodal_16d' and c['response']=='tanh' and c['order']==8)
 for workers in (1,2):
  active=[];before=time.monotonic()
  for i in range(workers):
   root=out/f'{workers}-workers-{i}'
   cmd=[sys.executable,str(HERE/'screen_trial.py'),'--cell',cell['id'],'--data-root',a.data_root,'--output',str(root),'--device','cpu','--steps','200']
   log=(out/f'{workers}-{i}.log').open('x');active.append((subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log,root))
  times=[]
  for proc,log,root in active:
   code=proc.wait();log.close();assert code==0
   f=root/TAG/cell['dataset']/cell['id']/'seed000/metrics.jsonl'
   last=json.loads(f.read_text().splitlines()[-1]);assert last['step']==200;times.append(last['wall_seconds'])
  elapsed=time.monotonic()-before;results.append(dict(workers=workers,elapsed=elapsed,updates_per_second=workers*200/elapsed,native_wall_seconds=times))
 ratio=results[1]['updates_per_second']/results[0]['updates_per_second'];selected=2 if ratio>1.1 else 1
 base.dump(out/'PASS.json',dict(passed=True,results=results,two_worker_speedup=ratio,selected_workers=selected,source_sha256=base.digest(__file__)))
 print(json.dumps(dict(passed=True,workers=selected,speedup=ratio)))

if __name__=='__main__':main()
