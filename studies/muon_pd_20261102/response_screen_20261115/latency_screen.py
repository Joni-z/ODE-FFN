"""Same-host frozen forward timings after training has stopped."""
import argparse,json,time,statistics
from pathlib import Path
from screen_trial import base,torch
from evaluate_screen import ev

def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
 a=p.parse_args();torch.set_num_threads(1);rows=[]
 for file in sorted((Path(a.runtime)/'selection').glob('*/binding.json')):
  b=json.loads(file.read_text());cfg=b['source_config'];c=cfg['cell'];cache=base.verify_data(a.data_root,c)
  model,identity=ev.restore(Path(b['source_folder']),cfg,cache/'manifest.json',20000,'cpu')
  snapshot=ev.tensor_snapshot(model);dim=model.dim;torch.manual_seed(731)
  x=torch.randn(512,dim);t=torch.rand(512,1)
  with torch.no_grad():
   for _ in range(10):model(x,t)
   times=[]
   for _ in range(3):
    before=time.perf_counter()
    for _ in range(20):model(x,t)
    times.append((time.perf_counter()-before)/20)
  ev.same_tensors(model,snapshot)
  rows.append(dict(id=c['id'],response=c['response'],order=c['order'],dataset=c['dataset'],median_seconds_per_forward=statistics.median(times),repeats=times,parameters=model.parameter_count(),**identity))
 assert len(rows)==36;base.dump(a.output,dict(passed=True,batch512=True,device='cpu',rows=rows,source_sha256=base.digest(__file__)))

if __name__=='__main__':main()
