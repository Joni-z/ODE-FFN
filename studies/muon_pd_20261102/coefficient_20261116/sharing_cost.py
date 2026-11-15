"""Same-host batch512 CPU work and timing for the complete sharing cohort."""
import argparse,json,time
from pathlib import Path
import numpy as np
from torch.utils._python_dispatch import TorchDispatchMode
from sharing_trial import base,torch
import sharing_evaluate as evaluation
ev=evaluation.evaluator.ev

class DenseMACs(TorchDispatchMode):
    def __init__(self):self.macs=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        name=str(func)
        if name=='aten.mm.default':
            a,b=args[:2];self.macs+=a.shape[0]*a.shape[1]*b.shape[1]
        elif name=='aten.addmm.default':
            a,b=args[1:3];self.macs+=a.shape[0]*a.shape[1]*b.shape[1]
        elif name=='aten.bmm.default':
            a,b=args[:2];self.macs+=a.shape[0]*a.shape[1]*a.shape[2]*b.shape[2]
        return func(*args,**(kwargs or {}))

def summary(values):return dict(median=float(np.median(values)),q25=float(np.quantile(values,.25)),q75=float(np.quantile(values,.75)),values=values)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root);out=Path(a.output);assert not out.exists();torch.set_num_threads(1)
    with DenseMACs() as test:torch.nn.functional.linear(torch.zeros(2,3),torch.zeros(4,3),torch.zeros(4))
    assert test.macs==24
    loaded=[];rows=[]
    for case in json.loads((root/'sharing-anchors.json').read_text())['cases']:
        folder=Path(case['folder']);cfg=json.loads((folder/'config.json').read_text());cell=cfg['cell'];assert (folder/'ADAPTER_COMPLETE').exists()
        cache=base.verify_data(a.data_root,cell);model,identity=ev.restore(folder,cfg,cache/'manifest.json',20000,'cpu')
        arrays,fixed,binding=evaluation.evaluator.assets(root/'sharing-assets',cell['dataset'],'selection','cpu')
        x=fixed['x'][:512];t=fixed['t'][:512];source=torch.from_numpy(arrays['source_0'][:512])
        with torch.no_grad():
            with DenseMACs() as counter:model(x,t)
            for _ in range(5):model(x,t)
        row=dict(id=case['id'],dataset=cell['dataset'],family=cell['family'],parameters=model.parameter_count(),
                 dense_macs_per_example=counter.macs//512,batch=512,checkpoint=identity,assets_sha256=binding['sha256'],
                 forward_cpu_seconds=[],forward_wall_seconds=[],sample_cpu_seconds=[],sample_wall_seconds=[])
        assert counter.macs%512==0
        loaded.append((model,x,t,source,ev.tensor_snapshot(model)));rows.append(row)
    assert len(loaded)==8;rng=np.random.default_rng(0)
    with torch.no_grad():
        for _ in range(9):
            for index in rng.permutation(8):
                model,x,t,_,_=loaded[index];cpu=time.process_time();wall=time.perf_counter()
                for _ in range(10):model(x,t)
                rows[index]['forward_wall_seconds'].append((time.perf_counter()-wall)/10)
                rows[index]['forward_cpu_seconds'].append((time.process_time()-cpu)/10)
        for repeat in range(3):
            for index in rng.permutation(8):
                model,_,_,source,_=loaded[index];cpu=time.process_time();wall=time.perf_counter()
                generated=base.train.generated_samples(model,source,128);assert torch.isfinite(generated).all()
                rows[index]['sample_wall_seconds'].append(time.perf_counter()-wall)
                rows[index]['sample_cpu_seconds'].append(time.process_time()-cpu)
            base.dump(out.with_suffix('.progress.json'),dict(sample_rounds_completed=repeat+1,total=3))
    for row,(model,_,_,_,snapshot) in zip(rows,loaded):
        ev.same_tensors(model,snapshot)
        for key in ['forward_cpu_seconds','forward_wall_seconds','sample_cpu_seconds','sample_wall_seconds']:row[key]=summary(row[key])
    for dataset in {r['dataset'] for r in rows}:
        selected=[r for r in rows if r['dataset']==dataset and r['family'] in ['state_midpoint4','stage_operator_midpoint4']]
        assert len(selected)==2 and selected[0]['dense_macs_per_example']==selected[1]['dense_macs_per_example']
    base.dump(out,dict(passed=True,scope='batch512 CPU time, not GPU throughput or total training cost; dense MACs exclude pointwise operations',
                       forward_repetitions=9,forwards_per_repetition=10,sampling_repetitions=3,heun_steps=128,rows=rows,
                       environment=dict(torch=torch.__version__,threads=torch.get_num_threads()),source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,cases=len(rows),output=str(out))))

if __name__=='__main__':main()
