"""CPU fallback: full-bank internal refinement or bounded outer resolution."""
import argparse,json,time
from pathlib import Path
import numpy as np
from gpu_resolution import base,torch,load,arrays

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);p.add_argument('--mode',choices=['full-internal','outer-subset'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    case=next(c for c in json.loads((root/'portable.json').read_text())['cases'] if c['id']==a.case)
    model=load(root,case,'cpu');data=arrays(root,case);n=32768 if a.mode=='full-internal' else 512
    if a.mode=='full-internal':
        assert case['family'] in ['state_midpoint4','free_stage_midpoint4']
        source=torch.from_numpy(np.concatenate([data[f'source_{i}'] for i in range(8)]));target=torch.from_numpy(np.concatenate([data[f'target_{i}'] for i in range(8)]))
        configs=[(4,128),(16,128)]
    else:
        source=torch.from_numpy(data['source_0'][:512]);target=torch.from_numpy(data['target_0'][:512]);configs=[(4,64),(4,128),(4,256),(4,512)]
    directions=torch.from_numpy(data['directions']).double();rows=[];values={}
    for inner,outer in configs:
        for _,layer in model.ode_layers():
            if case['family'] in ['state_midpoint4','free_stage_midpoint4']:layer.substeps=inner
        started=time.monotonic();generated=base.train.generated_samples(model,source,outer);assert torch.isfinite(generated).all()
        score=base.train.sliced_w2(generated.double(),target.double(),directions)
        values[(inner,outer)]=generated.numpy();filename=f'inner{inner}-heun{outer}.npy';np.save(out/filename,values[(inner,outer)],allow_pickle=False)
        row=dict(id=case['id'],dataset=case['dataset'],family=case['family'],mode=a.mode,inner_substeps=inner if case['family'] in ['state_midpoint4','free_stage_midpoint4'] else None,
            heun_steps=outer,nfe=2*outer,samples=n,pooled_sw2=score,seconds=time.monotonic()-started,
            generated_file=filename,generated_sha256=base.digest(out/filename),checkpoint_sha256=case['checkpoint_sha256'],assets_sha256=case['assets_sha256'])
        if a.mode=='full-internal' and inner==4:
            original=np.concatenate([np.load(root/f) for f in case['cpu_samples']])
            row['original_score_difference']=abs(score-case['original_cpu_pooled_sw2'])
            row['original_sample_max_error']=float(np.max(np.abs(values[(inner,outer)]-original)))
            assert row['original_score_difference']<1e-4
        if a.mode=='full-internal':row['bank_sw2']=[base.train.sliced_w2(generated[i*4096:(i+1)*4096].double(),target[i*4096:(i+1)*4096].double(),directions) for i in range(8)]
        rows.append(row);base.dump(out/'progress.json',dict(rows=rows))
    reference=values[(16,128)] if a.mode=='full-internal' else values[(4,512)]
    for r in rows:
        key=(r['inner_substeps'] or 4,r['heun_steps']);delta=values[key]-reference
        r['rms_coordinate_shift_to_finest']=float(np.sqrt(np.mean(delta**2)))
        r['rms_euclidean_shift_to_finest']=float(np.sqrt(np.mean(np.sum(delta**2,axis=1))))
    base.dump(out/'COMPLETE.json',dict(passed=True,case=case,mode=a.mode,rows=rows,source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,case=a.case,mode=a.mode)))

if __name__=='__main__':main()
