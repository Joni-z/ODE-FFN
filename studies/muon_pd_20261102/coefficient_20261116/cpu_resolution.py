"""CPU fallback: full-bank internal refinement or bounded outer resolution."""
import argparse,json,time
from pathlib import Path
import numpy as np
from gpu_resolution import base,torch,load,arrays,STATE_FAMILIES

def generated_bankwise(model,source,steps,progress=None):
    """Keep every source and its order; call the unchanged sampler per bank."""
    parts=[]
    for offset in range(0,len(source),4096):
        parts.append(base.train.generated_samples(model,source[offset:offset+4096],steps))
        if progress is not None:progress(min(offset+4096,len(source)))
    return torch.cat(parts)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);p.add_argument('--mode',choices=['full-internal','outer-subset','full-outer'],required=True);p.add_argument('--output',required=True)
    p.add_argument('--inner-substeps',nargs='+',type=int,choices=[4,8,16],default=[4,16])
    p.add_argument('--bank-batches',action='store_true',help='Evaluate the same ordered sources in batches of4096')
    a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    case=next(c for c in json.loads((root/'portable.json').read_text())['cases'] if c['id']==a.case)
    model=load(root,case,'cpu');data=arrays(root,case);n=32768 if a.mode!='outer-subset' else 512
    if a.mode!='outer-subset':
        assert case['family'] in STATE_FAMILIES
        source=torch.from_numpy(np.concatenate([data[f'source_{i}'] for i in range(8)]));target=torch.from_numpy(np.concatenate([data[f'target_{i}'] for i in range(8)]))
        if a.mode=='full-internal':
            assert a.inner_substeps == sorted(set(a.inner_substeps)) and a.inner_substeps[0] == 4 and a.inner_substeps[-1] == 16
            configs=[(steps,128) for steps in a.inner_substeps]
        else:configs=[(4,128),(4,256),(4,512)]
    else:
        source=torch.from_numpy(data['source_0'][:512]);target=torch.from_numpy(data['target_0'][:512]);configs=[(4,64),(4,128),(4,256),(4,512)]
    directions=torch.from_numpy(data['directions']).double();rows=[];values={}
    for inner,outer in configs:
        for _,layer in model.ode_layers():
            if case['family'] in STATE_FAMILIES:layer.substeps=inner
        started=time.monotonic()
        if a.bank_batches:
            def progress(completed):
                base.dump(out/'sampling-progress.json',dict(inner_substeps=inner,heun_steps=outer,
                    samples_completed=completed,total_samples=len(source),seconds=time.monotonic()-started))
            generated=generated_bankwise(model,source,outer,progress)
        else:
            generated=base.train.generated_samples(model,source,outer)
        assert torch.isfinite(generated).all()
        score=base.train.sliced_w2(generated.double(),target.double(),directions)
        values[(inner,outer)]=generated.numpy();filename=f'inner{inner}-heun{outer}.npy';np.save(out/filename,values[(inner,outer)],allow_pickle=False)
        row=dict(id=case['id'],dataset=case['dataset'],family=case['family'],mode=a.mode,inner_substeps=inner if case['family'] in STATE_FAMILIES else None,
            heun_steps=outer,nfe=2*outer,samples=n,pooled_sw2=score,seconds=time.monotonic()-started,
            sampling_batch_size=min(4096,n) if a.bank_batches else n,
            generated_file=filename,generated_sha256=base.digest(out/filename),checkpoint_sha256=case['checkpoint_sha256'],assets_sha256=case['assets_sha256'])
        if a.mode!='outer-subset' and inner==4 and outer==128:
            original=np.concatenate([np.load(root/f) for f in case['cpu_samples']])
            row['original_score_difference']=abs(score-case['original_cpu_pooled_sw2'])
            row['original_sample_max_error']=float(np.max(np.abs(values[(inner,outer)]-original)))
            assert row['original_score_difference']<1e-4
        if a.mode!='outer-subset':row['bank_sw2']=[base.train.sliced_w2(generated[i*4096:(i+1)*4096].double(),target[i*4096:(i+1)*4096].double(),directions) for i in range(8)]
        rows.append(row);base.dump(out/'progress.json',dict(rows=rows))
    reference=values[(16,128)] if a.mode=='full-internal' else values[(4,512)]
    for r in rows:
        key=(r['inner_substeps'] or 4,r['heun_steps']);delta=values[key]-reference
        r['rms_coordinate_shift_to_finest']=float(np.sqrt(np.mean(delta**2)))
        r['rms_euclidean_shift_to_finest']=float(np.sqrt(np.mean(np.sum(delta**2,axis=1))))
    base.dump(out/'COMPLETE.json',dict(passed=True,case=case,mode=a.mode,rows=rows,
        bank_batches=a.bank_batches,source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,case=a.case,mode=a.mode)))

if __name__=='__main__':main()
