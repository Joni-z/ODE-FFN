"""Independent full-bank CPU/CUDA/refinement score recomputation."""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
from verify_numerical import sha,sw2

def main():
    p=argparse.ArgumentParser();p.add_argument('--portable',required=True);p.add_argument('--runtime',required=True);p.add_argument('--output',required=True)
    p.add_argument('--runner-source',default=str(Path(__file__).with_name('sharing_gpu.py')));a=p.parse_args()
    root=Path(a.portable);runtime=Path(a.runtime);done=json.loads((runtime/'COMPLETE.json').read_text())
    assert done['passed'] and done['source_sha256']==sha(a.runner_source) and done['portable_manifest_sha256']==sha(root/'portable.json')
    cases=json.loads((root/'portable.json').read_text())['cases'];assert len(cases)==4 and len(done['rows'])==16
    rows=[];maximum=0.;primary=[]
    for case in cases:
        for key in ['checkpoint','config','dataset_manifest','assets','cpu_field']:assert sha(root/case[key])==case[key+'_sha256']
        with np.load(root/case['assets']) as z:data={k:z[k].copy() for k in z.files}
        target=np.concatenate([data[f'target_{i}'] for i in range(8)]);cpu=[]
        for file,digest in zip(case['cpu_samples'],case['cpu_sample_sha256']):assert sha(root/file)==digest;cpu.append(np.load(root/file))
        cpu_score=sw2(np.concatenate(cpu),target,data['directions']);assert abs(cpu_score-case['original_cpu_pooled_sw2'])<1e-10
        group=[r for r in done['rows'] if r['id']==case['id']]
        assert {(r['inner_substeps'],r['heun_steps']) for r in group}=={(4,128),(4,256),(4,512),(16,128)}
        values={}
        for row in group:
            assert row['checkpoint_sha256']==case['checkpoint_sha256'] and row['assets_sha256']==case['assets_sha256']
            file=runtime/row['generated_file'];assert sha(file)==row['generated_sha256'];x=np.load(file);assert x.shape==target.shape and np.isfinite(x).all()
            score=sw2(x,target,data['directions']);bank=[sw2(x[i*4096:(i+1)*4096],target[i*4096:(i+1)*4096],data['directions']) for i in range(8)]
            error=max(abs(score-row['pooled_sw2']),float(np.max(np.abs(np.array(bank)-row['bank_sw2']))));maximum=max(maximum,error);assert error<1e-10
            values[(row['inner_substeps'],row['heun_steps'])]=x
            entry=dict(id=case['id'],dataset=case['dataset'],family=case['family'],inner=row['inner_substeps'],outer=row['heun_steps'],pooled_sw2=score,bank_sw2=bank,
                       sample_sha256=sha(file),checkpoint_sha256=case['checkpoint_sha256'],assets_sha256=case['assets_sha256'])
            if (entry['inner'],entry['outer'])==(4,128):
                entry['cpu_primary_score_difference']=abs(score-cpu_score);assert entry['cpu_primary_score_difference']<1e-4;primary.append(entry['cpu_primary_score_difference'])
            rows.append(entry)
        reference=values[(4,512)].astype(float)
        for row in rows:
            if row['id']==case['id']:
                delta=values[(row['inner'],row['outer'])].astype(float)-reference;row['rms_euclidean_shift_to_inner4_outer512']=float(np.sqrt(np.mean(np.sum(delta**2,axis=1))))
    gaps=[]
    for dataset in {c['dataset'] for c in cases}:
        for inner,outer in [(4,128),(4,256),(4,512),(16,128)]:
            group=[r for r in rows if r['dataset']==dataset and r['inner']==inner and r['outer']==outer]
            left=next(r for r in group if r['family']=='state_midpoint4');right=next(r for r in group if r['family']=='stage_operator_midpoint4')
            delta=np.array(right['bank_sw2'])-np.array(left['bank_sw2'])
            gaps.append(dict(dataset=dataset,inner=inner,outer=outer,released_minus_shared=right['pooled_sw2']-left['pooled_sw2'],bank_differences=delta.tolist(),shared_better_banks=int(np.count_nonzero(delta>0))))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);shutil.copy2(runtime/'COMPLETE.json',out/'gpu-completion.json')
    if (runtime/'gpu-history.jsonl').exists():shutil.copy2(runtime/'gpu-history.jsonl',out/'gpu-history.jsonl')
    (out/'scores.json').write_text(json.dumps(dict(rows=rows,gaps=gaps),indent=2,allow_nan=False)+'\n')
    with (out/'scores.csv').open('w') as f:
        fields=['id','dataset','family','inner','outer','pooled_sw2','rms_euclidean_shift_to_inner4_outer512','sample_sha256','checkpoint_sha256'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    result=dict(passed=True,models=4,pooled_rows=16,bank_rows=128,max_numpy_difference=maximum,max_cpu_primary_score_difference=max(primary),
                source_sha256=sha(__file__),runner_source_sha256=sha(a.runner_source),portable_manifest_sha256=sha(root/'portable.json'))
    (out/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
