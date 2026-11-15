"""Portable sharing cohort and guarded full-bank CUDA outer refinement."""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
from gpu_resolution import base,torch,load,arrays,exclusive_gpu

GPU4='GPU-6c644192-ce4a-5d10-b912-07a6fc24f352'

def export(root,data_root,out):
    out.mkdir(parents=True,exist_ok=False);rows=[]
    for row in json.loads((root/'sharing-anchors.json').read_text())['cases']:
        folder=Path(row['folder']);cfg=json.loads((folder/'config.json').read_text());c=cfg['cell']
        if c['family'] not in ['state_midpoint4','stage_operator_midpoint4']:continue
        cache=base.verify_data(data_root,c);assert (folder/'ADAPTER_COMPLETE').exists()
        target=out/'cases'/c['id'];target.mkdir(parents=True)
        def copy(source,name):
            file=target/name;shutil.copy2(source,file);return str(file.relative_to(out)),base.digest(file)
        cfgpath,cfghash=copy(folder/'config.json','config.json');ckpt,ckpthash=copy(folder/'checkpoints/step_0020000.pt','step_0020000.pt');manifest,mhash=copy(cache/'manifest.json','dataset-manifest.json')
        evaluator=root/'sharing-analysis/confirmation'/c['id'];done=json.loads((evaluator/'COMPLETE.json').read_text());assert done['passed'] and not done['witness']
        asset=root/'sharing-assets'/c['dataset']/'confirmation.npz';assetpath,assethash=copy(asset,'confirmation.npz')
        field,fieldhash=copy(evaluator/'field-20000.npy','field.npy');samples=[];hashes=[]
        for bank in range(8):
            name,digest=copy(evaluator/f'20000-b{bank}.npy',f'b{bank}.npy');samples.append(name);hashes.append(digest)
        summary=next(r for r in json.loads((evaluator/'summary.json').read_text()) if r['step']==20000)
        rows.append(dict(id=c['id'],dataset=c['dataset'],family=c['family'],config=cfgpath,config_sha256=cfghash,checkpoint=ckpt,checkpoint_sha256=ckpthash,
            dataset_manifest=manifest,dataset_manifest_sha256=mhash,assets=assetpath,assets_sha256=assethash,cpu_field=field,cpu_field_sha256=fieldhash,
            cpu_samples=samples,cpu_sample_sha256=hashes,original_cpu_pooled_sw2=summary['pooled_sw2']))
    assert len(rows)==4
    base.dump(out/'portable.json',dict(passed=True,cases=rows,source_sha256=base.digest(__file__),cohort='new sharing confirmation only'))
    print(json.dumps(dict(passed=True,exported_cases=4,output=str(out))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true');p.add_argument('--root',required=True);p.add_argument('--data-root');p.add_argument('--output',required=True);p.add_argument('--readiness')
    p.add_argument('--gpu-uuid',default=GPU4)
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if a.export:assert a.data_root;export(root,a.data_root,out);return
    assert a.readiness
    ready=json.loads(Path(a.readiness).read_text());assert ready['passed'] and ready['witness'] and ready['cases']==4
    assert ready['portable_manifest_sha256']==base.digest(root/'portable.json')
    selected=a.gpu_uuid
    def exclusive():exclusive_gpu(selected)
    exclusive();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available();base.verify_sources();cases=json.loads((root/'portable.json').read_text())['cases'];assert len(cases)==4;rows=[]
    for case in cases:
        exclusive();model=load(root,case);data=arrays(root,case)
        source=torch.from_numpy(np.concatenate([data[f'source_{i}'] for i in range(8)])).cuda()
        target=torch.from_numpy(np.concatenate([data[f'target_{i}'] for i in range(8)])).cuda().double();directions=torch.from_numpy(data['directions']).cuda().double()
        for steps in [128,256,512]:
            exclusive();torch.cuda.synchronize();start=time.monotonic()
            value=base.train.generated_samples(model,source,steps);assert torch.isfinite(value).all()
            score=base.train.sliced_w2(value.double(),target,directions)
            scores=[base.train.sliced_w2(value[i*4096:(i+1)*4096].double(),target[i*4096:(i+1)*4096],directions) for i in range(8)]
            torch.cuda.synchronize();seconds=time.monotonic()-start;name=case['id']+f'-heun{steps}.npy';np.save(out/name,value.cpu().numpy(),allow_pickle=False)
            row=dict(id=case['id'],dataset=case['dataset'],family=case['family'],heun_steps=steps,nfe=2*steps,samples=32768,pooled_sw2=score,bank_sw2=scores,
                seconds=seconds,generated_file=name,generated_sha256=base.digest(out/name),checkpoint_sha256=case['checkpoint_sha256'],assets_sha256=case['assets_sha256'])
            if steps==128:
                row['cpu_primary_score_difference']=abs(score-case['original_cpu_pooled_sw2']);assert row['cpu_primary_score_difference']<1e-4,(case['id'],row)
            rows.append(row);base.dump(out/'progress.json',dict(pid=os.getpid(),completed=len(rows),total=12,last=row))
            resource=subprocess.run(['nvidia-smi','-i',selected,'--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            with (out/'gpu-history.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),id=case['id'],steps=steps,resource=resource.stdout.strip()))+'\n')
            exclusive();print(json.dumps(dict(id=case['id'],steps=steps,sw2=score,seconds=seconds)),flush=True)
        del model;torch.cuda.empty_cache()
    base.dump(out/'COMPLETE.json',dict(passed=True,rows=rows,portable_manifest_sha256=base.digest(root/'portable.json'),source_sha256=base.digest(__file__),gpu_uuid=selected,torch=torch.__version__))
    print(json.dumps(dict(passed=True,rows=len(rows),output=str(out))))

if __name__=='__main__':main()
