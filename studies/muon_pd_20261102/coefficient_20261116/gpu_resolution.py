"""Portable frozen CPU weights: native CUDA sampling and resolution controls."""
import argparse,json,time,os,subprocess
from pathlib import Path
import numpy as np
from coefficient_trial import base,torch,build_model as coefficient_builder
from state_trial import build_model as state_builder
from stage_trial import build_model as stage_builder

STATE_FAMILIES = ('state_midpoint4', 'free_stage_midpoint4',
                  'stage_operator_midpoint4', 'contrast_operator_midpoint4')

def verify_builder_sources(case):
    """New portable families bind their complete imported local model source."""
    files = case.get('builder_sources_sha256', {})
    if case['family'] == 'contrast_operator_midpoint4':
        assert files and any(p.endswith('/contrast_trial.py') for p in files)
    repo = Path(__file__).resolve().parents[3]
    for relative, digest in files.items():
        path = (repo/relative).resolve()
        assert path.is_relative_to(repo)
        assert base.digest(path) == digest, ('Builder source differs', relative)

def exclusive_gpu(selected):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==selected,'Bind exactly one registered physical GPU UUID'
    query=subprocess.run(['nvidia-smi','-i',selected,'--query-gpu=index','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    assert int(query.stdout.strip()) in [4,5,6,7],'Physical145 GPUs0-3 are prohibited'
    query=subprocess.run(['nvidia-smi','-i',selected,'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    foreign=[int(x.strip()) for x in query.stdout.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
    assert not foreign,('Selected GPU occupied; stop only our own work',foreign)

def builder(dim,family):
    if family=='contrast_operator_midpoint4':
        from contrast_trial import build_model as contrast_builder
        return contrast_builder(dim,family)
    if family=='stage_operator_midpoint4':
        from sharing_trial import build_model as sharing_builder
        return sharing_builder(dim,family)
    if family=='state_midpoint4':return state_builder(dim,family)
    if family=='free_stage_midpoint4':return stage_builder(dim,family)
    return coefficient_builder(dim,family)

def load(root,case,device='cuda'):
    verify_builder_sources(case)
    cfg=json.loads((root/case['config']).read_text());manifest=root/case['dataset_manifest']
    assert base.digest(root/case['checkpoint'])==case['checkpoint_sha256']
    assert base.digest(root/case['config'])==case['config_sha256']
    assert base.digest(manifest)==case['dataset_manifest_sha256']
    state=torch.load(root/case['checkpoint'],map_location='cpu',weights_only=False)
    assert state['step']==20000 and state['config_sha256']==case['config_sha256']
    assert state['dataset_manifest_sha256']==case['dataset_manifest_sha256']
    dim=json.loads(manifest.read_text())['datasets'][case['dataset']]['dim']
    torch.manual_seed(0);model=builder(dim,cfg['cell']['family']).to(device).eval();base.train.restore_model(model,state)
    return model

def arrays(root,case):
    assert base.digest(root/case['assets'])==case['assets_sha256']
    with np.load(root/case['assets']) as z:return {k:z[k].copy() for k in z.files}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--witness',action='store_true')
    p.add_argument('--gpu-uuid',default='GPU-8ffc220f-2b4f-5d14-0787-4f46ba75a023')
    a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
    selected=a.gpu_uuid
    def check_exclusive():exclusive_gpu(selected)
    check_exclusive()
    torch.set_num_threads(1);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;base.verify_sources()
    manifest=json.loads((root/'portable.json').read_text());rows=[]
    for case in manifest['cases']:
        check_exclusive()
        model=load(root,case);data=arrays(root,case);device='cuda'
        x=torch.from_numpy(data['field_x'][:512]).to(device);t=torch.from_numpy(data['field_t'][:512]).to(device)
        with torch.no_grad():pred=model(x,t).cpu().numpy()
        assert base.digest(root/case['cpu_field'])==case['cpu_field_sha256']
        cpu=np.load(root/case['cpu_field'])[:512]
        assert np.allclose(pred,cpu,atol=2e-5,rtol=2e-5),(case['id'],'field portability',float(np.max(np.abs(pred-cpu))))
        # The model is batch-separable; bind its portable trajectory to real CPU samples.
        for f,h in zip(case['cpu_samples'],case['cpu_sample_sha256']):assert base.digest(root/f)==h
        src=torch.from_numpy(data['source_0'][:64]).to(device)
        value=base.train.generated_samples(model,src,128).cpu().numpy();expected=np.load(root/case['cpu_samples'][0])[:64]
        assert np.max(np.abs(value-expected))<1e-3,(case['id'],'trajectory portability',float(np.max(np.abs(value-expected))))
        portability=dict(id=case['id'],field_max_error=float(np.max(np.abs(pred-cpu))),sample64_max_error=float(np.max(np.abs(value-expected))))
        if a.witness:
            rows.append(portability);base.dump(out/'progress.json',dict(completed=len(rows),total=len(manifest['cases'])));continue
        source=torch.from_numpy(np.concatenate([data[f'source_{i}'] for i in range(8)])).to(device)
        target=torch.from_numpy(np.concatenate([data[f'target_{i}'] for i in range(8)])).to(device).double()
        directions=torch.from_numpy(data['directions']).to(device).double()
        cpu_samples=np.concatenate([np.load(root/f) for f in case['cpu_samples']])
        configs=[(4,128),(4,256),(4,512)] if case['family'] in STATE_FAMILIES else [(0,128),(0,256),(0,512)]
        if case['family'] in STATE_FAMILIES:configs += [(8,128),(16,128)]
        for inner,outer in configs:
            if inner:
                for _,layer in model.ode_layers():layer.substeps=inner
            torch.cuda.synchronize();started=time.monotonic()
            generated=base.train.generated_samples(model,source,outer);assert torch.isfinite(generated).all()
            score=base.train.sliced_w2(generated.double(),target,directions)
            bank=[base.train.sliced_w2(generated[i*4096:(i+1)*4096].double(),target[i*4096:(i+1)*4096],directions) for i in range(8)]
            torch.cuda.synchronize();seconds=time.monotonic()-started
            filename=case['id']+f'__inner{inner}_heun{outer}.npy';value=generated.cpu().numpy()
            np.save(out/filename,value,allow_pickle=False)
            row=dict(id=case['id'],dataset=case['dataset'],family=case['family'],inner_substeps=inner,heun_steps=outer,nfe=2*outer,
                     pooled_sw2=score,bank_sw2=bank,seconds=seconds,samples=filename,samples_sha256=base.digest(out/filename),
                     checkpoint_sha256=case['checkpoint_sha256'],assets_sha256=case['assets_sha256'],portability=portability)
            if outer==128 and inner in [0,4]:
                row['full_cpu_max_error']=float(np.max(np.abs(value-cpu_samples)))
                cpu_score=base.train.sliced_w2(torch.from_numpy(cpu_samples).to(device).double(),target,directions)
                row['full_cpu_pooled_score_difference']=abs(score-cpu_score)
                assert row['full_cpu_pooled_score_difference']<1e-4,(case['id'],row)
            rows.append(row);base.dump(out/'rows.json',rows);base.dump(out/'progress.json',dict(completed=len(rows),last=row))
            q=subprocess.run(['nvidia-smi','-i',selected,'--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],capture_output=True,text=True,check=True)
            with (out/'gpu-history.jsonl').open('a') as f:f.write(json.dumps(dict(unix_time=time.time(),case=case['id'],inner=inner,outer=outer,resource=q.stdout.strip()))+'\n')
            check_exclusive()
            print(json.dumps(dict(id=case['id'],inner=inner,outer=outer,sw2=score,seconds=seconds)),flush=True)
        del model;torch.cuda.empty_cache()
    base.dump(out/'COMPLETE.json',dict(passed=True,witness=a.witness,cases=len(manifest['cases']),rows=rows,
              torch=torch.__version__,device=torch.cuda.get_device_name(0),source_sha256=base.digest(__file__),portable_manifest_sha256=base.digest(root/'portable.json')))
    print(json.dumps(dict(passed=True,witness=a.witness,output=str(out))))

if __name__=='__main__':main()
