"""Source-bound complete-predictor responses in observed image coordinates."""
from __future__ import annotations
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','PYTHONDONTWRITEBYTECODE'):os.environ[k]='1'
sys.dont_write_bytecode=True
import argparse,hashlib,importlib.util,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
import geometry_math as gm
if __name__=='__main__':sys.modules['geometry_study']=sys.modules[__name__]

BASE=Path('/Users/mr.z/ode-workspace-prep/autoresearch');PARENT=BASE/'oct29-image-state-risk-v1'
ARMS=('ode_k1','fixed06','matched_silu64','swiglu86','silu131')
DOMAINS=('paired_validation','paired_training','generated_K1','generated_Swi86');INDICES=(13,38,64,90,115);IMAGES=(0,1)
PYTHON='/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python'
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def setup():
    sys.path.insert(0,str(PARENT/'source'))
    if 'state_risk_study' not in sys.modules:
        spec=importlib.util.spec_from_file_location('state_risk_study',PARENT/'source/study.py');m=importlib.util.module_from_spec(spec);sys.modules['state_risk_study']=m;spec.loader.exec_module(m)
    g=sys.modules['state_risk_study'];assert Path(g.__file__).resolve()==PARENT/'source/study.py'
    g.setup();torch.set_num_threads(1);return g
def verify(root):
    g=setup();g.verify(PARENT)
    for name,digest in load(PARENT/'sealed-runtime.json').items():assert sha(PARENT/name)==digest,name
    for name,digest in load(root/'bindings.json').items():assert sha(name)==digest,name
def frames(root):
    with np.load(root/'frames.npz') as f:return {key:gm.PatchFrame(torch.from_numpy(f[key])) for key in f.files}
def prepare(root):
    g=setup();repo=Path(__file__).resolve().parents[3];assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    assert not root.exists();root.mkdir();shutil.copytree(Path(__file__).parent,root/'source');(root/'doc-validation').mkdir()
    values={};nulls={};info=[]
    for arm in ARMS:
        native=g.model_for(arm);u,null,b,s=gm.encoder_space(native);values[arm]=u.numpy();nulls[arm]=null
        info.append(dict(arm=arm,patch_rank=u.shape[1],pixel_dimension=48,singular_values=s.tolist(),checkpoint_sha256=sha(g.folder_for(arm)/'step_0020000.pt')))
    common=gm.intersection(nulls['ode_k1'],nulls['swiglu86']);values['common']=common.numpy();np.savez(root/'frames.npz',**values)
    write(root/'settings.json',dict(source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),models=info,
        common_patch_rank=common.shape[1],expected_rows=280 if common.shape[1] else 200,env_spec='475428fb',sketch_size=24,new_training=False,new_FID=False))
    paths=[*Path(root/'source').rglob('*'),root/'settings.json',root/'frames.npz',PARENT/'sealed-runtime.json',PARENT/'states.npz',PARENT/'bindings.json']
    write(root/'bindings.json',{str(p.resolve()):sha(p) for p in sorted(set(paths)) if p.is_file() and p.suffix!='.pyc'});verify(root)
    print('IMAGE_RESPONSE_PREPARED',flush=True)
def validate_frame(model,frame,image,t):
    u,_,b,_=gm.encoder_space(model);saved=frame.basis
    torch.testing.assert_close(saved@saved.T,u@u.T,rtol=0,atol=1e-10)
    patches=torch.nn.functional.unfold(image,kernel_size=4,stride=4).transpose(1,2)
    expected=patches@b.T+model.x_embedder.proj2.bias
    with torch.no_grad():actual=model.x_embedder(image)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
    gen=torch.Generator().manual_seed(0);random=torch.randn(image.shape,generator=gen,dtype=image.dtype)
    null=random-frame.lift(frame.encode(random));null=.01*null/torch.linalg.vector_norm(null)
    with torch.no_grad():
        a=model(image,torch.full((1,),t,dtype=image.dtype),torch.zeros(1,dtype=torch.long))
        b=model(image+null,torch.full((1,),t,dtype=image.dtype),torch.zeros(1,dtype=torch.long))
    torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    return float((a-b).abs().max())
def preflight(root):
    g=setup();verify(root);math=gm.fixtures();stored=frames(root);rows=[];n,_,_=g.setup();validation=n.bank(g.NATIVE/'banks/validation.npz');validation={k:v[:64] for k,v in validation.items()}
    with np.load(PARENT/'states.npz') as states:
        for arm in ARMS:
            native=g.model_for(arm);before=g.fingerprint(native);literal=gm.literal_model(native,promote=False);promoted=gm.literal_model(native);maximum=0.;replays=0
            prediction=n.predict(native,validation)
            with np.load(g.folder_for(arm)/'validation_0020000.npz') as f:assert prediction.tobytes()==f['prediction'][:64].tobytes()
            for domain in DOMAINS:
                for j in INDICES:
                    key=f'{domain}_{j}';x=torch.from_numpy(states[key]);t=torch.full((64,),j/128);labels=torch.zeros(64,dtype=torch.long)
                    with torch.no_grad():a=native(x,t,labels);b=literal(x,t,labels);c=promoted(x[:2].double(),t[:2].double(),labels[:2])
                    assert a.numpy().tobytes()==np.load(PARENT/'output/predictions'/f'{arm}_{key}.npy').tobytes()
                    assert torch.equal(a,b),'float32 literal path differs'
                    torch.testing.assert_close(c,a[:2].double(),rtol=1e-4,atol=1e-5);maximum=max(maximum,float((c-a[:2]).abs().max()));replays+=1
            image=torch.from_numpy(states['paired_validation_64'][:1]).double();frame=stored[arm];null=validate_frame(promoted,frame,image,.5)
            _,action=gm.input_action(promoted,image,.5,frame);gen=torch.Generator().manual_seed(0);q=torch.randn(frame.dimension,generator=gen,dtype=torch.float64);q=q/torch.linalg.vector_norm(q)
            response=action(q);assert float(response.norm())>1e-8;rayleigh=float(q@response)
            finite=gm.finite_rayleigh(promoted,image,.5,frame,q,1e-4);np.testing.assert_allclose(finite,rayleigh,rtol=3e-5,atol=3e-7)
            assert g.fingerprint(native)==before
            rows.append(dict(arm=arm,native_prefix_images=64,state_batch_replays=replays,float32_literal_bitwise=True,promoted_forward_max_absolute_error=maximum,
                null_perturbation_max_output_error=null,nonzero_VJP_norm=float(response.norm()),finite_difference_absolute_error=abs(finite-rayleigh)))
    if stored['common'].rank:
        h=stored['common'].basis
        for arm in ('ode_k1','swiglu86'):assert float(((torch.eye(48,dtype=h.dtype)-stored[arm].basis@stored[arm].basis.T)@h).abs().max())<1e-10
    verify(root);write(root/'preflight.json',dict(status='PASS',mathematics=math,models=rows,common_patch_rank=stored['common'].rank,formal_measurement_started=False));print('IMAGE_RESPONSE_NATIVE_PREFLIGHT_PASS',flush=True)
def measure(root):
    g=setup();verify(root);out=root/'output';out.mkdir();(out/'groups').mkdir();tick=time.monotonic();space=frames(root)
    models={a:gm.literal_model(g.model_for(a)) for a in ARMS};before={a:g.fingerprint(m) for a,m in models.items()};rows=[];groups=[]
    with np.load(PARENT/'states.npz') as states:
        for domain in DOMAINS:
            for j in INDICES:
                for image_index in IMAGES:
                    image=torch.from_numpy(states[f'{domain}_{j}'][image_index:image_index+1]).double();t=j/128
                    specifications=[('own-'+a,(a,),a) for a in ARMS]
                    if space['common'].rank:specifications.append(('shared',('ode_k1','swiglu86'),'common'))
                    for label,arms,frame_key in specifications:
                        assert time.monotonic()-tick<1800
                        name=f'{domain}-t{j:03d}-i{image_index}-{label}';frame=space[frame_key]
                        actions=[gm.input_action(models[arm],image,t,frame)[1] for arm in arms]
                        basis,res,breakdowns=gm.sketch(actions,frame.dimension);data=dict(basis=basis.numpy())
                        for arm,row in zip(arms,res):
                            for key in ('compressed_transpose','eigenvalues','witness'):data[arm+'__'+key]=row[key].numpy()
                            record=dict(domain=domain,index=j,time=t,image=image_index,frame=frame_key,arm=arm,group=name,**{k:v for k,v in row.items() if not isinstance(v,torch.Tensor)})
                            rows.append(record)
                        np.savez_compressed(out/'groups'/f'{name}.npz',**data);groups.append(dict(name=name,domain=domain,index=j,image=image_index,arms=arms,frame=frame_key,breakdowns=breakdowns))
                        write(out/'progress.json',dict(state='RUNNING',groups=len(groups),rows=len(rows),expected_rows=load(root/'settings.json')['expected_rows'],seconds=time.monotonic()-tick))
    for arm,model in models.items():assert g.fingerprint(model)==before[arm]
    assert len(rows)==load(root/'settings.json')['expected_rows'];verify(root)
    write(out/'result.json',dict(status='COMPLETE',rows=rows,groups=groups,model_tensors_unchanged=True,seconds=time.monotonic()-tick));print('IMAGE_RESPONSE_MEASUREMENT_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','preflight','measure','audit','run'));p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    if a.action=='audit':
        from geometry_audit import audit;audit(root)
    elif a.action=='run':
        from geometry_controller import run;run(root)
    else:globals()[a.action](root)
