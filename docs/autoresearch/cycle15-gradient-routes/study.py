"""Native replay and frozen next-batch gradient routes; no fitted optimizer steps."""
from __future__ import annotations
import argparse,copy,csv,hashlib,importlib.util,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import psutil
import torch

FAMILIES=('ode_k2','swiglu192','frame_k2','global_k2','frame_k1','swiglu257',
          'jet_c','affine_c','free_c','direct_b','swiglu258')
STEPS=(0,1000,2500,5000)

def load(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def table(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def imp(name,p):
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
        sys.modules[name]=m;spec.loader.exec_module(m)
    m=sys.modules[name];assert Path(m.__file__).resolve()==Path(p).resolve();return m

def setup(root):
    s=load(root/'settings.json');parent=Path(s['parent'])
    jet=imp('cycle15_parent_jet',parent/'source/jet_study.py')
    _,geo,native,core,_=jet.setup(parent)
    frame=imp('cycle15_parent_frame',Path(s['cycle11'])/'source/frame_operator.py')
    return s,jet,geo,native,core,frame

def build(root,law,family):
    s,jet,geo,native,core,frame=setup(root);p=Path(s['parent'])
    if family in ('jet_c','affine_c','free_c','direct_b','swiglu258'):
        return jet.build(p,law,family+'_clean'),p
    raw=native.old_build(128,'ode_k2' if family!='swiglu192' else family,'direct').raw
    if family in ('frame_k2','global_k2','frame_k1'):
        for block in raw.blocks:
            for name in ('ode1','ode2'):
                setattr(block.ffn,name,frame.ConditionalFrameLayer(getattr(block.ffn,name),
                    'global' if family=='global_k2' else 'state',1 if family=='frame_k1' else 2))
        source=Path(s['cycle11'])
    elif family=='swiglu257':
        from synthetic.models import VelocityNet
        from synthetic.recurrent_depth import copy_surrounding
        torch.manual_seed(0);ref=raw;raw=VelocityNet(128,'swiglu',width=64,depth=3,ffn_hidden=257)
        copy_surrounding(raw,ref);source=Path(s['cycle11'])
    else:source=Path(s['reference'])
    return core.CappedHead(raw,'clean'),source

def verify(root):
    for p,h in load(root/'bindings.json').items():assert sha(p)==h,p
    s,jet,*_=setup(root);jet.verify_inputs(Path(s['parent']))
    parent=Path(s['parent'])
    for p,h in load(parent/'sealed-runtime.json').items():assert sha(parent/p)==h,p

def prepare(root,parent):
    root.mkdir(exist_ok=False);src=root/'source';src.mkdir()
    here=Path(__file__).resolve().parent;repo=here.parents[2];ps=load(parent/'settings.json')
    for name in ('study.py','gradient_core.py','verify_fixtures.py','audit.py','protocol.md','theory.md','research-question-card.md'):
        shutil.copyfile(here/name,src/name)
    s=dict(parent=str(parent),cycle11=ps['parent_cycle11'],reference=ps['geometry_reference'],
           laws=ps['laws'],registration_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
           families=FAMILIES,steps=STEPS,phase_ceiling_seconds=600,threads=1)
    write(root/'settings.json',s);shutil.copyfile(parent/'basis.npy',root/'basis.npy')
    paths=[p for p in src.iterdir() if p.is_file()]+[root/'settings.json',root/'basis.npy',parent/'sealed-runtime.json']
    write(root/'bindings.json',{str(p):sha(p) for p in paths});verify(root)
    print(json.dumps(dict(status='PREPARED',root=str(root),bindings=len(paths))),flush=True)

def replay(model,field):
    with torch.no_grad():return np.concatenate([model(torch.from_numpy(field['x'][i:i+1024]),
        torch.from_numpy(field['t'][i:i+1024])).numpy() for i in range(0,len(field['x']),1024)])

def restored(root,law,family,step):
    s,_,_,native,_,_=setup(root);model,source=build(root,law,family)
    folder=source/'output/runs'/law/(family+'_clean');path=folder/f'step_{step:07d}.pt'
    state=torch.load(path,weights_only=False);native.restore_model(model,state);model.eval()
    return model,state,folder,path

def execute(root):
    from gradient_core import projector,routes,layout,summarize,clip
    torch.set_num_threads(1);verify(root)
    out=root/'output';out.mkdir(exist_ok=False);(out/'probes').mkdir()
    s,jet,geo,native,core,_=setup(root);q=np.load(root/'basis.npy');started=time.monotonic()
    records=[];checkpoints=[];proc=psutil.Process();peak=0
    write(out/'status.json',dict(state='RUNNING',pid=os.getpid(),checkpoints=0))
    try:
        for entry in s['laws']:
            law=entry['law'];bank=geo.training_bank(entry,native)
            with np.load(Path(entry['banks'])/'field.npz') as f:field={k:f[k] for k in f.files}
            for family in FAMILIES:
                for step in STEPS:
                    assert time.monotonic()-started<s['phase_ceiling_seconds']
                    model,state,folder,path=restored(root,law,family,step)
                    prediction=replay(model,field)
                    with np.load(folder/f'field_{step:07d}.npz') as f:np.testing.assert_array_equal(prediction,f['prediction'])
                    rng=torch.Generator();rng.set_state(state['data_rng'])
                    x,t,target,source=native.draw(bank,rng);label=core.regression_target(target,source,t)
                    params=list(model.parameters());actual=torch.autograd.grad((model(x,t)-label).square().mean(),params,allow_unused=True)
                    native_g=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for p,g in zip(params,actual)])
                    model=model.double();rows=layout(model);forced,rank=projector(model,q)
                    full,parts,energies=routes(model,x.double(),t.double(),label.double(),torch.from_numpy(q),torch.from_numpy(forced))
                    np.testing.assert_allclose(full.numpy(),parts.sum(0).numpy(),atol=1e-10,rtol=1e-8)
                    summaries,df,dr=summarize(full,parts,rows,state['optimizer'])
                    identity=dict(law=law,family=family,step=step)
                    for row in summaries:
                        if row['group']=='hidden':assert row['forced_norm']<1e-10*max(1.,row['full_norm'])
                        records.append(dict(**identity,**row))
                    key=f'{law}_{family}_{step:07d}'
                    np.savez_compressed(out/'probes'/f'{key}.npz',x=x.numpy(),t=t.numpy(),label=label.numpy(),
                        forced=forced,full_gradient=full.numpy(),parts=parts.numpy(),
                        native_gradient=native_g.numpy(),update_full=df.numpy(),update_removed=dr.numpy())
                    write(out/'probes'/f'{key}.json',dict(layout=rows,checkpoint=str(path),sha256=sha(path),
                        next_rng_sha256=hashlib.sha256(rng.get_state().numpy().tobytes()).hexdigest()))
                    checkpoints.append(dict(**identity,forced_dimension=rank,
                        tangent_loss=energies[0],forced_loss=energies[1],remaining_normal_loss=energies[2],
                        native_gradient_norm=float(native_g.norm()),native_clip=clip(native_g),
                        promoted_gradient_norm=float(full.norm()),checkpoint_sha256=sha(path),
                        field_rows=len(prediction)))
                    rss=proc.memory_info().rss;peak=max(peak,rss)
                    monitor=dict(pid=os.getpid(),completed=len(checkpoints),total=88,elapsed=time.monotonic()-started,
                        rss_bytes=rss,cpu_seconds=sum(proc.cpu_times()[:2]),**identity)
                    with (out/'monitor.jsonl').open('a') as f:f.write(json.dumps(monitor)+'\n')
                    write(out/'status.json',dict(state='RUNNING',**monitor));print(json.dumps(monitor),flush=True)
        table(out/'gradient-groups.csv',records);table(out/'checkpoints.csv',checkpoints);verify(root)
        write(out/'receipt.json',dict(status='PASS',pid=os.getpid(),seconds=time.monotonic()-started,
            checkpoints=len(checkpoints),group_rows=len(records),native_field_rows=sum(r['field_rows'] for r in checkpoints),
            peak_rss_GiB=peak/2**30,training_updates=0,threads=1))
        write(out/'status.json',dict(state='COMPLETE',**load(out/'receipt.json')))
    except BaseException as e:
        write(out/'status.json',dict(state='FAILED',error=repr(e)));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('prepare','run'));ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--parent',type=Path);args=ap.parse_args()
    if args.action=='prepare':prepare(args.root,args.parent)
    else:execute(args.root)
