"""Fixed constructors, replay streams and checkpoint operations for this study."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

BASE=Path('/Users/mr.z/ode-workspace-prep/architecture-search')
OLD=BASE/'conservative_readout_20261026/output'
DATASET='rotated_product_bimodal_16d'
BANKS=OLD/'banks'/DATASET
STUDENTS=('ode_k2','swiglu128','swiglu192');TEACHERS=('ode_k2','swiglu128');TIMES=(.1,.3,.5,.7,.9)
SNAPSHOTS=(0,1000,2500,5000)
def load(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def append(p,v):
    with Path(p).open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*2**20):h.update(b)
    return h.hexdigest()
def setup():
    provenance=load(OLD/'provenance.json');source=Path(provenance['source_root'])
    for name,digest in provenance['source_sha256'].items():assert sha(source/name)==digest,name
    if str(source) not in sys.path:sys.path.insert(0,str(source))
    return provenance
def student(family):
    setup()
    from synthetic.conservative_readout import build
    return build(16,family,'direct')
def teacher(family):
    m=student(family);path=OLD/'runs'/DATASET/(family+'_direct')/'step_0005000.pt'
    from synthetic.train import restore_model
    restore_model(m,torch.load(path,map_location='cpu',weights_only=False));m.eval()
    for p in m.parameters():p.requires_grad_(False)
    return m,path
def signature(m):
    h=hashlib.sha256()
    for name,p in list(m.named_parameters())+list(m.named_buffers()):
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def clone_state(m):return {k:v.detach().clone() for k,v in m.state_dict().items()}
def assert_equal(a,b):
    if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and torch.equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:assert_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_equal(x,y)
    else:assert a==b,(a,b)
def verify_cold(m,family):
    cp=OLD/'runs'/DATASET/(family+'_direct')/'step_0000000.pt';state=torch.load(cp,map_location='cpu',weights_only=False)
    assert_equal(m.state_dict(),state['model']);assert_equal(dict(m.named_buffers()),state['runtime_buffers'])
    return cp
def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=0.)
def update(m,opt,x,t,y):
    opt.zero_grad(set_to_none=True);loss=(m(x,t)-y).square().mean();assert torch.isfinite(loss)
    loss.backward();gradient=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gradient)
    opt.step();return float(loss.detach()),float(gradient)
def targets():
    setup()
    from scripts import run_recurrent_depth as base
    config=load(OLD/'study.json');root,spec,_=base.load_data(config,DATASET)
    return torch.from_numpy(base.load_targets(root/DATASET/'train.npz',spec['dim'])),root
def draw(bank,rng):
    target=bank[torch.randint(len(bank),(512,),generator=rng)];source=torch.randn(target.shape,generator=rng);t=torch.rand((512,1),generator=rng)
    return (1-t)*source+t*target,t,target,source
def replay_hash(previous,*values):
    h=hashlib.sha256(bytes.fromhex(previous))
    for value in values:
        array=value.detach().cpu().contiguous().numpy() if isinstance(value,torch.Tensor) else np.ascontiguousarray(value)
        h.update(str((array.shape,str(array.dtype))).encode('ascii'));h.update(array.tobytes())
    return h.hexdigest()
@torch.no_grad()
def predict(m,x,t):
    values=[];m.eval()
    for start in range(0,len(x),1024):values.append(m(torch.from_numpy(np.array(x[start:start+1024],copy=True)),torch.from_numpy(np.array(t[start:start+1024],copy=True))).numpy())
    return np.concatenate(values)
def batch(cache,which,step):
    return tuple(torch.from_numpy(np.array(cache[k][step],copy=True)) for k in ['x','t','label-'+which])
def open_cache(path):
    return {k:np.load(path/(k+'.npy'),mmap_mode='r') for k in ['x','t']+['label-'+t for t in TEACHERS]}
def snapshot(m,opt,step,folder,cache_metadata,teacher_name,actual_replay=None):
    path=folder/f'step_{step:07d}.pt';assert not path.exists()
    stream=cache_metadata['snapshots'][str(step)]
    if actual_replay is not None:
        assert actual_replay==dict(input=stream['input_replay_digest'],label=stream['label_replay_digests'][teacher_name])
    state=dict(step=step,model=m.state_dict(),optimizer=opt.state_dict(),runtime_buffers={k:v.detach().clone() for k,v in m.named_buffers()},
        torch_rng=torch.get_rng_state(),cache_position=step,input_stream_digest=stream['input_digest'],teacher_label_digest=stream['label_digests'][teacher_name],
        input_replay_mode='shared sequential immutable cache',actual_replay_digests=actual_replay)
    torch.save(state,path)
    from scripts.run_prediction_bridge import spectrum
    write(folder/f'spectrum_{step:07d}.json',spectrum(m))
def restore(m,opt,state):
    from synthetic.train import restore_model
    restore_model(m,state);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng'])
def external_bindings():
    provenance=setup();paths=[OLD/'study.json',OLD/'provenance.json']
    paths.extend(Path(provenance['source_root'])/name for name in provenance['source_sha256'])
    config=load(OLD/'study.json');data=Path(config['data_roots'][DATASET]);paths.append(data/'manifest.json')
    paths.extend(data/DATASET/name for name in ['train.npz','validation.npz','test.npz','probe.npz'])
    for family in STUDENTS:
        folder=OLD/'runs'/DATASET/(family+'_direct');paths.append(folder/'step_0000000.pt')
        if family in TEACHERS:
            paths.extend([folder/'step_0005000.pt',folder/'result.json',folder/'diagnostics/field.npz'])
            paths.extend(folder/'diagnostics'/f'panel_{t:.1f}.npz' for t in TIMES)
    paths.extend(BANKS/name for name in ['field.npz','probe.npz'])
    paths.extend(BANKS/f'panel_{t:.1f}.npz' for t in TIMES)
    return {str(p.resolve()):sha(p) for p in paths}
