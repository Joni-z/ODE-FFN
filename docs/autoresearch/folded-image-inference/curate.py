"""Independently check frozen tensors, sample errors and the timing ledger."""
import ast,csv,hashlib,json,shutil,statistics,subprocess
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/Users/mr.z/ode-workspace-prep/autoresearch/oct28-folded-image-inference')
DOC=Path(__file__).resolve().parent
REPO=DOC.parents[2]
BASE=ROOT.parent
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
assert load(ROOT/'status.json')['phase']=='COMPLETE'
execution=load(ROOT/'execution.json');assert execution['source_commit'].startswith('51348c7')
for name in ('benchmark.py','deployment.py','protocol.md'):
    assert sha(ROOT/'source'/name)==sha(DOC/name),name
parents={'native':'oct28-cycle17-v2','resolution':'oct28-cycle18','clocks':'oct28-cycle19-v2',
         'ordinary':'oct28-cycle20','maps':'oct28-cycle21-v2'}
for key,name in parents.items():
    folder=BASE/name;assert sha(folder/'sealed-runtime.json')==execution['parent_seals'][key]
    for file,digest in load(folder/'sealed-runtime.json').items():assert sha(folder/file)==digest
bundle=ROOT/'export';manifest=load(bundle/'manifest.json')
for name,digest in manifest['files'].items():assert sha(bundle/name)==digest
native=torch.load(BASE/parents['native']/'output/runs/ode_k2/step_0005000.pt',weights_only=False,map_location='cpu')
export=torch.load(bundle/'weights.pt',weights_only=True,map_location='cpu')
backbone=0
for key,value in native['model'].items():
    if key.startswith('blocks.') and '.mlp.' in key:continue
    assert torch.equal(value,export['model'][key]),key
    backbone+=1
assert native['runtime_buffers'].keys()==export['runtime_buffers'].keys()
assert all(torch.equal(v,export['runtime_buffers'][k]) for k,v in native['runtime_buffers'].items())
assert not any('.t_func.' in k or '.proj.' in k and '.mlp.' in k for k in export['model'])
with np.load(BASE/parents['maps']/'operators/ode_k2-0005000.npz') as maps:
    for j in range(3):
        for num,name in ((1,'ode1'),(2,'ode2')):
            assert np.array_equal(export['model'][f'blocks.{j}.mlp.fc{num}.weight'].numpy(),maps[f'blocks_{j}_{name}_T32'])
correctness=load(ROOT/'correctness.json');assert correctness['portable_fresh_process'] and correctness['historical_paths_forbidden']
assert len(correctness['rows'])==12
for row in correctness['rows']:
    if row['device']=='cpu':assert row['bitwise'] and row['max']==row['RMS']==0
    else:assert row['max']<=5e-4 and row['RMS']<=5e-5
generation=load(ROOT/'generation-timings.json');assert len(generation)==6
for row in generation:
    arm,device=row['arm'],row['device'];samples=np.load(ROOT/f'samples-{device}-{arm}.npy')
    assert samples.shape==(64,3,32,32) and np.isfinite(samples).all()
    if device=='cpu':
        target=np.load(BASE/parents['resolution']/'output/runs/ode_k2/generated-128.npy')[:64] if arm=='native' else np.load(BASE/parents['maps']/'output/runs/ode_k2_folded/generated.npy')[:64]
    else:target=np.load(ROOT/f'samples-cpu-{arm}.npy')
    diff=samples.astype(np.float64)-target.astype(np.float64)
    observed=dict(max=float(np.max(np.abs(diff))),RMS=float(np.sqrt(np.mean(diff*diff))),bitwise=samples.tobytes()==target.tobytes())
    assert observed==row['error']
    assert row['seconds']>0
    if device=='cpu':assert observed['bitwise']
    else:assert observed['max']<=.05 and observed['RMS']<=1e-3
summary=[]
for filename,mode,batches in (('resident-timings.json','resident',(1,16,64)),('transfer-timings.json','transfer',(64,))):
    rows=load(ROOT/filename);assert len(rows)==2*6*len(batches)*7
    for device in ('cpu','mps'):
        for batch in batches:
            for arm in ('native','literal','cached','compact','swiglu129','silu193'):
                selected=[r for r in rows if r['device']==device and r['batch']==batch and r['arm']==arm]
                assert sorted(r['trial'] for r in selected)==list(range(7))
                values=[r['seconds']*1000 for r in selected];assert min(values)>0
                summary.append(dict(mode=mode,device=device,batch=batch,arm=arm,
                    median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values),trials=7))
storage=load(ROOT/'storage.json')
for arm,state in (('native',native),('compact',export)):
    params={k:v for k,v in state['model'].items() if k not in state['runtime_buffers']}
    count=sum(v.numel() for v in params.values())
    size=sum(v.numel()*v.element_size() for v in params.values())+sum(v.numel()*v.element_size() for v in state['runtime_buffers'].values())
    assert count==storage[arm]['parameters'] and size==storage[arm]['stored_bytes']
work=load(ROOT/'dense-work.json')
assert work['compact']['total_FFN_dense_MACs_per_image']==3*64*2*64*64
assert work['literal']['total_FFN_dense_MACs_per_image']==2*work['compact']['total_FFN_dense_MACs_per_image']
protected=['src','synthetic','configs','scripts','reports/report1']
assert not subprocess.check_output(['git','-C',str(REPO),'diff','--name-only','98730e0','--',*protected],text=True)
for p in DOC.glob('*.py'):ast.parse(p.read_text())
out=DOC/'results';out.mkdir(exist_ok=False)
for name in ('execution.json','correctness.json','resident-timings.json','transfer-timings.json',
             'generation-timings.json','storage.json','dense-work.json','status.json'):
    shutil.copyfile(ROOT/name,out/name)
with (out/'timing-summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
rss=[json.loads(line) for line in (ROOT/'resources.jsonl').read_text().splitlines()]
write(out/'artifact-verification.json',dict(status='PASS',source_commit=execution['source_commit'],
    inherited_backbone_tensors=backbone,compact_matrices=6,unchanged_runtime_buffers=True,
    prediction_checks=12,generation_checks=6,resident_trials=252,transfer_trials=84,
    portable_fresh_process=True,historical_paths_forbidden=True,all_parent_hashes_preserved=True,
    peak_observed_process_RSS=max(row['RSS'] for row in rss),
    CPU_percent_telemetry='Unavailable: the original monitor re-created Process each sample and its unprimed cpu_percent values are not usable.',
    per_arm_peak_GPU_memory_measured=False,unchanged_scopes=protected,
    new_training=False,new_quality_estimate=False,new_ODE_method_demonstrated=False))
print(json.dumps(load(out/'artifact-verification.json')))
