"""One frozen model: orbit field errors, paired group sampling and numerical checks."""
import argparse,json,os,subprocess,time
from pathlib import Path
import numpy as np
from common import HERE,base,torch,load,arrays,law_for,Projection,orbit,pullback,decompose,sample_stats,exclusive_gpu,ev

def dump(path,obj):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);p.add_argument('--output',required=True);p.add_argument('--readiness',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--gpu-uuid');a=p.parse_args()
    def exclusive():
        if a.device=='cuda':assert a.gpu_uuid;exclusive_gpu(a.gpu_uuid)
    exclusive();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);base.verify_sources()
    ready=json.loads(Path(a.readiness).read_text());assert ready['passed'] and ready['cases']==8 and ready['device']==a.device
    assert ready['portable_manifest_sha256']==base.digest(root/'portable.json')
    if a.device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;assert ready['device_name']==torch.cuda.get_device_name(0)
    case=next(c for c in json.loads((root/'portable.json').read_text())['cases'] if c['id']==a.case)
    qualification=next(r for r in ready['rows'] if r['id']==case['id']);assert qualification['checkpoint_sha256']==case['checkpoint_sha256']
    model=load(root,case,a.device).double();data=arrays(root,case);law=law_for(root,case);snapshot=ev.tensor_snapshot(model)
    tensor=lambda x:torch.from_numpy(np.ascontiguousarray(x)).to(a.device,dtype=torch.double)
    group=tensor(data['group']);projected=Projection(model,group).eval();directions=tensor(data['directions']);dim=law.dim
    resources=[];rows=[];pools=[];resolutions=[];paths=[];started=time.monotonic()
    def progress(phase,**kw):dump(out/'progress.json',dict(pid=os.getpid(),phase=phase,sample_rows_completed=len(rows),sample_rows_total=8,seconds=time.monotonic()-started,**kw))
    def resource(phase,**kw):
        exclusive()
        if a.device=='cuda':
            value=subprocess.check_output(['nvidia-smi','-i',a.gpu_uuid,'--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip()
            resources.append(dict(unix_time=time.time(),phase=phase,resource=value,**kw));dump(out/'resource-history.json',resources)
    def generate(mode,source_base,source_orbits,steps):
        if mode=='original':return base.train.generated_samples(model,source_orbits,steps)
        if qualification['shortcut_allowed']:return orbit(base.train.generated_samples(projected,source_base,steps),group)
        return base.train.generated_samples(projected,source_orbits,steps)
    progress('field_decomposition');exclusive()
    with torch.no_grad():
        x=tensor(data['field_x']);t=tensor(data['field_t']);oracle=tensor(data['field_oracle'])
        values=model(orbit(x,group),t.expand(8,*t.shape).reshape(-1,1)).reshape(8,len(x),dim)
        pulled=pullback(values,group);field=decompose(pulled,oracle,t)
        field['base_mse']=float((pulled[0]-oracle).square().mean());np.save(out/'field-pulled.npy',pulled.cpu().numpy(),allow_pickle=False)
    dump(out/'field.json',field);resource('field_decomposition')
    targets=tensor(np.concatenate([data[f'orbit_target_{b}'] for b in range(4)]))
    for mode in ['original','projected']:
        samples=[];coarse=None
        for bank in range(4):
            exclusive();progress('sampling',mode=mode,bank=bank)
            before=time.monotonic();value=generate(mode,tensor(data[f'source_{bank}']),tensor(data[f'orbit_source_{bank}']),128)
            assert torch.isfinite(value).all();target=tensor(data[f'orbit_target_{bank}']);score=base.train.sliced_w2(value,target,directions)
            array=value.cpu().numpy();file=f'{mode}-bank{bank}.npy';np.save(out/file,array,allow_pickle=False);samples.append(array)
            if bank==0:coarse=array.reshape(8,4096,dim)[:,:512].reshape(-1,dim).copy()
            rows.append(dict(mode=mode,bank=bank,stratified_sw2=score,base_samples=4096,samples=32768,seconds=time.monotonic()-before,
                generated_file=file,generated_sha256=base.digest(out/file),**sample_stats(array,law,case['dataset'])))
            dump(out/'sample-rows.json',rows);resource('sampling',mode=mode,bank=bank)
        pooled=np.concatenate(samples);poolscore=base.train.sliced_w2(tensor(pooled),targets,directions)
        scores=[r['stratified_sw2'] for r in rows if r['mode']==mode]
        pools.append(dict(mode=mode,pooled_stratified_sw2=poolscore,bank_mean=float(np.mean(scores)),bank_sd=float(np.std(scores,ddof=1)),
                          independent_base_points=16384,group_stratified_points=131072,**sample_stats(pooled,law,case['dataset'])))
        progress('outer_resolution',mode=mode);exclusive()
        subsource=tensor(data['orbit_source_0'].reshape(8,4096,dim)[:,:512].reshape(-1,dim));subtarget=tensor(data['orbit_target_0'].reshape(8,4096,dim)[:,:512].reshape(-1,dim))
        fine=generate(mode,tensor(data['source_0'][:512]),subsource,256).cpu().numpy()
        for steps,value in [(128,coarse),(256,fine)]:
            file=f'{mode}-subset-heun{steps}.npy';np.save(out/file,value,allow_pickle=False)
            resolutions.append(dict(mode=mode,heun_steps=steps,base_samples=512,stratified_sw2=base.train.sliced_w2(tensor(value),subtarget,directions),
                                    rms_endpoint_shift_to256=float(np.sqrt(np.mean(np.sum((value-fine)**2,axis=1)))),generated_file=file,generated_sha256=base.digest(out/file)))
        net=model if mode=='original' else projected;end,path=ev.sample(net,None,tensor(data['source_0'][:16]),128,record=True)
        assert path.shape==(129,16,dim) and np.isfinite(path).all();file=mode+'-paths.npy';np.save(out/file,path,allow_pickle=False)
        paths.append(dict(mode=mode,file=file,sha256=base.digest(out/file)));resource('resolution_and_paths',mode=mode)
    ev.same_tensors(model,snapshot);exclusive()
    dump(out/'COMPLETE.json',dict(passed=True,id=case['id'],dataset=case['dataset'],family=case['family'],dtype='float64',device=a.device,gpu_uuid=a.gpu_uuid,
        checkpoint_sha256=case['checkpoint_sha256'],assets_sha256=case['assets_sha256'],portable_manifest_sha256=base.digest(root/'portable.json'),readiness_sha256=base.digest(a.readiness),
        shortcut_allowed=qualification['shortcut_allowed'],field=field,field_pulled_sha256=base.digest(out/'field-pulled.npy'),sample_rows=rows,pooled=pools,resolution=resolutions,paths=paths,
        parameters_unchanged=True,source_sha256={f.name:base.digest(f) for f in [Path(__file__),HERE/'common.py']}))
    progress('complete');print(json.dumps(dict(passed=True,id=case['id'],output=str(out))))

if __name__=='__main__':main()
