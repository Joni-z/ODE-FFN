"""Seeded group/oracle checks, real CPU references and shortcut qualification."""
import argparse,importlib.util,json
from pathlib import Path
import numpy as np
from common import HERE,base,torch,load,arrays,law_for,Projection,orbit,kernel_checks,exclusive_gpu,ev

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--gpu-uuid');a=p.parse_args()
    if a.device=='cuda':assert a.gpu_uuid;exclusive_gpu(a.gpu_uuid)
    root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);base.verify_sources()
    if a.device=='cuda':torch.backends.cuda.matmul.allow_tf32=False;assert torch.cuda.is_available()
    manifest=json.loads((root/'portable.json').read_text());assert manifest['passed'] and len(manifest['cases'])==8
    kernels=[];rows=[];checked=set()
    for case in manifest['cases']:
        if a.device=='cuda':exclusive_gpu(a.gpu_uuid)
        law=law_for(root,case)
        if case['dataset'] not in checked:kernels.append(kernel_checks(law,case['dataset'],a.device));checked.add(case['dataset'])
        model=load(root,case,a.device).double();data=arrays(root,case);snapshot=ev.tensor_snapshot(model)
        tensor=lambda x:torch.from_numpy(x).to(a.device,dtype=torch.double)
        with torch.no_grad():prediction=model(tensor(data['field_x'][:512]),tensor(data['field_t'][:512])).cpu().numpy()
        for key in ['cpu_field64','cpu_sample64']:assert base.digest(root/case[key])==case[key+'_sha256']
        field_error=float(np.max(np.abs(prediction-np.load(root/case['cpu_field64']))));assert field_error<2e-9
        source=tensor(data['source_0'][:64]);native=base.train.generated_samples(model,source,128).cpu().numpy()
        native_error=float(np.max(np.abs(native-np.load(root/case['cpu_sample64']))));assert native_error<1e-7
        group=tensor(data['group']);projected=Projection(model,group).eval();dim=law.dim;n=16
        base_source=source[:n];direct_source=tensor(data['orbit_source_0'].reshape(8,4096,dim)[:,:n].reshape(-1,dim))
        direct=base.train.generated_samples(projected,direct_source,128)
        shortcut=orbit(base.train.generated_samples(projected,base_source,128),group)
        error=float((direct-shortcut).abs().max());allowed=error<1e-9
        # A failed shortcut uses the full projected field; never relax its tolerance.
        rows.append(dict(id=case['id'],checkpoint_sha256=case['checkpoint_sha256'],assets_sha256=case['assets_sha256'],
                         field_max_error=field_error,native64_sample_max_error=native_error,shortcut_max_error=error,shortcut_allowed=allowed))
        ev.same_tensors(model,snapshot);base.dump(out/'progress.json',dict(completed=len(rows),total=8,last=rows[-1]))
        if a.device=='cuda':exclusive_gpu(a.gpu_uuid)
    stamp=dict(passed=True,cases=8,device=a.device,gpu_uuid=a.gpu_uuid,device_name=torch.cuda.get_device_name(0) if a.device=='cuda' else 'CPU',
        kernels=kernels,rows=rows,portable_manifest_sha256=base.digest(root/'portable.json'),source_sha256={f.name:base.digest(f) for f in [Path(__file__),HERE/'common.py']})
    provisional=out/'entry-readiness.json';base.dump(provisional,stamp)
    spec=importlib.util.spec_from_file_location('sharing_symmetry_case',HERE/'evaluate.py')
    evaluate=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluate)
    entries=[]
    for dataset,family in [('gmm_ring8_2d','state_midpoint4'),('rotated_product_bimodal_16d','stage_operator_midpoint4')]:
        case=next(c for c in manifest['cases'] if c['dataset']==dataset and c['family']==family);target=out/('entry-'+dataset)
        cmd=['--root',str(root),'--case',case['id'],'--output',str(target),'--readiness',str(provisional),'--device',a.device,'--witness']
        if a.gpu_uuid:cmd+=['--gpu-uuid',a.gpu_uuid]
        evaluate.main(cmd);done=json.loads((target/'COMPLETE.json').read_text());assert done['passed'] and done['witness'] and done['parameters_unchanged']
        entries.append(dict(id=case['id'],completion_sha256=base.digest(target/'COMPLETE.json')))
    stamp['entry_witnesses']=entries;base.dump(out/'COMPLETE.json',stamp)
    print(json.dumps(dict(passed=True,cases=8,shortcuts=sum(r['shortcut_allowed'] for r in rows),output=str(out))))

if __name__=='__main__':main()
