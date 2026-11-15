"""Bind four fresh banks, exact group orbits and eight immutable CPU checkpoints."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,base,torch,ev,load,arrays,law_for,group_matrices,orbit_np,kernel_checks
from synthetic.data import fm_batch

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.source_root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);base.verify_sources()
    records=[];asset_rows=[];kernels=[]
    sources=json.loads((source/'sharing-anchors.json').read_text())['cases'];assert len(sources)==8
    for dataset in ['gmm_ring8_2d','rotated_product_bimodal_16d']:
        selected=next(r for r in sources if json.loads((Path(r['folder'])/'config.json').read_text())['cell']['dataset']==dataset)
        cfg=json.loads((Path(selected['folder'])/'config.json').read_text());cache=base.verify_data(a.data_root,cfg['cell'])
        law=base.train.load_distribution(json.loads((cache/'manifest.json').read_text())['datasets'][dataset]);group=group_matrices(dataset,law)
        kernels.append(kernel_checks(law,dataset))
        rng=np.random.default_rng(2026114200);directions=rng.standard_normal((256,law.dim));directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        values=ev.bank_arrays(law,directions,4096,4,2026112200);values['group']=group
        rng=np.random.default_rng(2026113200);x,t,y=fm_batch(law.sample(rng,4096),rng)
        values.update(field_x=x,field_t=t,field_y=y,field_oracle=law.velocity(x.astype(float),t.astype(float)))
        for bank in range(4):
            values[f'orbit_source_{bank}']=orbit_np(values[f'source_{bank}'],group)
            values[f'orbit_target_{bank}']=orbit_np(values[f'target_{bank}'],group)
        folder=out/'assets'/dataset;folder.mkdir(parents=True);file=folder/'banks.npz';np.savez_compressed(file,**values)
        asset_rows.append(dict(dataset=dataset,file=str(file.relative_to(out)),sha256=base.digest(file),stream=2026112200,banks=4,base_samples=4096,orbit_size=8,arrays={k:ev.array_hash(v) for k,v in values.items()}))
    for row in sources:
        folder=Path(row['folder']);cfg=json.loads((folder/'config.json').read_text());cell=cfg['cell'];assert (folder/'ADAPTER_COMPLETE').exists()
        cache=base.verify_data(a.data_root,cell);target=out/'cases'/cell['id'];target.mkdir(parents=True)
        def copy(path,name):
            f=target/name;shutil.copy2(path,f);return str(f.relative_to(out)),base.digest(f)
        config,ch=copy(folder/'config.json','config.json');checkpoint,ph=copy(folder/'checkpoints/step_0020000.pt','step_0020000.pt');manifest,mh=copy(cache/'manifest.json','dataset-manifest.json')
        asset=next(r for r in asset_rows if r['dataset']==cell['dataset'])
        case=dict(id=cell['id'],dataset=cell['dataset'],family=cell['family'],config=config,config_sha256=ch,checkpoint=checkpoint,checkpoint_sha256=ph,
                  dataset_manifest=manifest,dataset_manifest_sha256=mh,assets=asset['file'],assets_sha256=asset['sha256'])
        model=load(out,case,'cpu').double();data=arrays(out,case);snapshot=ev.tensor_snapshot(model)
        with torch.no_grad():prediction=model(torch.from_numpy(data['field_x'][:512]).double(),torch.from_numpy(data['field_t'][:512]).double()).numpy()
        generated=base.train.generated_samples(model,torch.from_numpy(data['source_0'][:64]).double(),128).numpy()
        for key,value in [('cpu_field64',prediction),('cpu_sample64',generated)]:
            f=target/(key+'.npy');np.save(f,value,allow_pickle=False);case[key]=str(f.relative_to(out));case[key+'_sha256']=base.digest(f)
        ev.same_tensors(model,snapshot);records.append(case);base.dump(out/'progress.json',dict(cpu_references_completed=len(records),total=8))
    base.dump(out/'portable.json',dict(passed=True,cases=records,asset_bindings=asset_rows,kernel_checks=kernels,
        protocol=dict(stream=2026112200,training_seed=0,banks=4,base_samples=4096,group_size=8,field_points=4096,directions=256,dtype='float64',heun_steps=128,hidden_substeps=4),
        source_sha256={f.name:base.digest(f) for f in [Path(__file__),HERE/'common.py']}))
    print(json.dumps(dict(passed=True,cases=8,output=str(out))))

if __name__=='__main__':main()
