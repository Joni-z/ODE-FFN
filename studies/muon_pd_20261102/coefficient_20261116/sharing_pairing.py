"""Verify formal-run initialization, recipe and training RNG pairing."""
import argparse,json
from pathlib import Path
from sharing_trial import base,torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root);out=Path(a.output);assert not out.exists();rows=[];anchors=json.loads((root/'sharing-anchors.json').read_text())['cases']
    for dataset in ['gmm_ring8_2d','rotated_product_bimodal_16d']:
        pair={}
        for row in anchors:
            folder=Path(row['folder']);cfg=json.loads((folder/'config.json').read_text());c=cfg['cell']
            if c['dataset']==dataset and c['family'] in ['state_midpoint4','stage_operator_midpoint4']:pair[c['family']]=(folder,cfg)
        assert len(pair)==2;left,lc=pair['state_midpoint4'];right,rc=pair['stage_operator_midpoint4']
        ignored={'cell','grid_sha256','experiment_id','models'}
        assert {k:v for k,v in lc.items() if k not in ignored}=={k:v for k,v in rc.items() if k not in ignored}
        records=[]
        for step in [0,15000,20000]:
            lf=left/'checkpoints'/f'step_{step:07d}.pt';rf=right/'checkpoints'/f'step_{step:07d}.pt'
            l=torch.load(lf,map_location='cpu',weights_only=False);r=torch.load(rf,map_location='cpu',weights_only=False)
            assert l['step']==r['step']==step
            assert l['dataset_manifest_sha256']==r['dataset_manifest_sha256']
            assert torch.equal(l['data_rng'],r['data_rng'])
            if step==0:
                for name,tensor in l['model'].items():assert torch.equal(tensor,r['model'][name]),name
                new=set(r['model'])-set(l['model']);assert len(new)==6 and all(k.endswith('stage_delta') for k in new)
                assert all(torch.count_nonzero(r['model'][k])==0 for k in new)
            records.append(dict(step=step,data_rng_exact=True,left_sha256=base.digest(lf),right_sha256=base.digest(rf)))
        rows.append(dict(dataset=dataset,recipe_equal_except_architecture_identity=True,initial_common_tensors_exact=True,new_matrices_zero=True,checkpoints=records))
    base.dump(out,dict(passed=True,scope='formal initialization/recipe/RNG pairing; not equality of optimization trajectories',rows=rows,source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,datasets=len(rows),output=str(out))))

if __name__=='__main__':main()
