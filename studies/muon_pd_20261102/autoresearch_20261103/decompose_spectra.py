"""Read-only row/column magnitude/coherence decomposition of archived P tensors."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);p.add_argument('--dataset',required=True);p.add_argument('--save',required=True);a=p.parse_args()
torch.set_num_threads(1)
rows=[]
for method,lr in [('adamw',.0003),('adamw',.001),('muon',.006),('muon',.02)]:
    folder=Path(a.output_root)/'muon_pd_20261102'/a.dataset/f'{a.dataset}__scalar_k2__{method}_{lr}'/'seed000'
    for step in [0,1000,2500,5000,10000,15000,20000]:
        file=folder/'checkpoints'/f'step_{step:07d}.pt'
        checkpoint=torch.load(file,map_location='cpu',weights_only=False)
        assert checkpoint['step']==step
        assert checkpoint['config_sha256']==hashlib.sha256((folder/'config.json').read_bytes()).hexdigest()
        checksum=hashlib.sha256(file.read_bytes()).hexdigest()
        for name,value in checkpoint['model'].items():
            if '.ffn.' not in name or not name.endswith('.proj.weight'):continue
            x=value.double().numpy();s=np.linalg.svd(x,compute_uv=False)
            rownorm=np.linalg.norm(x,axis=1);colnorm=np.linalg.norm(x,axis=0)
            r=x/np.maximum(rownorm[:,None],1e-30);c=x/np.maximum(colnorm[None,:],1e-30)
            sr=np.linalg.svd(r,compute_uv=False)[0];sc=np.linalg.svd(c,compute_uv=False)[0]
            assert s[0]<=rownorm.max()*sr+1e-10 and s[0]<=colnorm.max()*sc+1e-10
            rows.append(dict(dataset=a.dataset,optimizer=method,lr=lr,step=step,layer=name,
                sigma_max=float(s[0]),stable_rank=float(np.sum(s*s)/s[0]**2),
                row_norm_mean=float(rownorm.mean()),row_norm_max=float(rownorm.max()),row_normalized_sigma=float(sr),
                col_norm_mean=float(colnorm.mean()),col_norm_max=float(colnorm.max()),col_normalized_sigma=float(sc),
                row_bound_slack=float(rownorm.max()*sr/s[0]),col_bound_slack=float(colnorm.max()*sc/s[0]),
                checkpoint_sha256=checksum))
        del checkpoint
save=Path(a.save);save.parent.mkdir(parents=True,exist_ok=True)
with save.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(dict(rows=len(rows),saved=str(save))))
