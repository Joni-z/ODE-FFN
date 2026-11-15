"""Read-only oracle floor and exact finite-orbit error decomposition for Ring8."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from evaluate import base, torch, C8, Oracle, restore, array_hash, self_test

p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
a=p.parse_args();root=Path(a.runtime);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1);base.verify_sources();base.dump(out/'self-test.json',self_test(a.device))
folders=[]
for gpu in range(4,8):
 for folder in (root/f'gpu{gpu}/evaluations').glob('*'):
  if not (folder/'COMPLETE.json').exists():continue
  b=json.loads((folder/'binding.json').read_text())
  if b['source_config']['cell']['dataset']=='gmm_ring8_2d':folders.append((folder,b))
assert len(folders)==4,'All four Ring8 evaluations must be complete'
with np.load(folders[0][0]/'banks.npz') as z:arrays={k:z[k].copy() for k in z.files}
hashes={k:array_hash(v) for k,v in arrays.items()}
assert all(b['bank_hashes']==hashes for _,b in folders)
cell=folders[0][1]['source_config']['cell'];cache=base.verify_data(a.data_root,cell)
manifest=cache/'manifest.json';spec=json.loads(manifest.read_text())['datasets'][cell['dataset']]
law=base.train.load_distribution(spec);oracle=Oracle(law);directions=torch.from_numpy(arrays['directions']).to(a.device).double()
values=[];banks=[]
for bank in range(8):
 source=torch.from_numpy(arrays[f'source_{bank}']).to(a.device)
 y=base.train.generated_samples(oracle,source,128);values.append(y.cpu().numpy())
 target=torch.from_numpy(arrays[f'target_{bank}']).to(a.device)
 sw2=base.train.sliced_w2(y.double(),target.double(),directions)
 banks.append(sw2);np.save(out/f'oracle-b{bank}.npy',values[-1])
 base.dump(out/'progress.json',dict(phase='oracle',bank=bank))
pooled=torch.from_numpy(np.concatenate(values)).to(a.device)
target=torch.from_numpy(np.concatenate([arrays[f'target_{i}'] for i in range(8)])).to(a.device)
counts=np.bincount(np.argmax(pooled.cpu().numpy()@law.means.T,axis=1),minlength=8)
base.dump(out/'oracle.json',dict(bank_sw2=banks,sw2_mean=float(np.mean(banks)),sw2_sd=float(np.std(banks,ddof=1)),pooled_sw2=base.train.sliced_w2(pooled.double(),target.double(),directions),counts=counts.tolist(),sector_l1=float(np.abs(counts/len(pooled)-law.weights).sum()),heun_steps=128,bank_hashes=hashes))
fixed=base.train.fixed_eval_data(cache/cell['dataset'],law,2,0,torch.device(a.device),4096,256,'test')
rows=[]
for folder,b in folders:
 cfg=b['source_config'];training=Path(b['source_folder']);family=cfg['cell']['family']
 for step in (15000,20000):
  model,identity=restore(training,cfg,manifest,step,a.device)
  # Evaluate in float64 for the finite-orbit algebra; do not replace float32 sampling results.
  model=model.double();projected=C8(model).to(a.device);x=fixed['x'].double();t=fixed['t'].double()
  rot=projected.rotations.to(x);orbit=torch.einsum('nd,ked->kne',x,rot).reshape(-1,2)
  times=t.expand(8,*t.shape).reshape(-1,1)
  with torch.no_grad():
   v=model(orbit,times);vp=projected(orbit,times);u=oracle(orbit,times)
   total=(v-u).square().mean();symmetric=(vp-u).square().mean();asymmetric=(v-vp).square().mean()
   cross=2*((vp-u)*(v-vp)).mean()
  residual=float((total-symmetric-asymmetric).abs())
  assert residual<1e-10 and abs(float(cross))<1e-10
  row=dict(family=family,step=step,total_orbit_mse=float(total),symmetric_orbit_mse=float(symmetric),asymmetric_orbit_mse=float(asymmetric),asymmetric_fraction=float(asymmetric/total),cross_term=float(cross),identity_residual=residual,**identity)
  rows.append(row);base.dump(out/'decomposition.json',rows)
base.dump(out/'COMPLETE.json',dict(passed=True,models=4,checkpoints=8,source_sha256=base.digest(__file__),scope='Oracle sampling uses native float32 Heun128; exact finite-orbit algebra evaluated in float64'))
print(json.dumps(dict(passed=True,output=str(out))))
