"""Fixed-weight native upper-duration intervention on the complete validation bank."""
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT,DEST=map(Path,sys.argv[1:3])
sys.path.insert(0,str(ROOT/'source'))
import curvature_study as study
from teacher_support import TEACHERS,load,write,sha,signature

torch.set_num_threads(1)
out=ROOT/'upper-clamp';out.mkdir(exist_ok=False)
rows=[];max_explicit=0.
for teacher in TEACHERS:
    for position in range(3):
        for key,(module,family) in study.subjects(teacher,position).items():
            if family!='ode_k2':continue
            module=module.float();before=signature(module)
            clone=copy.deepcopy(module)
            for layer in [clone.ode1,clone.ode2]:layer.gate_source='provided'
            z=np.load(study.PARENT/'output/cache'/f'validation-z{position}.npy',mmap_mode='r')
            target=np.load(study.PARENT/'output/cache'/f'validation-{teacher}-y{position}.npy').astype(np.float64)
            native,clamped=[],[]
            with torch.no_grad():
                for start in range(0,len(z),1024):
                    x=torch.from_numpy(np.array(z[start:start+1024],copy=True))
                    native.append(module(x).numpy());clamped.append(clone(x,torch.ones((len(x),1))).numpy())
            native,clamped=np.concatenate(native),np.concatenate(clamped)
            if key=='target':expected=np.load(study.PARENT/'output/cache'/f'validation-ode_k2-y{position}.npy')
            else:expected=np.load(study.PARENT/'output/runs'/f'{teacher}--ode_k2--{position}'/'validation_0005000.npy')
            assert np.array_equal(native,expected)
            double=copy.deepcopy(clone).double();x=torch.from_numpy(np.array(z[:128],copy=True)).double()
            eye=torch.eye(64,dtype=torch.float64)
            with torch.no_grad():
                p1,p2=double.ode1.proj.weight,double.ode2.proj.weight
                q1,q2=eye+p1+.5*p1@p1,eye+p2+.5*p2@p2
                explicit=torch.nn.functional.silu(x@q1.T)@q2.T
                actual=double(x,torch.ones((len(x),1),dtype=torch.float64))
            torch.testing.assert_close(actual,explicit,rtol=2e-8,atol=5e-10)
            max_explicit=max(max_explicit,float((actual-explicit).abs().max()))
            base=native.astype(np.float64);upper=clamped.astype(np.float64)
            delta=float(np.mean((base-upper)**2));native_energy=float(np.mean(base**2));target_energy=float(np.mean(target**2))
            native_risk=float(np.mean((base-target)**2));upper_risk=float(np.mean((upper-target)**2))
            np.save(out/f'{teacher}--{position}--{key}.npy',clamped)
            rows.append(dict(teacher=teacher,position=position,subject=key,
                native_clamped_difference_mse=delta,native_output_energy=native_energy,
                native_clamped_relative_squared_difference=delta/native_energy,
                native_teacher_mse=native_risk,clamped_teacher_mse=upper_risk,target_energy=target_energy,
                native_relative_teacher_mse=native_risk/target_energy,clamped_relative_teacher_mse=upper_risk/target_energy))
            assert before==signature(module)
            for name,value in module.state_dict().items():assert torch.equal(value,clone.state_dict()[name])
with (DEST/'upper-clamp.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
receipt=dict(status='PASS',modules=len(rows),validation_rows_per_module=20000,native_replay='all bitwise',
             maximum_explicit_map_difference=max_explicit,source_sha256=sha(__file__),
             arrays_sha256={p.name:sha(p) for p in out.glob('*.npy')})
write(DEST/'upper-clamp-receipt.json',receipt)
print(receipt)
for row in rows:print(row['teacher'],row['position']+1,row['subject'],'function relative RMS %',100*np.sqrt(row['native_clamped_relative_squared_difference']),
                      'native/clamped relative teacher MSE %',100*row['native_relative_teacher_mse'],100*row['clamped_relative_teacher_mse'])
