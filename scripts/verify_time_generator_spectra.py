"""Independently bind raw inputs and verify base/effective generator spectra."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('group','prior','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    group,prior=args.group.resolve(),args.prior.resolve()
    dataset='rotated_product_bimodal_16d';base_count=effective_count=0
    for mode in ('static','time'):
        run=group/'runs'/dataset/f'ode_k2_{mode}_final'
        trajectory=json.loads((run/'effective-generator-trajectory.json').read_text())
        for step in (0,1000,2500,5000):
            state=torch.load(run/f'step_{step:07d}.pt',map_location='cpu',weights_only=False)['model']
            base_records=json.loads((run/f'spectrum_{step:07d}.json').read_text())
            for name,record in trajectory[str(step)].items():
                p=state[name+'.proj.weight'].double()
                singular=torch.linalg.svdvals(p).numpy();eigen=torch.linalg.eigvals(p).numpy()
                # The base logger iterates model.raw; effective records iterate the wrapper.
                raw=base_records[name.removeprefix('raw.')]
                np.testing.assert_allclose(singular,raw['singular_values'],rtol=1e-12,atol=1e-13)
                np.testing.assert_allclose(eigen.real,raw['eigenvalue_real'],rtol=1e-12,atol=1e-13)
                np.testing.assert_allclose(eigen.imag,raw['eigenvalue_imag'],rtol=1e-12,atol=1e-13)
                base_count+=1
                delta=state[name+'.adapter.up.weight'].double()@state[name+'.adapter.down.weight'].double()
                for time_record in record['times']:
                    coefficient=math.sqrt(12)*(time_record['time']-.5) if mode=='time' else 1.
                    matrix=p+coefficient*delta;eigen=torch.linalg.eigvals(matrix).numpy()
                    np.testing.assert_allclose(torch.linalg.svdvals(matrix).numpy(),time_record['singular_values'],rtol=1e-12,atol=1e-13)
                    np.testing.assert_allclose(eigen.real,time_record['eigenvalue_real'],rtol=1e-12,atol=1e-13)
                    np.testing.assert_allclose(eigen.imag,time_record['eigenvalue_imag'],rtol=1e-12,atol=1e-13)
                    effective_count+=1
    assert base_count==48 and effective_count==240
    files=[p for p in group.rglob('*') if p.is_file()]
    old_config=json.loads((prior/'study.json').read_text())
    for arm,setting in old_config['arms'].items():
        if (setting['family'].startswith('swiglu') and setting['mode']!='shared') or arm=='ode_k2_native_final':
            folder=prior/'runs'/dataset/arm
            files.extend([folder/'step_0000000.pt',folder/'step_0005000.pt',folder/'result.json'])
            files.extend(p for p in (folder/'diagnostics').glob('*') if p.is_file())
    files.extend(p for p in (prior/'banks'/dataset).glob('*') if p.is_file())
    receipt=dict(status='PASS',base_matrices=base_count,effective_matrices=effective_count,
        eigenvalue_checks='real and imaginary components',singular_value_checks='full spectra',
        script_sha256=sha(Path(__file__)),bindings={str(p):sha(p) for p in sorted(set(files))})
    args.output.resolve().write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',base_matrices=base_count,effective_matrices=effective_count,bindings=len(receipt['bindings']))))
