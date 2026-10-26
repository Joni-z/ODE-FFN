"""Verify only constructed witnesses and previously exposed positive-input panels."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from runtime_helpers import ARMS,TIMES,MODELS,PROBES,GLU,DATASET,load,sha,setup,build,capture,check_alias
from manual_forward import manual
from parity_math import self_test

def main(root):
    torch.set_num_threads(1);started=time.monotonic();setup();alias=check_alias();checks=[]
    rotation=np.array(load(PROBES/'population.json')['rotation'])
    for arm in ARMS:
        m,cp=build(arm);digest=sha(cp);native=0
        if arm!='sigmoid_glu128':
            for t in TIMES:
                with np.load(MODELS/'banks'/DATASET/f'panel_{t:.1f}.npz') as bank,np.load(cp.parent/'diagnostics'/f'panel_{t:.1f}.npz') as saved:
                    with torch.enable_grad():value=m(torch.from_numpy(bank['x']).requires_grad_(True),torch.from_numpy(bank['t'])).detach().numpy()
                    np.testing.assert_array_equal(value,saved['prediction']);native+=1
        m=m.double()
        if arm=='sigmoid_glu128':
            for t in TIMES:
                with np.load(GLU/'diagnostics'/DATASET/'sigmoid_glu'/f'common-{t:.1f}.npz') as bank:
                    with torch.no_grad():value=m(torch.from_numpy(bank['x']),torch.full((len(bank['x']),1),t,dtype=torch.float64)).numpy()
                    np.testing.assert_array_equal(value,bank['prediction']);native+=1
        with np.load(PROBES/'banks/time-0.5.npz') as bank:x=bank['A'][:256]@rotation.T
        actual,features,meta=capture(m,x,.5,arm);explicit,manual_features,manual_meta=manual(m,x,.5,arm)
        np.testing.assert_array_equal(actual,explicit);assert meta==manual_meta
        assert set(features)==set(manual_features)
        for key,parts in features.items():
            assert set(parts)==set(manual_features[key])
            for part,value in parts.items():np.testing.assert_array_equal(value,manual_features[key][part])
        if arm!='sigmoid_glu128':
            with np.load(PROBES/'fields'/arm/'time-0.5.npz') as saved:np.testing.assert_array_equal(actual@rotation,saved['native'][:256])
        assert sha(cp)==digest
        checks.append(dict(arm=arm,native_panels=native,positive_manual_rows=256,sites=len(meta),checkpoint_sha256=digest))
    result=dict(status='PASS',elapsed_seconds=time.monotonic()-started,self_tests=self_test(),alias=alias,models=checks,
        new_negative_input_model_evaluations=0,new_model_updates=0)
    path=root/'preflight.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
