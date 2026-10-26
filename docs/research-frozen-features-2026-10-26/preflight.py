"""Actual immutable-model compatibility before frozen-feature fits."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from run_features import ARMS,TIMES,DATASET,features,load,sha
from linear_probe import self_test

def main(root):
    torch.set_num_threads(1)
    models=Path('/Users/mr.z/ode-workspace-prep/architecture-search/conservative_readout_20261026/output')
    probes=Path('/Users/mr.z/ode-workspace-prep/architecture-search/coordinate_dependence_20261026/output-v2')
    provenance=load(models/'provenance.json');old=Path(provenance['source_root'])
    for path,digest in provenance['source_sha256'].items():assert sha(old/path)==digest
    sys.path.insert(0,str(old))
    from synthetic.conservative_readout import build
    from synthetic.train import restore_model
    settings=load(models/'study.json')['arms'];pop=load(probes/'population.json');rotation=np.array(pop['rotation'])
    checks=[]
    for arm in ARMS:
        s=settings[arm];folder=models/'runs'/DATASET/arm;checkpoint=folder/'step_0005000.pt';initial_hash=sha(checkpoint)
        m=build(16,s['family'],s['readout']).eval();snapshot=torch.load(checkpoint,map_location='cpu',weights_only=False);restore_model(m,snapshot)
        for p in m.parameters():p.requires_grad_(False)
        for t in TIMES:
            with np.load(models/'banks'/DATASET/f'panel_{t:.1f}.npz') as bank,np.load(folder/'diagnostics'/f'panel_{t:.1f}.npz') as target:
                with torch.enable_grad():v=m(torch.from_numpy(bank['x']).requires_grad_(True),torch.from_numpy(bank['t'])).detach().numpy()
                np.testing.assert_array_equal(v,target['prediction'])
        m=m.double();t=.5
        with np.load(probes/'banks'/f'time-{t:.1f}.npz') as bank:x=bank['A'][:256]@rotation.T
        f,v=features(m,x,t)
        for key,h in f.items():assert h.shape==(256,64) and np.isfinite(h).all(),key
        with np.load(probes/'fields'/arm/f'time-{t:.1f}.npz') as saved:np.testing.assert_array_equal(v@rotation,saved['native'][:256])
        with torch.no_grad():normal=m.raw.norm(torch.from_numpy(f['pre_norm'])).numpy()
        np.testing.assert_array_equal(normal,f['post_norm'])
        np.testing.assert_allclose(f['post_norm']@m.raw.output.weight.detach().numpy().T+m.raw.output.bias.detach().numpy(),v,atol=1e-12)
        assert sha(checkpoint)==initial_hash
        checks.append(dict(arm=arm,native_float32_panels=5,native_float64_rows=256,checkpoint_sha256=initial_hash))
    result=dict(status='PASS',self_test=self_test(),actual_models=checks,source_root=str(old),new_head_fits=0)
    path=root/'preflight.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',models=4,native_panels=20,feature_rows=1024,new_head_fits=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
