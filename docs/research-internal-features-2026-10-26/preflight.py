"""Native-hook and already-known final-readout compatibility before internal fits."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from run_probe import ARMS,TIMES,DATASET,extract,sites,load,sha
from manual_features import manual
from probe_math import design,solve,predict,self_test

def main(root):
    torch.set_num_threads(1);base=Path('/Users/mr.z/ode-workspace-prep/architecture-search')
    models=base/'conservative_readout_20261026/output';probes=base/'coordinate_dependence_20261026/output-v2';previous=base/'frozen_features_20261026/output'
    provenance=load(models/'provenance.json');old=Path(provenance['source_root'])
    for name,digest in provenance['source_sha256'].items():assert sha(old/name)==digest
    sys.path.insert(0,str(old))
    from synthetic.conservative_readout import build
    from synthetic.train import restore_model
    population=load(probes/'population.json');rotation=np.array(population['rotation']);mu=np.array(population['locations']);settings=load(models/'study.json')['arms']
    with np.load(probes/'standardized-draws.npz') as d:pair=(d['sign_A']*mu+.2*d['target_noise_A']-d['source_A'])@rotation.T
    max_compat=0.;checks=[]
    for arm in ARMS:
        s=settings[arm];folder=models/'runs'/DATASET/arm;cp=folder/'step_0005000.pt';digest=sha(cp)
        m=build(16,s['family'],s['readout']).eval();restore_model(m,torch.load(cp,map_location='cpu',weights_only=False))
        for p in m.parameters():p.requires_grad_(False)
        for t in TIMES:
            with np.load(models/'banks'/DATASET/f'panel_{t:.1f}.npz') as b,np.load(folder/'diagnostics'/f'panel_{t:.1f}.npz') as wanted:
                with torch.enable_grad():value=m(torch.from_numpy(b['x']).requires_grad_(True),torch.from_numpy(b['t'])).detach().numpy()
                np.testing.assert_array_equal(value,wanted['prediction'])
        m=m.double();t=.5
        with np.load(probes/'banks'/f'time-{t:.1f}.npz') as b:x=b['A'][:256]@rotation.T
        hook,va=extract(m,x,t,arm);explicit,vb=manual(m,x,t,arm)
        assert set(hook)==set(explicit)==set(sites(arm))
        for site in hook:np.testing.assert_array_equal(hook[site],explicit[site])
        np.testing.assert_array_equal(va,vb)
        with np.load(probes/'fields'/arm/f'time-{t:.1f}.npz') as wanted:np.testing.assert_array_equal(va@rotation,wanted['native'][:256])
        old_features=[];residual=[];native=[]
        for t in TIMES:
            with np.load(previous/arm/f'features_{t:.1f}.npz') as oldf:feature=oldf['post_norm'];base_latent=oldf['native']
            with np.load(probes/'banks'/f'time-{t:.1f}.npz') as bank:mean=bank['oracle']@rotation.T
            base_physical=feature@m.raw.output.weight.detach().numpy().T+m.raw.output.bias.detach().numpy()
            old_features.append(feature);residual.append(np.column_stack([mean-base_physical,pair-base_physical]));native.append(base_latent)
        n=len(pair);coeff=[]
        for sl in [slice(0,n//2),slice(n//2,n)]:
            xx=design(np.concatenate([f[sl] for f in old_features]));yy=np.concatenate([y[sl] for y in residual]);c,_=solve(xx,yy);coeff.append(c)
        coeff=np.array(coeff)
        for j,t in enumerate(TIMES):
            value=native[j][:,None,:]+predict(old_features[j],coeff,rotation)
            with np.load(previous/arm/f'predictions_{t:.1f}.npz') as wanted:
                max_compat=max(max_compat,float(np.max(np.abs(value-wanted['post_norm_shared_A']))))
                np.testing.assert_allclose(value,wanted['post_norm_shared_A'],atol=1e-10,rtol=1e-10)
        assert sha(cp)==digest
        checks.append(dict(arm=arm,sites=len(hook),native_float32_panels=5,manual_feature_rows=256,checkpoint_sha256=digest))
    result=dict(status='PASS',self_test=self_test(),models=checks,known_final_class_max_error=max_compat,known_final_compatibility_fits=8,new_internal_fits=0)
    dest=root/'preflight.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',sites=sum(r['sites'] for r in checks),native_panels=20,max_final_compatibility=max_compat,new_internal_fits=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
