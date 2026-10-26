"""Compare every hybrid's hook and manual tensors without archiving duplicates."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
import psutil

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(root):
    torch.set_num_threads(1);started=time.monotonic();out=root/'output';dest=root/'hybrid-feature-audit.json'
    assert not dest.exists() and load(root/'independent-audit.json')['status']=='PASS'
    source=root/'source';manifest=load(source/'source-sha256.json')
    for name,wanted in manifest.items():assert sha(source/name)==wanted
    sys.path.insert(0,str(source))
    from run_probe import extract,ARMS,TIMES,sites
    from manual_features import manual
    base=Path('/Users/mr.z/ode-workspace-prep/architecture-search');models=base/'conservative_readout_20261026/output';probes=base/'coordinate_dependence_20261026/output-v2'
    provenance=load(models/'provenance.json');old=Path(provenance['source_root'])
    sys.path.insert(0,str(old))
    from synthetic.conservative_readout import build
    from synthetic.train import restore_model
    settings=load(models/'study.json')['arms'];rotation=np.array(load(probes/'population.json')['rotation'])
    inputs=load(out/'input-bindings.json')
    for name,wanted in inputs.items():assert sha(name)==wanted,name
    tensors=0;cases=0;peak=0.
    for arm in ARMS:
        s=settings[arm];model=build(16,s['family'],s['readout']).eval()
        restore_model(model,torch.load(models/'runs'/'rotated_product_bimodal_16d'/arm/'step_0005000.pt',map_location='cpu',weights_only=False))
        for p in model.parameters():p.requires_grad_(False)
        model=model.double()
        for t in TIMES:
            with np.load(probes/'banks'/f'time-{t:.1f}.npz') as bank:a=bank['A'];b=bank['B']
            for i in range(16):
                q=b.copy();q[:,i]=a[:,i];x=q@rotation.T
                hooked,y=extract(model,x,t,arm);explicit,z=manual(model,x,t,arm)
                assert set(hooked)==set(explicit)==set(sites(arm))
                for key in hooked:np.testing.assert_array_equal(hooked[key],explicit[key]);tensors+=1
                np.testing.assert_array_equal(y,z);cases+=1
            rss=psutil.Process().memory_info().rss/2**20;peak=max(peak,rss)
            print(json.dumps(dict(phase='all hybrid tensor equality',arm=arm,time=t,pid=psutil.Process().pid,rss_mib=rss)),flush=True)
        del model,hooked,explicit
    assert cases==320 and tensors==4960
    for name,wanted in inputs.items():assert sha(name)==wanted,name
    for name,wanted in manifest.items():assert sha(source/name)==wanted
    result=dict(status='PASS',cases=cases,feature_tensors_bitwise=tensors,native_output_cases_bitwise=cases,
        elapsed_seconds=time.monotonic()-started,peak_observed_rss_mib=peak,source_sha256=sha(__file__),
        frozen_source_manifest_sha256=sha(source/'source-sha256.json'),input_bindings_sha256=sha(out/'input-bindings.json'))
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
