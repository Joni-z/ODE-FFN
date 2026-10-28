"""Post-timing audit: replay all original starts, without another speed trial."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from deployment import load_export
from benchmark import sample,PARENTS

root=Path(sys.argv[1]);torch.set_num_threads(1)
model=load_export(root/'export')
source=np.load(PARENTS['native']/'banks/sampling-source.npy')
reference=np.load(PARENTS['maps']/'output/runs/ode_k2_folded/generated.npy')
assert source.shape==reference.shape==(1024,3,32,32)
answers=[]
with torch.no_grad():
    for start in range(0,len(source),64):
        actual=sample(model,source[start:start+64].copy(),'cpu',time.monotonic()+120)
        assert actual.tobytes()==reference[start:start+64].tobytes(),start
        answers.append(actual)
        (root/'full-replay-progress.json').write_text(json.dumps(dict(verified_images=start+len(actual),total=1024))+'\n')
np.save(root/'export-generated-1024.npy',np.concatenate(answers))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(root/'export-generated-1024.npy')==sha(PARENTS['maps']/'output/runs/ode_k2_folded/generated.npy')
result=dict(status='PASS',images=1024,steps=128,bitwise=True,backend='cpu',
    reference_sha256=sha(PARENTS['maps']/'output/runs/ode_k2_folded/generated.npy'),
    inherited_FID=179.7184452562523,new_FID_estimate=False,
    scope='Post-timing verification of the unchanged export; no speed result or model selection.')
(root/'full-replay.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
