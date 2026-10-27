"""Reuse an identical, pre-quality reference feature bank without recomputing it."""
import argparse,json,shutil,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--previous',type=Path,required=True);a=ap.parse_args()
sys.path.insert(0,str(a.root/'source'));import study
study.verify(a.root);study.verify(a.previous,True)
for name in ('image_metrics.py','metric_core.py'):
    assert study.sha(a.root/'source'/name)==study.sha(a.previous/'source'/name)
assert study.sha(a.root/'banks/fid-reference.uint8.npy')==study.sha(a.previous/'banks/fid-reference.uint8.npy')
old=study.load(a.previous/'bindings.json');new=study.load(a.root/'bindings.json')
for p,h in old.items():
    if '/site-packages/torch_fidelity/' in p or p.endswith('.pth'):assert new[p]==h
folder=a.root/'metric-inputs';folder.mkdir(exist_ok=False);path=folder/'reference-features.npy'
shutil.copyfile(a.previous/'metric-inputs/reference-features.npy',path)
assert study.sha(path)==study.sha(a.previous/'metric-inputs/reference-features.npy')
study.write(a.root/'metric-bindings.json',{str(path):study.sha(path)})
study.write(a.root/'metric-ready.json',dict(status='PASS',reference_images=2048,features=2048,device='mps',
    reused_from=str(a.previous),reason='Byte-identical references, extractor source, metric source, package sources and pretrained weights before any predictor quality run.'))
print(json.dumps(study.load(a.root/'metric-ready.json')))
