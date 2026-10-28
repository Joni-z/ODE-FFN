"""Correct reused native out-of-range metadata, preserving scores and outputs."""
from pathlib import Path
import argparse,copy,json,sys
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
sys.path.insert(0,str(root/'source'));import study
study.verify(root,True);assert study.load(root/'orchestrator/status.json')['state']=='COMPLETE'
assert study.load(root/'audit-features/result.json')['status']=='PASS'
folder=root/'metric-metadata-v2';folder.mkdir();source=Path(__file__);(folder/'correct_metadata.py').write_bytes(source.read_bytes())
original=study.load(root/'metric-output/result.json');corrected=copy.deepcopy(original);changes=[];bindings={}
for row in corrected['rows']:
    arm=row['family']
    path=root/'output/runs'/arm.removesuffix('_constant')/'generated.npy' if arm.endswith('_constant') else study.paths(root)['resolution_parent']/'output/runs'/arm/'generated-128.npy'
    bindings[str(path)]=study.sha(path);value=float(np.mean(np.abs(np.load(path))>1))
    if arm in study.NATIVE:
        changes.append(dict(family=arm,field='out_of_range_fraction',previous=float(row['out_of_range_fraction']),corrected=value,source=str(path)))
        row['out_of_range_fraction']=value
    else:assert row['out_of_range_fraction']==value
for old,new in zip(original['rows'],corrected['rows']):
    for key in old:
        if key!='out_of_range_fraction':assert old[key]==new[key]
assert len(changes)==4
for file,digest in bindings.items():assert study.sha(file)==digest
study.write(folder/'result.json',corrected);study.write(folder/'receipt.json',dict(status='PASS',changed_fields=changes,unchanged_FIDs=True,unchanged_losses=True,unchanged_raw_bindings=bindings,source_sha256=study.sha(source)))
print(json.dumps(dict(status='PASS',changed_fields=changes)),flush=True)
