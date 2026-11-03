"""Exact native20k reference states/samples; different run metadata is explicit."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output-root',required=True);p.add_argument('--old-output',required=True);p.add_argument('--cycle3-output',required=True);p.add_argument('--host',required=True);p.add_argument('--save',required=True);p.add_argument('--only-ready',action='store_true');a=p.parse_args()
source=Path(a.source).resolve();sys.path.insert(0,str(source/'autoresearch_20261103/cycle10'));from trial10 import base
sys.path.insert(0,str(source/'autoresearch_20261103/cycle6'));from prefix import numerical_equal
base.verify_sources();torch.set_num_threads(1);grid=json.loads((source/'autoresearch_20261103/cycle10/grid.json').read_text());rows=[];pending=[]
for cell in [c for c in grid['cells'] if c['host']==a.host and c['role']=='anchor']:
    folder=Path(a.output_root)/'professor_autoresearch_cycle10_20261103'/cell['dataset']/cell['id']/'seed000'
    if not (folder/'ADAPTER_COMPLETE').exists():pending.append(cell['id']);continue
    root=Path(a.old_output if cell['parent_group']=='old' else a.cycle3_output)
    experiment='muon_pd_20261102' if cell['parent_group']=='old' else 'professor_autoresearch_cycle3_20261103'
    parent=root/experiment/cell['dataset']/cell['parent_id']/'seed000';parent_file=parent/'checkpoints/step_0020000.pt';file=folder/'checkpoints/step_0020000.pt'
    assert base.digest(parent_file)==cell['parent_checkpoint_sha256']
    left=torch.load(parent_file,map_location='cpu',weights_only=False);right=torch.load(file,map_location='cpu',weights_only=False)
    for state,directory in [(left,parent),(right,folder)]:
        cfg=json.loads((directory/'config.json').read_text());assert cfg['steps']==state['step']==20000
        assert state['config_sha256']==base.digest(directory/'config.json') and state['dataset_manifest_sha256']==base.digest(Path(cfg['data_root'])/'manifest.json')
    numerical_equal(left,right)
    with np.load(parent/'final_test_examples.npz',allow_pickle=False) as l,np.load(folder/'final_test_examples.npz',allow_pickle=False) as r:
        assert np.array_equal(l['generated'],r['generated']) and np.array_equal(l['target'],r['target'])
    rows.append(dict(cell=cell['id'],parent=cell['parent_id'],parent_sha256=base.digest(parent_file),new_checkpoint_sha256=base.digest(file),exact_numerical_state=True,exact_samples=True))
assert a.only_ready or (not pending and len(rows)==4)
base.dump(a.save,dict(passed=True,partial=a.only_ready,host=a.host,anchors=rows,pending=pending,analysis_source_sha256=base.digest(__file__)))
print(json.dumps(dict(passed=True,anchors=len(rows),pending=pending)))
