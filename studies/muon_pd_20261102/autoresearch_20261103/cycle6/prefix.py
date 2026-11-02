"""Compare numerical states while allowing explicit run/config metadata changes."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from trial6 import base

NUMERICAL_KEYS=['model','runtime_buffers','optimizer','data_rng','torch_rng','cuda_rng','numpy_rng','python_rng']

def same(left,right,path='state'):
    if torch.is_tensor(left):
        assert torch.is_tensor(right) and left.dtype==right.dtype and torch.equal(left.cpu(),right.cpu()),path
    elif isinstance(left,np.ndarray):assert isinstance(right,np.ndarray) and left.dtype==right.dtype and np.array_equal(left,right),path
    elif isinstance(left,dict):
        assert left.keys()==right.keys(),path
        for key in left:same(left[key],right[key],f'{path}/{key}')
    elif isinstance(left,(list,tuple)):
        assert type(left)==type(right) and len(left)==len(right),path
        for i,(a,b) in enumerate(zip(left,right)):same(a,b,f'{path}/{i}')
    else:assert left==right,path


def numerical_equal(left,right):
    assert left['step']==right['step']
    for key in NUMERICAL_KEYS:same(left[key],right[key],key)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);p.add_argument('--old-output',required=True);p.add_argument('--cycle3-output',required=True);p.add_argument('--host',required=True);p.add_argument('--save',required=True);p.add_argument('--only-ready',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);grid=json.loads(Path(__file__).with_name('grid.json').read_text());rows=[];pending=[]
    for cell in [c for c in grid['cells'] if c['host']==a.host]:
        folder=Path(a.output_root)/'professor_autoresearch_cycle6_20261103'/cell['dataset']/cell['id']/'seed000'
        try:step=json.loads((folder/'metrics.jsonl').read_text().splitlines()[-1])['step']
        except (FileNotFoundError,json.JSONDecodeError,IndexError):step=0
        if step<20500:
            pending.append(cell['id']);continue
        root=Path(a.old_output if cell['parent_group']=='old' else a.cycle3_output)
        experiment='muon_pd_20261102' if cell['parent_group']=='old' else 'professor_autoresearch_cycle3_20261103'
        parent=root/experiment/cell['dataset']/cell['parent_id']/'seed000';original_file=parent/'checkpoints/step_0020000.pt';new_file=folder/'checkpoints/step_0020000.pt'
        assert base.digest(original_file)==cell['parent_checkpoint_sha256']
        original=torch.load(original_file,map_location='cpu',weights_only=False);new=torch.load(new_file,map_location='cpu',weights_only=False)
        assert original['step']==new['step']==20000
        for state,directory in [(original,parent),(new,folder)]:
            cfg=json.loads((directory/'config.json').read_text());assert state['config_sha256']==base.digest(directory/'config.json') and state['dataset_manifest_sha256']==base.digest(Path(cfg['data_root'])/'manifest.json')
        assert original['dataset_manifest_sha256']==new['dataset_manifest_sha256'];numerical_equal(original,new)
        rows.append(dict(cell=cell['id'],parent=cell['parent_id'],parent_checkpoint_sha256=base.digest(original_file),new_prefix_sha256=base.digest(new_file),exact_numerical_state=True))
    assert a.only_ready or not pending,pending
    result=dict(passed=True,rows=rows,pending=pending,scope='All numerical model/optimizer/buffer/RNG states match; different explicit configuration/run metadata retained.')
    base.dump(a.save,result);print(json.dumps(result))
