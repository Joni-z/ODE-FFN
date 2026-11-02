"""Read-only archived anchor identity and final post-relaxation logger checks."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output-root',required=True);p.add_argument('--old-output',required=True);p.add_argument('--cycle3-output',required=True);p.add_argument('--host',required=True);p.add_argument('--save',required=True);p.add_argument('--only-anchors',action='store_true');a=p.parse_args()
source=Path(a.source);sys.path.insert(0,str(source/'autoresearch_20261103/cycle8'));from trial8 import base,STRENGTH,CADENCE
sys.path.insert(0,str(source/'autoresearch_20261103/cycle6'));from prefix import numerical_equal
base.verify_sources();torch.set_num_threads(1);grid=json.loads((source/'autoresearch_20261103/cycle8/grid.json').read_text());rows=[];anchors=[]
for cell in [c for c in grid['cells'] if c['host']==a.host]:
    if a.only_anchors and cell['relaxation']!='none':continue
    folder=Path(a.output_root)/'professor_autoresearch_cycle8_20261103'/cell['dataset']/cell['id']/'seed000'
    assert (folder/'ADAPTER_COMPLETE').exists()
    file=folder/'checkpoints/step_0020000.pt';state=torch.load(file,map_location='cpu',weights_only=False)
    cfg=json.loads((folder/'config.json').read_text());assert state['step']==state['optimizer']['count']==20000
    assert state['config_sha256']==base.digest(folder/'config.json') and state['dataset_manifest_sha256']==base.digest(Path(cfg['data_root'])/'manifest.json')
    if cell['relaxation']=='none':
        root=Path(a.old_output if cell['parent_group']=='old' else a.cycle3_output)
        experiment='muon_pd_20261102' if cell['parent_group']=='old' else 'professor_autoresearch_cycle3_20261103'
        parent=root/experiment/cell['dataset']/cell['parent_id']/'seed000';parent_file=parent/'checkpoints/step_0020000.pt'
        assert base.digest(parent_file)==cell['parent_checkpoint_sha256']
        old=torch.load(parent_file,map_location='cpu',weights_only=False);numerical_equal(old,state)
        with np.load(parent/'final_test_examples.npz',allow_pickle=False) as l,np.load(folder/'final_test_examples.npz',allow_pickle=False) as r:
            assert np.array_equal(l['generated'],r['generated']) and np.array_equal(l['target'],r['target'])
        anchors.append(dict(cell=cell['id'],parent=cell['parent_id'],parent_sha256=base.digest(parent_file),exact_numerical_state=True,exact_samples=True))
    events=[json.loads(line) for line in (folder/'update_spectra.jsonl').read_text().splitlines()]
    assert len({e['step'] for e in events})==len(events)
    last=next(e for e in events if e['step']==20000)
    max_weight_error=0.;max_energy_error=0.;ratios=[]
    for event in events:
        applied=cell['relaxation']!='none' and event['step']%CADENCE==0
        assert event['relaxation_applied']==applied and event['relaxation']==cell['relaxation']
        assert event['strength']==(STRENGTH if applied else 0.)
        for name,m in event['matrices'].items():
            energy=m['norm_energy'];den=max(m['weight']['frobenius']**2,1.)
            err=abs(energy['observed_weight_squared_norm_change']-energy['radial_cross_term']-energy['update_squared_norm'])/den
            max_energy_error=max(max_energy_error,err);assert err<1e-12
            assert abs(m['delta']['frobenius']**2-energy['update_squared_norm'])/den<1e-12
            if not applied:assert m['relaxation_delta']['frobenius']==0.
            elif m['native_delta']['frobenius']:ratios.append(m['relaxation_delta']['frobenius']/m['native_delta']['frobenius'])
            if event['step']==20000:
                expected=base.spectral(state['model'][name]);err=max(abs(x-y) for x,y in zip(expected['singular_values'],m['weight']['singular_values']));max_weight_error=max(max_weight_error,err);assert err<1e-10
    rows.append(dict(cell=cell['id'],checkpoint_sha256=base.digest(file),events=len(events),total_delta_energy_identity_max_relative_error=max_energy_error,final_logged_weight_spectrum_error=max_weight_error,median_relaxation_over_native_delta_frobenius=float(np.median(ratios)) if ratios else 0.,logger_delta_is_total_post_relaxation=True))
assert len(rows)==(2 if a.only_anchors else 6) and len(anchors)==2
base.dump(a.save,dict(passed=True,partial=a.only_anchors,host=a.host,rows=rows,anchors=anchors,analysis_source_sha256=base.digest(__file__),scope='Native anchor states/samples exact; final logged spectra match checkpoint; total-delta energy identities and zero-relaxation events checked. Logged deltas are not independently replayed from unsaved pre-step states.'))
print(json.dumps(dict(passed=True,rows=len(rows),exact_anchors=len(anchors))))
