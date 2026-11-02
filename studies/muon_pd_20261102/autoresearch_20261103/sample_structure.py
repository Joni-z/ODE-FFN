"""Frozen Product16 sample shapes; CPU only, primary metrics unchanged."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--cases',required=True);p.add_argument('--save',required=True);a=p.parse_args()
torch.set_num_threads(1);manifest=json.loads(Path(a.cases).read_text());assert len(manifest['cases'])==74
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
digest_array=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
rows=[];histograms=[];reference=None;target_identity=None;seen={}
for case in manifest['cases']:
    folder=Path(case['folder']);assert (folder/'ADAPTER_COMPLETE').exists()
    cfg=json.loads((folder/'config.json').read_text());assert cfg['seeds']==[0] and cfg['steps']==20000 and cfg['sample_count']==4096 and cfg['heun_steps']==32
    checkpoint=folder/'checkpoints/step_0020000.pt';assert sha(checkpoint)==case['checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    data_manifest=Path(cfg['data_root'])/'manifest.json'
    assert state['step']==20000 and state['config_sha256']==sha(folder/'config.json') and state['dataset_manifest_sha256']==sha(data_manifest)
    assert state['identity']==dict(dataset=case['dataset'],model=case['id'],seed=0)
    spec=json.loads(data_manifest.read_text())['datasets'][case['dataset']];assert spec['family']=='rotated_product_mixture'
    distribution=spec['distribution'];rotation=np.asarray(distribution['rotation']);locations=np.asarray(distribution['locations']);variances=np.asarray(distribution['variances'])
    np.testing.assert_allclose(rotation.T@rotation,np.eye(16),atol=1e-12,rtol=1e-12)
    with np.load(folder/'final_test_examples.npz',allow_pickle=False) as data:
        generated=data['generated'].copy();target=data['target'].copy()
    assert generated.shape==target.shape==(4096,16) and np.isfinite(generated).all()
    target_hash=digest_array(target);target_identity=target_hash if target_identity is None else target_identity;assert target_hash==target_identity
    final=json.loads((folder/'metrics.jsonl').read_text().splitlines()[-1]);sampling=json.loads((folder/'final_test_sampling.json').read_text())
    assert final['step']==20000 and abs(final['validation_oracle_mse']-case['replayed_field_mse'])<2e-6
    assert sampling['sample']['heun_steps']==32 and abs(sampling['sample']['sliced_w2']-case['numpy_test_sw2'])<2e-6
    z=generated.astype(np.float64)@rotation;truth=target.astype(np.float64)@rotation
    if reference is None:
        with np.load(Path(cfg['data_root'])/case['dataset']/'test.npz',allow_pickle=False) as data:targets=data['x1'][:8192].copy()
        assert np.array_equal(targets[:4096],target)
        independent=targets[4096:].astype(np.float64)@rotation
        marginal_floor=np.mean((np.sort(truth,axis=0)-np.sort(independent,axis=0))**2,axis=0)
        population_gap=[]
        for mean,var in zip(locations,variances):
            normal_cdf=lambda v:.5*(1+math.erf(v/math.sqrt(2)))
            population_gap.append(normal_cdf(-mean/(2*math.sqrt(var)))-normal_cdf(-3*mean/(2*math.sqrt(var))))
        reference=dict(target_sha256=target_hash,rotation=rotation.tolist(),locations=locations.tolist(),variances=variances.tolist(),target_axis_w2_reference=float(np.sqrt(marginal_floor.mean())),target_gap_mass=np.mean(np.abs(truth)<locations[None,:]/2,axis=0).tolist(),population_gap_mass=population_gap,target_near_mode_mse=np.mean((np.abs(truth)-locations[None,:])**2,axis=0).tolist(),target_second_moment=np.mean(truth**2,axis=0).tolist())
    axis=np.sqrt(np.mean((np.sort(z,axis=0)-np.sort(truth,axis=0))**2,axis=0))
    gap=np.mean(np.abs(z)<locations[None,:]/2,axis=0);near=np.mean((np.abs(z)-locations[None,:])**2,axis=0);second=np.mean(z*z,axis=0)
    generated_hash=digest_array(generated);duplicate=seen.get(generated_hash);seen.setdefault(generated_hash,f"{case['group']}:{case['id']}")
    rows.append(dict(group=case['group'],cell=case['id'],family=case['family'],optimizer=case['optimizer'],lr=case['projection_lr'],validation_field_mse=case['replayed_field_mse'],primary_random_sw2=case['numpy_test_sw2'],latent_axis_sw2=float(np.sqrt(np.mean(axis**2))),central_gap_mass=float(gap.mean()),near_mode_mse=float(near.mean()),second_moment_mae=float(np.mean(np.abs(second-(locations**2+variances)))),axis_w2=axis.tolist(),axis_gap_mass=gap.tolist(),axis_near_mode_mse=near.tolist(),axis_second_moment=second.tolist(),generated_sha256=generated_hash,duplicate_samples_of=duplicate,checkpoint_sha256=case['checkpoint_sha256'],config_sha256=state['config_sha256'],manifest_sha256=state['dataset_manifest_sha256']))
    if case.get('case_study'):
        for j in [0,7,15]:
            limit=locations[j]+6*math.sqrt(variances[j]);edges=np.linspace(-limit,limit,61);counts,_=np.histogram(z[:,j],edges)
            histograms.append(dict(group=case['group'],cell=case['id'],label=case['case_study'],axis=j,edges=edges.tolist(),counts=counts.tolist(),sample_count=len(z),outside_fraction=float(np.mean(np.abs(z[:,j])>limit))))
    del state
out=Path(a.save);out.parent.mkdir(parents=True,exist_ok=True)
result=dict(rows=rows,histograms=histograms,reference=reference,cases_manifest_sha256=sha(a.cases),analysis_source_sha256=sha(__file__),unique_generated_arrays=len(seen),scope='Frozen test arrays, CPU diagnostic only; no new primary metric, model selection, generation or training.')
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(rows=len(rows),histograms=len(histograms),unique_generated_arrays=len(seen),saved=str(out))))
