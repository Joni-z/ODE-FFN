"""CPU-only registered examples/readouts of independently replayed endpoints."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);p.add_argument('--audit',required=True);p.add_argument('--save',required=True);p.add_argument('--endpoint-step',type=int,default=80000);p.add_argument('--expected-count',type=int,default=4);p.add_argument('--experiment-id',default='professor_autoresearch_cycle6_20261103');a=p.parse_args()
torch.set_num_threads(1);audit=json.loads(Path(a.audit).read_text());assert audit['passed'] and audit['endpoint_step']==a.endpoint_step and len(audit['endpoints'])==a.expected_count
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
rows=[];examples=[];reference=None;target_hash=None
for row in audit['endpoints']:
    cell=row['id'];ds=row['dataset'];folder=Path(a.output_root)/a.experiment_id/ds/cell/'seed000';cfg=json.loads((folder/'config.json').read_text())
    checkpoint=folder/f'checkpoints/step_{a.endpoint_step:07d}.pt';assert sha(checkpoint)==row['checkpoint_sha256'] and cfg['steps']==a.endpoint_step and cfg['heun_steps']==32 and cfg['sample_count']==4096
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);manifest_file=Path(cfg['data_root'])/'manifest.json'
    assert state['step']==a.endpoint_step and state['config_sha256']==sha(folder/'config.json') and state['dataset_manifest_sha256']==sha(manifest_file)
    with np.load(folder/'final_test_examples.npz',allow_pickle=False) as saved:generated=saved['generated'].copy();target=saved['target'].copy()
    digest=hashlib.sha256(target.tobytes()).hexdigest();target_hash=digest if target_hash is None else target_hash;assert digest==target_hash and len(target)==4096 and np.isfinite(generated).all()
    result=dict(cell=cell,dataset=ds,family=row['family'],validation_field_mse=row['replayed_field_mse'],primary_random_sw2=row['numpy_test_sw2'],checkpoint_sha256=row['checkpoint_sha256'],config_sha256=state['config_sha256'],manifest_sha256=state['dataset_manifest_sha256'])
    if 'relaxation' in row:result['relaxation']=row['relaxation']
    if ds=='rotated_product_bimodal_16d':
        spec=json.loads(manifest_file.read_text())['datasets'][ds]['distribution'];rotation=np.asarray(spec['rotation']);locations=np.asarray(spec['locations']);variances=np.asarray(spec['variances'])
        z=generated.astype(np.float64)@rotation;truth=target.astype(np.float64)@rotation
        axis=np.mean((np.sort(z,axis=0)-np.sort(truth,axis=0))**2,axis=0);gap=np.mean(np.abs(z)<locations[None,:]/2,axis=0)
        result.update(latent_axis_sw2=float(np.sqrt(axis.mean())),central_gap_mass=float(gap.mean()),near_mode_mse=float(np.mean((np.abs(z)-locations[None,:])**2)),axis_gap_mass=gap.tolist())
        if reference is None:
            population=[]
            for mean,var in zip(locations,variances):
                cdf=lambda v:.5*(1+math.erf(v/math.sqrt(2)))
                population.append(cdf(-mean/(2*math.sqrt(var)))-cdf(-3*mean/(2*math.sqrt(var))))
            reference=dict(locations=locations.tolist(),variances=variances.tolist(),population_gap_mass=population,target_gap_mass=np.mean(np.abs(truth)<locations[None,:]/2,axis=0).tolist(),target_near_mode_mse=float(np.mean((np.abs(truth)-locations[None,:])**2)))
        for j in [0,7,15]:
            limit=locations[j]+6*math.sqrt(variances[j]);edges=np.linspace(-limit,limit,61);counts,_=np.histogram(z[:,j],edges)
            examples.append(dict(cell=cell,family=row['family'],axis=j,edges=edges.tolist(),counts=counts.tolist(),sample_count=len(z),outside_fraction=float(np.mean(np.abs(z[:,j])>limit))))
    else:
        assert ds=='gmm_ring8_2d' and generated.shape==(4096,2)
        result['outside_plot_fraction']=float(np.mean(np.any(np.abs(generated)>4,axis=1)))
        examples.append(dict(cell=cell,family=row['family'],generated=generated[::4].tolist(),target=target[::4].tolist(),displayed_indices='0,4,...,4092;1024points;metrics use all4096'))
    rows.append(result);del state
out=Path(a.save);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(rows=rows,examples=examples,reference=reference,audit_sha256=sha(a.audit),analysis_source_sha256=sha(__file__),endpoint_step=a.endpoint_step,scope='CPU read-only registered final examples, no checkpoint/model selection or replacement primary metric.'),indent=2)+'\n');print(json.dumps(dict(rows=len(rows),saved=str(out))))
