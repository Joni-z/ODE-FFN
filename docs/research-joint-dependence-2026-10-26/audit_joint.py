"""Reconstruct data and streams, then replay every native model and oracle record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time as timer

import numpy as np
import torch

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
ROOT=args.root.resolve();sys.path.insert(0,str(ROOT/'source'))
import joint_study as study
from joint_core import target_geometry

settings=study.setup(ROOT);torch.set_num_threads(1);started=timer.monotonic()
ready=study.load(ROOT/'ready.json');assert study.load(ROOT/'output/status.json')['state']=='COMPLETE'
for key in ('bindings','prior_documents'):
    for path,digest in ready[key].items():assert study.sha(path)==digest,path
manifest=study.load(ROOT/'data-manifest.json')
for name,digest in manifest['files'].items():assert study.sha(ROOT/name)==digest,name
target=study.distribution(ROOT);mu=target.locations;rotation=target.rotation
original=study.legacy.load_distribution(manifest['original_specification'])
np.testing.assert_array_equal(original.variances,np.full(16,.04))
means=target.means
# Recreate generation without calling the new distribution's sample method.
rng=np.random.default_rng(0)
def sample(count):
    index=rng.integers(32,size=count)
    latent=means[index]+.2*rng.standard_normal((count,16))
    return (latent@rotation.T).astype(np.float32)
train=sample(100000);validation=sample(20000);probe_target=sample(2048)
np.testing.assert_array_equal(train,np.load(ROOT/'data/train.npy'))
np.testing.assert_array_equal(validation,np.load(ROOT/'data/validation.npy'))
probe_source=rng.standard_normal((2048,16)).astype(np.float32);probe_t=rng.random((2048,1)).astype(np.float32)
probe=(1-probe_t)*probe_source+probe_t*probe_target
with np.load(ROOT/'banks/probe.npz') as data:
    np.testing.assert_array_equal(probe[:256],data['x']);np.testing.assert_array_equal(probe_t[:256],data['t'])

native_config=Path(settings['native_training_config']);old_banks=native_config.parent/'banks/rotated_product_bimodal_16d'
old_data=Path(study.load(native_config)['data_roots']['rotated_product_bimodal_16d'])/'rotated_product_bimodal_16d'
old_bank=study.legacy.load_targets(old_data/'train.npz',16)
reference={row['family']:row for row in study.reference_rows(ROOT)}
cohort=[];spectra=[];geometry=[];raw_geometry=[];counts=dict(checkpoints=0,panels=0,sampling_replays=0,matrix_spectra=0,stream_batches=0,oracle_rows=0)

def close(a,b):np.testing.assert_allclose(a,b,rtol=2e-6,atol=1e-9)


for law,banks,bank in [('product',old_banks,old_bank),('sign_code',ROOT/'banks',train)]:
    # Independent full consumed-stream reconstruction, including every snapshot boundary.
    data_rng=torch.Generator().manual_seed(100000);tensor=torch.from_numpy(bank)
    digest=study.EMPTY;digests={0:digest}
    for step in range(1,5001):
        a=tensor[torch.randint(len(tensor),(512,),generator=data_rng)]
        b=torch.randn(a.shape,generator=data_rng);t=torch.rand((512,1),generator=data_rng)
        digest=study.legacy.fingerprint_batch(digest,a,b,t)
        if step in study.SNAPSHOTS:digests[step]=digest
    counts['stream_batches']+=5000
    # Validate all saved oracle rows before using them as model evaluation targets.
    with np.load(banks/'field.npz') as data:
        field_x,field_t,field_oracle=data['x'],data['t'],data['oracle']
    if law=='sign_code':
        from synthetic.data import GaussianMixture
        enumerated=GaussianMixture(means,np.full_like(means,.04),np.ones(32))
        checked=enumerated.velocity(field_x@rotation,field_t)@rotation.T
    else:checked=original.velocity(field_x,field_t)
    np.testing.assert_allclose(checked,field_oracle,rtol=1e-6,atol=2e-6);counts['oracle_rows']+=len(field_x)
    for clock in study.TIMES:
        with np.load(banks/f'panel_{clock:.1f}.npz') as data:
            x,t,velocity,jacobian=[data[k] for k in ('x','t','oracle','oracle_jacobian')]
        if law=='sign_code':expected,jj,_=target.oracle(x,t,True)
        else:expected,jj=original.velocity(x,t),original.jacobian(x,t)
        np.testing.assert_allclose(expected,velocity,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(jj,jacobian,rtol=1e-9,atol=1e-10)
        values=target_geometry(jj,rotation)
        geometry.append(dict(law=law,time=clock,off_diagonal_mean=float(values['off_diagonal_energy_fraction'].mean()),
            off_diagonal_max=float(values['off_diagonal_energy_fraction'].max()),
            commutator_mean=float(values['normalized_pair_commutator'].mean()),
            commutator_max=float(values['normalized_pair_commutator'].max())))
        raw_geometry.append(dict(law=law,time=clock,**{k:v.tolist() for k,v in values.items()}))
    for family in study.FAMILIES:
        reused=law=='product';folder=Path(reference[family]['folder']) if reused else ROOT/'output/runs'/family
        assert study.load(folder/'status.json')['state']=='COMPLETE'
        if reused:
            reuse=study.load(ROOT/'output/reused'/f'{family}.json')
            for name,expected in reuse['files_sha256'].items():assert study.sha(folder/name)==expected,name
        result=study.load(folder/'result.json');records=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        for step in study.SNAPSHOTS:
            model=study.build(family);state=torch.load(folder/f'step_{step:07d}.pt',map_location='cpu',weights_only=False)
            study.restore_model(model,state);assert state['input_stream_digest']==digests[step]
            actual=study.legacy.field_evaluation(model,banks)
            row=next(r for r in records if r['step']==step)
            for key in ('oracle_mse','pair_mse','prediction_rms'):close(actual[key],row[key])
            if not reused:
                with torch.no_grad():prediction=np.concatenate([model(torch.from_numpy(field_x[k:k+1024]),torch.from_numpy(field_t[k:k+1024])).numpy() for k in range(0,len(field_x),1024)])
                with np.load(folder/f'field_{step:07d}.npz') as data:np.testing.assert_array_equal(prediction,data['prediction'])
                for entry in study.load(folder/f'spectrum_{step:07d}.json'):
                    p=state['model'][entry['layer']+'.proj.weight'].double().numpy()
                    np.testing.assert_allclose(np.linalg.svd(p,compute_uv=False),entry['singular_values'],rtol=1e-9,atol=1e-10)
                    np.testing.assert_allclose(np.linalg.eigvalsh(.5*(p+p.T)),entry['symmetric_eigenvalues'],rtol=1e-9,atol=1e-10)
                    spectra.append(dict(law=law,family=family,step=step,**entry));counts['matrix_spectra']+=1
            counts['checkpoints']+=1
        for key in ('oracle_mse','pair_mse','prediction_rms'):close(actual[key],result[key])
        for clock in study.TIMES:
            with np.load(banks/f'panel_{clock:.1f}.npz') as data:x,t,oracle,jj=[data[k] for k in ('x','t','oracle','oracle_jacobian')]
            value,jac=study.legacy.jacobian(model,torch.from_numpy(x),torch.from_numpy(t))
            with np.load(folder/'diagnostics'/f'panel_{clock:.1f}.npz') as data:
                np.testing.assert_array_equal(value.numpy(),data['prediction']);np.testing.assert_array_equal(jac.numpy(),data['jacobian'])
            row=next(r for r in result['panels'] if abs(r['time']-clock)<1e-6)
            close(float(np.mean((value.numpy()-oracle)**2)),row['field_mse'])
            close(float(np.mean(np.sum((jac.numpy()-jj)**2,axis=(1,2)))),row['jacobian_error']);counts['panels']+=1
        with np.load(banks/'sampling.npz') as data:source,sample_target,directions=[torch.from_numpy(data[k]) for k in ('source','target','directions')]
        samples=study.legacy.generated_samples(model,source,32);assert torch.isfinite(samples).all()
        with np.load(folder/'diagnostics/samples.npz') as data:np.testing.assert_array_equal(samples.numpy(),data['generated'])
        close(study.legacy.sliced_w2(samples,sample_target,directions),result['sw2']);counts['sampling_replays']+=1
        assert result['input_stream_digest']==digests[5000]
        cohort.append(dict(law=law,family=family,reused=reused,folder=str(folder),costs=study.compute_budget(model)))
    print(dict(law=law,audited_cells=5,elapsed_seconds=timer.monotonic()-started),flush=True)

for key in ('bindings','prior_documents'):
    for path,digest in ready[key].items():assert study.sha(path)==digest,path
study.write(ROOT/'audit-cohort.json',cohort);study.write(ROOT/'audit-spectra.json',spectra)
study.write(ROOT/'target-geometry.json',geometry);study.write(ROOT/'target-geometry-values.json',raw_geometry)
receipt=dict(status='PASS',new_cells=5,reused_cells=5,**counts,input_bindings=len(ready['bindings']),
    preserved_prior_files=len(ready['prior_documents']),source_sha256=study.sha(Path(__file__)),
    cohort_sha256=study.sha(ROOT/'audit-cohort.json'),geometry_sha256=study.sha(ROOT/'target-geometry.json'),elapsed_seconds=timer.monotonic()-started)
study.write(ROOT/'audit.json',receipt);print(json.dumps(receipt),flush=True)
