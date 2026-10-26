"""Verify frozen source, checkpoints and common-point oracle evidence."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.train import fixed_eval_data,load_distribution,restore_model,sha256
from synthetic.models import VelocityNet
from scripts.check_gate_geometry import oracle_geometry


def analyze(study: Path,source: Path,data_root: Path,output: Path) -> dict:
    torch.set_num_threads(1)
    status=json.loads((study/'status.json').read_text())
    assert status['state']=='COMPLETE'
    provenance=json.loads((study/'provenance.json').read_text())
    plan=provenance['plan']
    assert status['completed']==status['total']==len(plan['arms'])*len(plan['datasets'])
    for name,digest in provenance['source_sha256'].items():
        assert sha256(source/name.replace('\\','/'))==digest,name
    assert sha256(data_root/'manifest.json')==provenance['dataset_manifest_sha256']
    manifest=json.loads((data_root/'manifest.json').read_text())
    reported=json.loads((study/'results.json').read_text())
    rows=[]
    n_arrays=n_checkpoints=0
    stream_replays=0
    for dataset in plan['datasets']:
        common={}
        initials={}
        population_streams={}
        input_rng_states={}
        dist=load_distribution(manifest['datasets'][dataset])
        for arm in plan['arms']:
            label,kind=arm['label'],arm['model']
            cfg_path=study/'configs'/f'{label}.json'
            cfg=json.loads(cfg_path.read_text())
            exp=cfg['experiment_id']
            folder=study/'runs'/exp/dataset/kind/'seed000'
            assert (folder/'COMPLETE').read_text().strip()==str(plan['steps'])
            assert sha256(cfg_path)==sha256(folder/'config.json')
            if 'resume_study' in plan:
                resume=json.loads((folder/'resumes.jsonl').read_text().splitlines()[-1])
                continuation=json.loads((folder/'continuation-input.json').read_text())
                assert resume['from_step']==continuation['from_step']
                assert resume['requested_total_steps']==plan['steps']
                assert sha256(folder/f'checkpoints/step_{resume["from_step"]:07d}.pt')==continuation['checkpoint_sha256']
                for name,digest in resume['provenance']['source_sha256'].items():
                    assert sha256(source/'synthetic'/name)==digest,name
            snapshots=sorted((folder/'checkpoints').glob('step_*.pt'))
            assert [p.name for p in snapshots]==[f'step_{i:07d}.pt' for i in range(0,plan['steps']+1,1000)]
            stream_checkpoints=[]
            rng_checkpoints=[]
            for p in snapshots:
                cp=torch.load(p,map_location='cpu',weights_only=False)
                assert cp['identity']==dict(dataset=dataset,model=kind,seed=0)
                assert cp['config_sha256']==sha256(cfg_path)
                assert cp['dataset_manifest_sha256']==provenance['dataset_manifest_sha256']
                if cp['step']==0:initials[label]=cp['model']
                rng_checkpoints.append(cp['data_rng'])
                if cfg.get('training_targets','bank_pair')!='bank_pair':
                    stream=cp['training_stream']
                    assert stream['mode']==cfg['training_targets']
                    assert stream['batches']==cp['step']
                    stream_checkpoints.append((cp['step'],stream['digest'],cp['data_rng']))
                n_checkpoints+=1
            if stream_checkpoints:population_streams[label]=(cfg['batch_size'],stream_checkpoints)
            input_rng_states[label]=rng_checkpoints
            diagnostic=study/'diagnostics'/dataset/label
            summary=json.loads((diagnostic/'summary.json').read_text())
            assert summary['checkpoint_sha256']==sha256(folder/f'checkpoints/step_{plan["steps"]:07d}.pt')
            errors=[]
            for name,digest in summary['arrays_sha256'].items():
                path=diagnostic/name
                assert sha256(path)==digest
                time=float(name.removeprefix('common-').removesuffix('.npz'))
                with np.load(path,allow_pickle=False) as a:
                    assert all(np.isfinite(a[k]).all() for k in a.files)
                    if name in common:np.testing.assert_array_equal(a['x'],common[name])
                    else:common[name]=a['x'].copy()
                    truth=oracle_geometry(dist,a['x'],time)
                    np.testing.assert_allclose(a['oracle'],truth['oracle'],rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(a['oracle_jacobian'],truth['oracle_jacobian'],rtol=1e-12,atol=1e-12)
                    errors.append([np.mean((a['prediction']-a['oracle'])**2),
                        np.mean(np.sum((a['jacobian']-a['oracle_jacobian'])**2,axis=(1,2)))])
                n_arrays+=1
            measured=np.mean(errors,axis=0)
            row=next(r for r in reported if r['dataset']==dataset and r['label']==label)
            np.testing.assert_allclose(measured,[row['field_mse'],row['jacobian_error']],rtol=1e-12)
            assert row['step']==plan['steps']
            model=VelocityNet(dist.dim,kind,**cfg['model_options']).eval()
            restore_model(model,cp)
            fixed=fixed_eval_data(data_root/dataset,dist,dist.dim,0,'cpu',
                cfg['sample_count'],cfg['sliced_directions'],cfg['sampling_split'])
            error=0.
            with torch.no_grad():
                for left in range(0,len(fixed['x']),1024):
                    value=model(fixed['x'][left:left+1024],fixed['t'][left:left+1024])
                    error+=float((value-fixed['oracle'][left:left+1024]).square().sum())
            audited=error/fixed['x'].numel()
            np.testing.assert_allclose(audited,row['validation_oracle_mse'],rtol=2e-5,atol=1e-8)
            row['audited_validation_oracle_mse']=audited
            rows.append(row)
        if population_streams:
            from synthetic.population import PopulationStream,fingerprint_batch
            import hashlib
            assert plan['device']=='cpu','Exact fresh-stream audit currently covers CPU studies'
            reference=next(iter(population_streams.values()))
            for label,(batch_size,checkpoints) in population_streams.items():
                assert batch_size==reference[0]
                assert [(s,d) for s,d,_ in checkpoints]==[(s,d) for s,d,_ in reference[1]]
                for actual,expected in zip(checkpoints,reference[1]):
                    torch.testing.assert_close(actual[2],expected[2],atol=0,rtol=0)
            checkpoint_map={step:(digest,state) for step,digest,state in reference[1]}
            generator=torch.Generator().manual_seed(plan['seed']+100000)
            population=PopulationStream(dist)
            digest=hashlib.sha256(b'').hexdigest()
            assert checkpoint_map[0][0]==digest
            torch.testing.assert_close(generator.get_state(),checkpoint_map[0][1],atol=0,rtol=0)
            for step in range(1,plan['steps']+1):
                target=population.sample(reference[0],generator)
                source=torch.randn(target.shape,generator=generator)
                time=torch.rand((len(target),1),generator=generator)
                digest=fingerprint_batch(digest,target,source,time)
                if step in checkpoint_map:
                    assert digest==checkpoint_map[step][0],step
                    torch.testing.assert_close(generator.get_state(),checkpoint_map[step][1],atol=0,rtol=0)
            for family in ('vector','swiglu_budget','swiglu'):
                a,b=initials[family+'_pair'],initials[family+'_mean']
                assert a.keys()==b.keys()
                for name,value in a.items():torch.testing.assert_close(value,b[name],atol=0,rtol=0)
            stream_replays+=1
        if 'scalar_k1' in initials:
            reference=initials['scalar_k1']
            for label in ('scalar_k2','vector_k1','vector_k2'):
                if label not in initials:
                    continue
                for name,value in reference.items():
                    if '.t_func.' not in name:
                        torch.testing.assert_close(value,initials[label][name],atol=0,rtol=0)
        for label,parent in [('scalar_identity','scalar_k1'),('vector_identity','vector_k1')]:
            if label in initials:
                for name,value in initials[parent].items():
                    torch.testing.assert_close(value,initials[label][name],atol=0,rtol=0)
        for label in ('vector_tied','vector_reuse'):
            if label in initials:
                for name,value in initials[label].items():
                    if '.t_func.' not in name or '.ode1.t_func.' in name:
                        torch.testing.assert_close(value,initials['vector_k1'][name],atol=0,rtol=0)
        for label in ('vector_wide','vector_affine'):
            if label in initials:
                for name,value in initials['vector_k1'].items():
                    torch.testing.assert_close(value,initials[label][name],atol=0,rtol=0)
        if 'vector_reparam' in initials:
            for name,value in initials['vector_k1'].items():
                actual=initials['vector_reparam'][name]
                torch.testing.assert_close(value,actual*8 if '.t_func.2.' in name else actual,atol=0,rtol=0)
                torch.testing.assert_close(value,initials['vector_tau'][name],atol=0,rtol=0)
            for label,states in input_rng_states.items():
                for actual,expected in zip(states,input_rng_states['vector_k1']):
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        if 'vector_before' in initials:
            for name,value in initials['vector_k1'].items():
                torch.testing.assert_close(value,initials['vector_before'][name],atol=0,rtol=0)
            for label,states in input_rng_states.items():
                for actual,expected in zip(states,input_rng_states['vector_k1']):
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        if 'vector_increment' in initials:
            for label,parent in [('scalar_increment','scalar_k1'),('vector_increment','vector_k1')]:
                for name,value in initials[parent].items():
                    torch.testing.assert_close(value,initials[label][name],atol=0,rtol=0)
            for label in ('vector_const_map','vector_const_increment'):
                for name,value in initials['vector_k1'].items():
                    expected=value
                    if '.t_func.2.' in name:expected=torch.zeros_like(value)
                    elif label=='vector_const_increment' and name.endswith('.proj.weight'):
                        cfg=json.loads((study/'configs'/f'{label}.json').read_text())
                        options=cfg['model_options'];center=options['shift']+options['scale']/2
                        expected=value+torch.eye(len(value),dtype=value.dtype)/center
                    torch.testing.assert_close(expected,initials[label][name],atol=0,rtol=0)
            for label,states in input_rng_states.items():
                for actual,expected in zip(states,input_rng_states['vector_k1']):
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        if 'vector_mi' in initials:
            for family in ('scalar','vector'):
                reference=initials[family+'_mm']
                for pattern in ('mm','mi','im','ii'):
                    cfg=json.loads((study/'configs'/f'{family}_{pattern}.json').read_text())
                    center=cfg['model_options']['shift']+cfg['model_options']['scale']/2
                    for name,value in reference.items():
                        expected=value
                        if '.t_func.2.' in name:
                            torch.testing.assert_close(value,torch.zeros_like(value),atol=0,rtol=0)
                        elif name.endswith('.proj.weight'):
                            position=0 if '.ode1.' in name else 1
                            if pattern[position]=='i':expected=value+torch.eye(len(value),dtype=value.dtype)/center
                        torch.testing.assert_close(expected,initials[family+'_'+pattern][name],atol=0,rtol=0)
            for name,value in initials['scalar_mm'].items():
                if '.t_func.' not in name:
                    torch.testing.assert_close(value,initials['vector_mm'][name],atol=0,rtol=0)
            for label,states in input_rng_states.items():
                for actual,expected in zip(states,input_rng_states['vector_mm']):
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        if 'vector_add' in initials:
            for family in ('scalar','vector'):
                for name,value in initials[family+'_mul'].items():
                    torch.testing.assert_close(value,initials[family+'_add'][name],atol=0,rtol=0)
                    if '.t_func.2.' in name:
                        torch.testing.assert_close(value,torch.zeros_like(value),atol=0,rtol=0)
            for name,value in initials['scalar_mul'].items():
                if '.t_func.' not in name:
                    torch.testing.assert_close(value,initials['vector_mul'][name],atol=0,rtol=0)
            for label,states in input_rng_states.items():
                for actual,expected in zip(states,input_rng_states['vector_mul']):
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        if 'cov_tied' in initials:
            for name,value in initials['cov_tied'].items():
                for label in ('cov_untied','cov_linear'):
                    torch.testing.assert_close(value,initials[label][name],atol=0,rtol=0)
                if '.field.' not in name:
                    torch.testing.assert_close(value,initials['vector_k1'][name],atol=0,rtol=0)
                elif name.endswith('.field.weight'):
                    original=name.replace('.field.weight','.proj.weight')
                    torch.testing.assert_close(value,initials['vector_k1'][original][:len(value)],atol=0,rtol=0)
                    decoder=name.removesuffix('weight')+'decoder'
                    torch.testing.assert_close(value.t(),initials['cov_untied'][decoder],atol=0,rtol=0)
        if 'drift_increment' in initials:
            for label in ('drift_map','drift_increment','drift_fixed'):
                for name,value in initials[label].items():
                    mapped=name.replace('.ffn.proj1.','.ffn.ode1.proj.').replace('.ffn.proj2.','.ffn.ode2.proj.').replace('.ffn.t_func.','.ffn.ode1.t_func.')
                    torch.testing.assert_close(value,initials['linear_reuse'][mapped],atol=0,rtol=0)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    audit=dict(state='PASS',status=status,arrays_checked=n_arrays,checkpoints_checked=n_checkpoints,
        source_verified=True,oracles_recomputed=True,common_inputs_match=True,
        rows=rows,plan=plan,
        population_stream_regenerated=stream_replays>0,
        population_stream_replay_batches=stream_replays*plan['steps'])
    (output/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','data-root','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    result=analyze(args.study,args.source,args.data_root,args.output)
    print(json.dumps(result,indent=2))
