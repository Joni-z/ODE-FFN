"""Read-only Stage17 checkpoint, pairing, cost, and registered-screen audit."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

DATASETS=('gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d')
TITLES=('Gaussian2D','Ring8','Product16')
ARMS=('vector_parent','vector_shared','vector_untied64','swiglu_budget','swiglu','swiglu_large')
LABELS=('ODE parent32','ODE shared64','ODE untied64','SwiGLU96','SwiGLU128','SwiGLU192')
COLORS=('#73777c','#bd413e','#dc963c','#77a3b8','#29628b','#403c75')
STEPS=tuple(range(0,5001,1000))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def csv_write(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(current:Path,output:Path):
    torch.set_num_threads(1)
    current=current.resolve();output=output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    source=current/'source';frozen=output/'source';frozen.mkdir(exist_ok=True)
    sys.path.insert(0,str(source))
    from synthetic.models import VelocityNet
    from synthetic.train import restore_model
    from scripts.check_gate_geometry import jacobian
    from scripts.run_architecture_study import macs
    hashes={};rows=[];curves=[];geometry=[];initials=[];alias_checks=[];functional=[];telemetry=[];rng_comparisons=0
    def record(path):hashes[str(path)]=sha(path)
    def read(path):record(path);return json.loads(path.read_text())
    def load(path):record(path);return torch.load(path,map_location='cpu',weights_only=False)
    def equal(a,b):torch.testing.assert_close(a,b,atol=0,rtol=0)
    def runfolder(study,cfg,dataset,kind):return study/'runs'/cfg['experiment_id']/dataset/kind/'seed000'
    queue=read(current/'output/shared_features_queue_20261024_m1/status.json')
    assert queue['state']=='COMPLETE' and queue['completed']==queue['total']==2
    manifest=read(current/'source-manifest.json')
    for name,digest in manifest['all_files'].items():assert sha(source/name)==digest,name
    shutil.copy2(Path(__file__),frozen/Path(__file__).name)
    replay_source=Path(__file__).parent/'audit_architecture_endpoints.py'
    shutil.copy2(replay_source,frozen/replay_source.name);record(replay_source)
    native_audits=[];endpoint_replays=[]
    for suffix in ('v1','product16'):
        study=current/'output'/f'shared_features_{suffix}_20261024_m1'
        status=read(study/'status.json');assert status['state']=='COMPLETE'
        data_root=Path('/Users/mr.z/ode-workspace-prep/data/synthetic')/('v1' if suffix=='v1' else 'product16_v1')
        commands=[('native',[str(source/'scripts/analyze_architecture_study.py'),'--study',str(study),'--source',str(source),'--data-root',str(data_root),'--output',str(output/f'{suffix}-native')]),
                  ('replay',[str(frozen/replay_source.name),'--study',str(study),'--source',str(source),'--data-root',str(data_root),'--output',str(output/f'{suffix}-endpoint-replay.json')])]
        for label,args in commands:
            result=subprocess.run([sys.executable,*args],cwd=source,capture_output=True,text=True)
            (output/f'{suffix}-{label}.log').write_text(result.stdout+result.stderr)
            assert result.returncode==0,(suffix,label,result.stderr)
        audit=read(output/f'{suffix}-native/audit.json');assert audit['state']=='PASS';native_audits.append(audit)
        replay=read(output/f'{suffix}-endpoint-replay.json');assert replay['state']=='PASS';endpoint_replays.append(replay)
        prov=read(study/'provenance.json');plan=prov['plan']
        assert [arm['label'] for arm in plan['arms']]==list(ARMS)
        assert (plan['seed'],plan['steps'],plan['workers'],plan['device'])==(0,5000,2,'cpu')
        log=study/'telemetry.jsonl';record(log);history=[json.loads(line) for line in log.read_text().splitlines()]
        assert history and all(r['memory_available_gib']>0 for r in history)
        telemetry.append(dict(study=study.name,records=len(history),first=history[0],last_actual_sample=history[-1],
            minimum_available_gib=min(r['memory_available_gib'] for r in history),terminal_monitor=read(study/'monitor.json')))
        rows.extend(dict(**row,study=str(study)) for row in audit['rows'])
        for dataset in plan['datasets']:
            dim=16 if dataset.startswith('rotated') else 2
            reference_rng={};initial_states={};initial_models={}
            for arm in plan['arms']:
                label,kind=arm['label'],arm['model']
                cfg_path=study/'configs'/f'{label}.json';cfg=read(cfg_path);folder=runfolder(study,cfg,dataset,kind)
                reference=cfg['backbone_initialization'][dataset]
                assert sha(reference['checkpoint'])==reference['sha256']
                parent_initial=load(Path(reference['checkpoint']))
                metric_path=folder/'metrics.jsonl';record(metric_path)
                trace=[json.loads(line) for line in metric_path.read_text().splitlines()]
                assert [m['step'] for m in trace]==list(range(0,5001,500))
                for metric in trace:
                    curves.append(dict(dataset=dataset,label=label,step=metric['step'],train_pair_mse=metric['train_pair_mse'],
                        validation_pair_mse=metric['validation_pair_mse'],validation_oracle_mse=metric['validation_oracle_mse'],
                        wall_seconds=metric['wall_seconds'],grad_norm_before_clip=metric['grad_norm_before_clip']))
                checkpoints=sorted((folder/'checkpoints').glob('step_*.pt'))
                assert [p.name for p in checkpoints]==[f'step_{s:07d}.pt' for s in STEPS]
                saved={s:load(p) for s,p in zip(STEPS,checkpoints)}
                initial=saved[0];initial_states[label]=initial
                torch.manual_seed(0);native=VelocityNet(dim,kind,**cfg['model_options'])
                native_rng=torch.get_rng_state().clone()
                copied=[];native_ffn=[]
                for name,value in initial['model'].items():
                    if '.ffn.' in name:equal(value,native.state_dict()[name]);native_ffn.append(name)
                    else:equal(value,parent_initial['model'][name]);copied.append(name)
                equal(initial['torch_rng'],native_rng)
                for name,value in initial['runtime_buffers'].items():equal(value,parent_initial['runtime_buffers'][name])
                run_prov=read(folder/'provenance.json');receipt=run_prov['backbone_initialization']
                assert receipt['source_sha256']==reference['sha256']
                assert receipt['copied_state_keys']==sorted(copied)
                assert receipt['copied_runtime_buffer_keys']==sorted(initial['runtime_buffers'])
                assert native.parameter_count()==next(r['parameters'] for r in rows if r['dataset']==dataset and r['label']==label)
                assert macs(native,dim)==next(r['forward_macs'] for r in rows if r['dataset']==dataset and r['label']==label)
                for step,cp in saved.items():
                    if label==ARMS[0]:reference_rng[step]=cp['data_rng']
                    else:equal(cp['data_rng'],reference_rng[step]);rng_comparisons+=1
                    restored=VelocityNet(dim,kind,**cfg['model_options']).double().eval();restore_model(restored,cp)
                    if step==0:initial_models[label]=restored
                    if kind=='ode_vector':
                        layer_count=0
                        for name,layer in restored.ode_layers():
                            is_shared=layer.t_func[0].weight is layer.proj.weight
                            assert is_shared==(label=='vector_shared')
                            assert len(cp['optimizer']['param_groups'][0]['params'])==len(list(restored.parameters()))
                            if is_shared:equal(cp['model'][name+'.t_func.0.weight'],cp['model'][name+'.proj.weight'])
                            if step==0:
                                equal(layer.proj.weight,parent_initial['model'][name+'.proj.weight'].double())
                                assert torch.count_nonzero(layer.t_func[-1].weight)==0
                                assert torch.count_nonzero(layer.t_func[-1].bias)==0
                                if label!='vector_parent':
                                    equal(layer.t_func[0].weight,layer.proj.weight)
                                    assert torch.count_nonzero(layer.t_func[0].bias)==0
                            layer_count+=1
                        alias_checks.append(dict(dataset=dataset,label=label,step=step,layers=layer_count,
                            shared_alias_expected=label=='vector_shared',all_aliases_correct=True,optimizer_deduplicated=True))
                initials.append(dict(dataset=dataset,label=label,reference=reference,non_ffn_tensors=len(copied),
                    native_ffn_state_entries=len(native_ffn),buffers=sorted(initial['runtime_buffers']),native_ffn_and_torch_rng_unchanged=True))
                if kind=='ode_vector':
                    for metric in trace:
                        for layer in metric['ode_layers']:
                            name=layer['layer'];step=metric['step'];activation=layer['activation'];raw=layer['raw']
                            drift=None;separation=None
                            if step in saved:
                                p=saved[step]['model'][name+'.proj.weight'].double();p0=saved[0]['model'][name+'.proj.weight'].double()
                                spectrum=torch.linalg.svdvals(p)
                                np.testing.assert_allclose(float(spectrum[0]),raw['sigma_max'],atol=1e-10,rtol=1e-10)
                                np.testing.assert_allclose(float(p.square().sum()/spectrum[0].square()),raw['stable_rank'],atol=1e-10,rtol=1e-10)
                                drift=float((p-p0).norm()/p0.norm())
                                if label!='vector_parent':
                                    r=saved[step]['model'][name+'.t_func.0.weight'].double()
                                    separation=float((r-p).norm()/p.norm())
                            geometry.append(dict(dataset=dataset,label=label,step=step,layer=name,sigma_max=raw['sigma_max'],
                                spectral_radius=raw['spectral_radius'],stable_rank=raw['stable_rank'],nonnormality=raw['nonnormality'],
                                eig_real_max=raw['eig_real_max'],eig_real_min=raw['eig_real_min'],
                                s_mean=activation['s_mean'],s_std=activation['s_std'],s_min=activation['s_min'],s_max=activation['s_max'],
                                logit_sensitivity_relative_initial=activation['clock_logit_sensitivity_over_initial'],
                                outer_saturated_fraction=activation['clock_outer_saturated_fraction'],
                                term1_rms_over_input=activation['term1_rms_over_input'],
                                P_relative_drift=drift,R_P_relative_separation=separation))
            # Shared and untied feature tensors match as complete saved states at step0.
            a,b=initial_states['vector_shared']['model'],initial_states['vector_untied64']['model']
            assert set(a)==set(b)
            for name,value in a.items():equal(value,b[name])
            # A common nonzero output head avoids the trivial zero-velocity test.
            generator=torch.Generator().manual_seed(0)
            x=torch.randn(19,dim,generator=generator,dtype=torch.float64)
            t=torch.rand(19,1,generator=generator,dtype=torch.float64)
            head=torch.randn(dim,64,generator=generator,dtype=torch.float64)*.2
            expected=None;expected_j=None
            for label in ARMS[:3]:
                model=initial_models[label]
                with torch.no_grad():model.output.weight.copy_(head)
                value,derivative=jacobian(model,x,t)
                if expected is None:expected=value;expected_j=derivative
                equal(value,expected)
                torch.testing.assert_close(derivative,expected_j,atol=2e-13,rtol=2e-13)
                functional.append(dict(dataset=dataset,label=label,nonzero_head_forward_max_error=float((value-expected).abs().max()),
                    input_jacobian_max_error=float((derivative-expected_j).abs().max()),declared_jacobian_atol=2e-13,declared_jacobian_rtol=2e-13))
    assert len(rows)==18 and len(initials)==18 and len(alias_checks)==54 and rng_comparisons==90
    rows.sort(key=lambda r:(DATASETS.index(r['dataset']),ARMS.index(r['label'])))
    csv_write(output/'comparison.csv',rows);csv_write(output/'learning-curves.csv',curves);csv_write(output/'P-trajectories.csv',geometry)
    write(output/'initialization-and-alias-audit.json',dict(state='PASS',initial_checkpoints=initials,checkpoint_alias_checks=alias_checks,
        shared_untied_initial_complete_states_equal=True,initial_function_checks=functional,
        snapshot_data_rng_comparisons=rng_comparisons,scope='Each of five other arms compared to parent at every one of six snapshots on each task; all six arms have the same data RNG.'))
    write(output/'monitoring-evidence.json',telemetry)
    def get(dataset,label):return next(r for r in rows if r['dataset']==dataset and r['label']==label)
    screens={}
    metrics=('validation_oracle_mse','jacobian_error','validation_sw2')
    for label,baselines in [('vector_shared',ARMS[3:5]),('vector_untied64',ARMS[3:])]:
        details={};leads=[]
        for dataset in DATASETS:
            item=get(dataset,label);parent=get(dataset,'vector_parent')
            best={m:min(get(dataset,b)[m] for b in baselines) for m in metrics}
            ratio={m:item[m]/best[m] for m in metrics}
            parent_ratio=item['validation_oracle_mse']/parent['validation_oracle_mse']
            if dataset!=DATASETS[0] and ratio['validation_oracle_mse']<=.9 and parent_ratio<1:leads.append(dataset)
            details[dataset]=dict(ratio_to_better_registered_baseline=ratio,parent_oracle_ratio=parent_ratio)
        primary=all(details[d]['ratio_to_better_registered_baseline']['validation_oracle_mse']<=1.05 for d in DATASETS)
        secondary=all(details[d]['ratio_to_better_registered_baseline'][m]<=1.1 for d in DATASETS[1:] for m in metrics[1:])
        screens[label]=dict(passes=bool(leads) and primary and secondary,baselines=list(baselines),nonlinear_leads=leads,
            all_primary_within_5pct=primary,nonlinear_Jacobian_and_SW2_within_10pct=secondary,details=details)
    nominated=[label for label,result in screens.items() if result['passes']]
    write(output/'screen-decision.json',dict(state='NOMINATION' if nominated else 'NO_CANDIDATE',nominated=nominated,comparisons=screens,
        scope='Unchanged registered engineering screen. Nomination is not architecture success; no new width, temperature, order, or partial sharing rescue.'))
    figures=output/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(3,3,figsize=(12.5,9),layout='constrained')
    for column,(dataset,title) in enumerate(zip(DATASETS,TITLES)):
        for row,(metric,name) in enumerate(zip(metrics,('Oracle MSE','Jacobian error','Validation SW2'))):
            ax=axes[row,column];best=min(get(dataset,b)[metric] for b in ARMS[3:5])
            ax.bar(range(6),[get(dataset,a)[metric]/best for a in ARMS],color=COLORS)
            ax.axhline(1,color='black',ls='--',lw=.8)
            ax.set_xticks(range(6),LABELS,rotation=45,ha='right',fontsize=7)
            ax.set(ylabel=name+' / better SwiGLU96,128',title=title if row==0 else '')
            ax.grid(axis='y',alpha=.15)
    for ext in ('png','pdf'):fig.savefig(figures/f'shared-endpoints.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.2),layout='constrained')
    for ax,dataset,title in zip(axes,DATASETS,TITLES):
        for arm,label,color in zip(ARMS,LABELS,COLORS):
            trace=[r for r in curves if r['dataset']==dataset and r['label']==arm]
            ax.plot([r['step'] for r in trace],[r['validation_oracle_mse'] for r in trace],label=label,color=color,lw=1.5)
        ax.set(title=title,xlabel='Training update',ylabel='Validation oracle MSE',yscale='log');ax.grid(alpha=.15)
    axes[-1].legend(fontsize=8,frameon=False)
    for ext in ('png','pdf'):fig.savefig(figures/f'shared-learning.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12.5,9),layout='constrained')
    for column,(dataset,title) in enumerate(zip(DATASETS,TITLES)):
        for row,(metric,name) in enumerate((('sigma_max','P largest singular value'),('P_relative_drift','||P-P0||F / ||P0||F'),('logit_sensitivity_relative_initial','Clock slope / initial slope'))):
            ax=axes[row,column]
            for arm,label,color in zip(ARMS[:3],LABELS[:3],COLORS[:3]):
                trace=[r for r in geometry if r['dataset']==dataset and r['label']==arm and r[metric] is not None]
                steps=sorted({r['step'] for r in trace});values=np.array([[r[metric] for r in trace if r['step']==step] for step in steps])
                ax.plot(steps,values.mean(1),label=label,color=color,lw=1.5)
                ax.fill_between(steps,values.min(1),values.max(1),color=color,alpha=.1)
            ax.set(title=title if row==0 else '',xlabel='Training update',ylabel=name);ax.grid(alpha=.15)
    axes[0,-1].legend(fontsize=8,frameon=False)
    for ext in ('png','pdf'):fig.savefig(figures/f'shared-P-trajectories.{ext}',dpi=180)
    plt.close(fig)
    write(output/'provenance.json',dict(state='PASS',analysis_sha256=sha(__file__),input_sha256=hashes,
        source_files_verified=len(manifest['all_files']),checkpoints_native_audited=sum(a['checkpoints_checked'] for a in native_audits),
        diagnostic_arrays_native_audited=sum(a['arrays_checked'] for a in native_audits),
        endpoint_replays=sum(a['final_checkpoints_reloaded'] for a in endpoint_replays),
        maximum_forward_replay_difference=max(a['maximum_forward_difference'] for a in endpoint_replays),
        maximum_derivative_replay_difference=max(a['maximum_derivative_difference'] for a in endpoint_replays),
        scope='Existing validation endpoints and training snapshots. No new training, fresh holdout, or generated-sampling trajectory replay.'))
    return dict(state='PASS',rows=len(rows),learning_rows=len(curves),P_trajectory_rows=len(geometry),snapshot_rng_comparisons=rng_comparisons,nominated=nominated)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(analyze(args.current,args.output)))
