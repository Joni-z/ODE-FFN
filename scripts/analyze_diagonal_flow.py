"""Audit the registered diagonal-flow family without modifying training artifacts."""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

DATASETS=('gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d')
TITLES=('Gaussian2D','Ring8','Product16')
ARMS=('ffn_silu','diagonal_static','diagonal_k1','diagonal_k2','diagonal_exp','swiglu_budget','swiglu')
LABELS=('SiLU128','Static K2','Adaptive K1','Adaptive K2','Adaptive exp','SwiGLU96','SwiGLU128')
COLORS=('#777777','#9e7b53','#d5983a','#bb4242','#923b8f','#7ba8bd','#255f88')
STEPS=tuple(range(0,5001,1000))
STATE_KEYS=('model','optimizer','runtime_buffers','data_rng','torch_rng','numpy_rng','python_rng')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def csv_write(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

def analyze(current,output):
    current,output=current.resolve(),output.resolve();source=current/'source'
    output.mkdir(parents=True,exist_ok=True);archived=output/'source';archived.mkdir(exist_ok=True)
    sys.path.insert(0,str(source));torch.set_num_threads(1)
    from synthetic.models import VelocityNet
    from scripts.run_architecture_study import macs
    hashes={};rows=[];curves=[];initials=[];diagonal=[];channels=[];raw_diagonal=[];projections=[];baselines=[];monitors=[];audits=[];replays=[];rng_checks=0
    def read(path):hashes[str(path)]=sha(path);return json.loads(path.read_text())
    def load(path):hashes[str(path)]=sha(path);return torch.load(path,map_location='cpu',weights_only=False)
    def folder(study,cfg,ds,kind):return study/'runs'/cfg['experiment_id']/ds/kind/'seed000'
    manifest=read(current/'source-manifest.json')
    for name,digest in manifest['all_files'].items():assert sha(source/name)==digest,name
    queue=read(current/'output/diagonal_flow_queue_20261024_m1/status.json');assert queue['state']=='COMPLETE'
    preflight=read(current/'preflight/summary.json');assert preflight['state']=='PASS'
    paired_preflight=read(current/'preflight/full-width-initialization.json');assert paired_preflight['state']=='PASS'
    shutil.copy2(__file__,archived/Path(__file__).name)
    replay_source=Path(__file__).resolve().parents[1]/'docs/research-architecture-search/analysis-shared-features/source/audit_architecture_endpoints.py'
    shutil.copy2(replay_source,archived/replay_source.name)
    old_roots={'ffn_silu':current.parent/'silu_reference_20261024_m1','swiglu_budget':current.parent/'shared_features_20261024_m1','swiglu':current.parent/'shared_features_20261024_m1'}
    for oldroot in set(old_roots.values()):
        oldmanifest=read(oldroot/'source-manifest.json')
        for name,digest in oldmanifest['all_files'].items():assert sha(oldroot/'source'/name)==digest,name
    for suffix in ('v1','product16'):
        study=current/'output'/f'diagonal_flow_{suffix}_20261024_m1'
        assert read(study/'status.json')['state']=='COMPLETE'
        data=Path('/Users/mr.z/ode-workspace-prep/data/synthetic')/('v1' if suffix=='v1' else 'product16_v1')
        commands=[('native',[source/'scripts/analyze_architecture_study.py','--study',study,'--source',source,'--data-root',data,'--output',output/f'{suffix}-native']),
            ('replay',[archived/replay_source.name,'--study',study,'--source',source,'--data-root',data,'--output',output/f'{suffix}-endpoint-replay.json'])]
        for label,args in commands:
            result=subprocess.run([sys.executable,*map(str,args)],cwd=source,capture_output=True,text=True)
            (output/f'{suffix}-{label}.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
        audit=read(output/f'{suffix}-native/audit.json');assert audit['state']=='PASS';audits.append(audit)
        replay=read(output/f'{suffix}-endpoint-replay.json');assert replay['state']=='PASS';replays.append(replay)
        rows.extend(dict(**r,study=str(study)) for r in audit['rows'])
        prov=read(study/'provenance.json');plan=prov['plan'];assert [a['label'] for a in plan['arms']]==list(ARMS)
        telemetry_path=study/'telemetry.jsonl';hashes[str(telemetry_path)]=sha(telemetry_path)
        history=[json.loads(line) for line in telemetry_path.read_text().splitlines()]
        assert history and all(r['memory_available_gib']>0 for r in history)
        monitors.append(dict(study=study.name,rows=len(history),first=history[0],last_actual_sample=history[-1],
            minimum_available_gib=min(r['memory_available_gib'] for r in history),terminal=read(study/'status.json')))
        for ds in plan['datasets']:
            dim=16 if ds.startswith('rotated') else 2;reference_rng={};base_initial=None;base_rng=None
            for arm in plan['arms']:
                label,kind=arm['label'],arm['model'];cfg=read(study/'configs'/f'{label}.json');run=folder(study,cfg,ds,kind)
                reference=cfg['backbone_initialization'][ds];assert sha(reference['checkpoint'])==reference['sha256'];ref=load(Path(reference['checkpoint']))
                cp_paths=sorted((run/'checkpoints').glob('step_*.pt'));assert [p.name for p in cp_paths]==[f'step_{s:07d}.pt' for s in STEPS]
                saved={step:load(path) for step,path in zip(STEPS,cp_paths)};initial=saved[0]
                torch.manual_seed(0);model=VelocityNet(dim,kind,**cfg['model_options']);native_rng=torch.get_rng_state().clone()
                common=[];ffn=[]
                for name,value in initial['model'].items():
                    if '.ffn.' in name:equal(value,model.state_dict()[name]);ffn.append(name)
                    else:equal(value,ref['model'][name]);common.append(name)
                equal(initial['torch_rng'],native_rng)
                equal(initial['runtime_buffers'],ref['runtime_buffers'])
                receipt=read(run/'provenance.json')['backbone_initialization']
                assert receipt['source_sha256']==reference['sha256'] and receipt['copied_state_keys']==sorted(common)
                assert receipt['copied_runtime_buffer_keys']==sorted(initial['runtime_buffers'])
                if label=='ffn_silu':base_initial=initial['model'];base_rng=initial['torch_rng']
                if kind=='ode_diagonal':
                    for name,value in base_initial.items():equal(value,initial['model'][name])
                    eta_keys=set(initial['model'])-set(base_initial);assert len(eta_keys)==6
                    assert all(name.endswith('.raw_eigenvalues') for name in eta_keys)
                    for name in eta_keys:assert torch.count_nonzero(initial['model'][name])==0
                    equal(initial['torch_rng'],base_rng)
                row=next(r for r in rows if r['dataset']==ds and r['label']==label)
                assert model.parameter_count()==row['parameters'] and macs(model,dim)==row['forward_macs']
                initials.append(dict(dataset=ds,label=label,reference=reference,non_ffn_tensors=len(common),native_ffn_entries=len(ffn),
                    native_ffn_rng_unchanged=True,SiLU_projection_bias_rng_paired=kind=='ode_diagonal',eta_zero=kind=='ode_diagonal'))
                old_saved={}
                if label in old_roots:
                    oldroot=old_roots[label];prefix='silu_reference' if label=='ffn_silu' else 'shared_features'
                    oldstudy=oldroot/'output'/f'{prefix}_{suffix}_20261024_m1';oldcfg=read(oldstudy/'configs'/f'{label}.json')
                    oldprov=read(oldstudy/'provenance.json');assert oldprov['dataset_manifest_sha256']==prov['dataset_manifest_sha256']
                    for key in ('steps','batch_size','learning_rate','weight_decay','grad_clip','eval_interval','sample_interval','checkpoint_interval','sample_count','sliced_directions','heun_steps','sampling_split','model_options','backbone_initialization'):
                        assert cfg[key]==oldcfg[key],(label,key)
                    for step in STEPS:old_saved[step]=load(folder(oldstudy,oldcfg,ds,kind)/'checkpoints'/f'step_{step:07d}.pt')
                for step,cp in saved.items():
                    if label=='ffn_silu':reference_rng[step]=cp['data_rng']
                    else:equal(cp['data_rng'],reference_rng[step]);rng_checks+=1
                    if label in old_roots:
                        for key in STATE_KEYS:equal(cp[key],old_saved[step][key])
                        baselines.append(dict(dataset=ds,label=label,step=step,state='BIT_IDENTICAL',compared_keys=list(STATE_KEYS)))
                    if label in ARMS[:5]:
                        for block in range(3):
                            prefix=f'blocks.{block}.ffn';e=cp['model'][prefix+'.net.0.weight'].double();r=cp['model'][prefix+'.net.2.weight'].double()
                            e0=initial['model'][prefix+'.net.0.weight'].double();r0=initial['model'][prefix+'.net.2.weight'].double()
                            en=float(torch.linalg.matrix_norm(e,ord=2));rn=float(torch.linalg.matrix_norm(r,ord=2))
                            effective_en=None;effective_rn=None
                            if label=='diagonal_static':
                                l1=cp['model'][prefix+'.ode1.raw_eigenvalues'].double().tanh();l2=cp['model'][prefix+'.ode2.raw_eigenvalues'].double().tanh()
                                d1=1+.6*l1+.5*(.6*l1).square();d2=1+.6*l2+.5*(.6*l2).square()
                                effective_en=float(torch.linalg.matrix_norm(d1[:,None]*e,ord=2));effective_rn=float(torch.linalg.matrix_norm(r*d2[None,:],ord=2))
                            projections.append(dict(dataset=ds,label=label,step=step,block=block,E_sigma_max=en,R_sigma_max=rn,
                                projection_norm_product=en*rn,E_relative_drift=float((e-e0).norm()/e0.norm()),R_relative_drift=float((r-r0).norm()/r0.norm()),
                                E_bias_norm=float(cp['model'][prefix+'.net.0.bias'].norm()),R_bias_norm=float(cp['model'][prefix+'.net.2.bias'].norm()),
                                static_absorbed_E_sigma_max=effective_en,static_absorbed_R_sigma_max=effective_rn))
                metric_path=run/'metrics.jsonl';hashes[str(metric_path)]=sha(metric_path)
                trace=[json.loads(line) for line in metric_path.read_text().splitlines()];assert [r['step'] for r in trace]==list(range(0,5001,500))
                for metric in trace:
                    step=metric['step'];curves.append(dict(dataset=ds,label=label,step=step,train_pair_mse=metric['train_pair_mse'],validation_pair_mse=metric['validation_pair_mse'],
                        validation_oracle_mse=metric['validation_oracle_mse'],wall_seconds=metric['wall_seconds'],grad_norm_before_clip=metric['grad_norm_before_clip'],
                        validation_sw2=metric.get('sample',{}).get('sliced_w2')))
                    if kind!='ode_diagonal':continue
                    assert len(metric['ode_layers'])==6
                    for entry in metric['ode_layers']:
                        name=entry['layer'];eta=np.array(entry['eigenvalue_coordinates']);lam=np.array(entry['P_eigenvalues']);act=entry['activation']
                        assert entry['operator_type']=='conditional_diagonal_flow' and entry['beta']==1.
                        np.testing.assert_allclose(lam,np.tanh(eta),atol=8e-8,rtol=2e-7)
                        assert np.max(np.abs(lam))==entry['P_sigma_max']
                        if step in saved:
                            equal(torch.tensor(eta,dtype=torch.float32),saved[step]['model'][name+'.raw_eigenvalues'])
                            equal(torch.tensor(lam,dtype=torch.float32),saved[step]['model'][name+'.raw_eigenvalues'].tanh())
                        raw_diagonal.append(dict(dataset=ds,label=label,step=step,**entry))
                        for channel,(a,l) in enumerate(zip(eta,lam)):channels.append(dict(dataset=ds,label=label,step=step,layer=name,channel=channel,eta=float(a),lambda_value=float(l)))
                        diagonal.append(dict(dataset=ds,label=label,step=step,layer=name,solver=entry['solver'],clock=entry['clock'],
                            eta_min=float(eta.min()),eta_max=float(eta.max()),eta_abs_max=float(np.max(np.abs(eta))),eta_norm=float(np.linalg.norm(eta)),
                            lambda_min=float(lam.min()),lambda_max=float(lam.max()),lambda_mean=float(lam.mean()),lambda_std=float(lam.std()),lambda_negative_fraction=float(np.mean(lam<0)),
                            P_sigma_max=entry['P_sigma_max'],conditional_G_min=entry['conditional_generator_eigen_min'],conditional_G_max=entry['conditional_generator_eigen_max'],
                            conditional_G_abs_max=max(abs(entry['conditional_generator_eigen_min']),abs(entry['conditional_generator_eigen_max'])),
                            duration_min=act['duration_min'],duration_max=act['duration_max'],duration_mean=act['duration_mean'],duration_std=act['duration_std'],
                            flow_derivative_min=act['derivative_min'],flow_derivative_max=act['derivative_max'],negative_derivative_fraction=act['negative_derivative_fraction'],
                            gain_min=act['gain_min'],gain_max=act['gain_max'],increment_rms_over_input=act['increment_rms_over_input']))
    assert len(rows)==21 and len(initials)==21 and len(baselines)==54 and rng_checks==108
    assert len(diagonal)==792 and len(channels)==101376 and len(projections)==270
    rows.sort(key=lambda r:(DATASETS.index(r['dataset']),ARMS.index(r['label'])))
    for name,data in [('comparison',rows),('learning-curves',curves),('diagonal-trajectories',diagonal),('eigenvalue-channels',channels),('projection-trajectories',projections)]:csv_write(output/(name+'.csv'),data)
    write(output/'raw-diagonal-trajectory.json',raw_diagonal)
    write(output/'initialization-audit.json',dict(state='PASS',initializations=initials,formal_projection_and_bias_pairing=True,eta_zero=True,snapshot_data_rng_comparisons=rng_checks,
        nonzero_head_function_proof='Formal initial E/R, biases, buffers and surrounding tensors equal the preflight-paired SiLU state. See the hash-bound full-width preflight proof.',
        full_width_preflight_sha256=sha(current/'preflight/full-width-initialization.json')))
    write(output/'baseline-trajectory-replay.json',dict(state='PASS',rows=baselines,scope='All six saved states per baseline trajectory match historical runs exactly; these are deterministic reruns, not independent repeated evidence.'))
    write(output/'monitoring-evidence.json',monitors)
    def get(ds,label):return next(r for r in rows if r['dataset']==ds and r['label']==label)
    metrics=('validation_oracle_mse','jacobian_error','validation_sw2');screens={}
    for label in ('diagonal_k2','diagonal_exp'):
        details={};leads=[]
        for ds in DATASETS:
            row=get(ds,label);best={m:min(get(ds,b)[m] for b in ARMS[-2:]) for m in metrics}
            ratio={m:row[m]/best[m] for m in metrics}
            kr=row['validation_oracle_mse']/get(ds,'diagonal_k1')['validation_oracle_mse']
            fr=row['validation_oracle_mse']/get(ds,'ffn_silu')['validation_oracle_mse'];sr=row['validation_oracle_mse']/get(ds,'diagonal_static')['validation_oracle_mse']
            if ds!=DATASETS[0] and ratio['validation_oracle_mse']<=.9 and kr<=.95 and fr<1 and sr<1:leads.append(ds)
            details[ds]=dict(ratio_to_better_swiglu=ratio,oracle_ratio_to_K1=kr,oracle_ratio_to_SiLU=fr,oracle_ratio_to_static=sr)
        primary=all(details[d]['ratio_to_better_swiglu']['validation_oracle_mse']<=1.05 for d in DATASETS)
        secondary=all(details[d]['ratio_to_better_swiglu'][m]<=1.1 for d in DATASETS[1:] for m in metrics[1:])
        screens[label]=dict(passes=bool(leads) and primary and secondary,qualifying_nonlinear_leads=leads,all_primary_within_5pct=primary,nonlinear_Jacobian_SW2_within_10pct=secondary,details=details)
    nominated=[label for label in screens if screens[label]['passes']]
    nominated.sort(key=lambda label:(get(DATASETS[2],label)['validation_oracle_mse'],get(DATASETS[2],label)['forward_macs'],get(DATASETS[1],label)['validation_oracle_mse']))
    write(output/'screen-decision.json',dict(state='NOMINATION' if nominated else 'NO_HIGHER_ORDER_CANDIDATE',nominated=nominated,comparisons=screens,
        scope='Exact registered engineering screen. K1 is a generic activation control, not a higher-order nominee. No beta/clock/width/order rescue.'))
    figures=output/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(3,3,figsize=(13,9.2),layout='constrained')
    for col,(ds,title) in enumerate(zip(DATASETS,TITLES)):
        for row,(metric,name) in enumerate(zip(metrics,('Oracle MSE','Jacobian error','Validation SW2'))):
            ax=axes[row,col];best=min(get(ds,b)[metric] for b in ARMS[-2:]);ax.bar(range(7),[get(ds,a)[metric]/best for a in ARMS],color=COLORS)
            ax.axhline(1,color='black',ls='--',lw=.8);ax.set_xticks(range(7),LABELS,rotation=45,ha='right',fontsize=7)
            ax.set(ylabel=name+' / better SwiGLU',title=title if row==0 else '');ax.grid(axis='y',alpha=.15)
    for ext in ('png','pdf'):fig.savefig(figures/f'diagonal-endpoints.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    for ax,ds,title in zip(axes,DATASETS,TITLES):
        for arm,label,color in zip(ARMS,LABELS,COLORS):
            trace=[r for r in curves if r['dataset']==ds and r['label']==arm]
            ax.plot([r['step'] for r in trace],[r['validation_oracle_mse'] for r in trace],label=label,color=color,lw=1.4)
        ax.set(title=title,xlabel='Training update',ylabel='Validation oracle MSE',yscale='log');ax.grid(alpha=.15)
    axes[-1].legend(fontsize=7,frameon=False)
    for ext in ('png','pdf'):fig.savefig(figures/f'diagonal-learning.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(13,9),layout='constrained')
    for col,(ds,title) in enumerate(zip(DATASETS,TITLES)):
        for row,(metric,name) in enumerate((('P_sigma_max','P sigma_max; dotted: conditional G'),('flow_derivative_min','Observed scalar-map minimum derivative'),('projection_norm_product','||E||2 * ||R||2'))):
            ax=axes[row,col];entries=projections if row==2 else diagonal
            arms=range(5) if row==2 else range(1,5)
            for i in arms:
                trace=[r for r in entries if r['dataset']==ds and r['label']==ARMS[i]];steps=sorted({r['step'] for r in trace})
                values=np.array([[r[metric] for r in trace if r['step']==s] for s in steps]);ax.plot(steps,values.mean(1),label=LABELS[i],color=COLORS[i])
                ax.fill_between(steps,values.min(1),values.max(1),color=COLORS[i],alpha=.08)
                if row==0:
                    g=np.array([[r['conditional_G_abs_max'] for r in trace if r['step']==s] for s in steps]);ax.plot(steps,g.mean(1),color=COLORS[i],ls=':',lw=1)
            ax.set(title=title if row==0 else '',xlabel='Training update',ylabel=name);ax.grid(alpha=.15)
    axes[0,-1].legend(fontsize=7,frameon=False);axes[2,-1].legend(fontsize=7,frameon=False)
    for ext in ('png','pdf'):fig.savefig(figures/f'diagonal-internals.{ext}',dpi=180)
    plt.close(fig)
    for name,digest in manifest['all_files'].items():assert sha(source/name)==digest,name
    write(output/'provenance.json',dict(state='PASS',analysis_sha256=sha(__file__),input_sha256=hashes,source_files_verified=len(manifest['all_files']),
        checkpoints_native_audited=sum(a['checkpoints_checked'] for a in audits),arrays_native_audited=sum(a['arrays_checked'] for a in audits),endpoint_replays=sum(a['final_checkpoints_reloaded'] for a in replays),
        maximum_forward_replay_difference=max(a['maximum_forward_difference'] for a in replays),maximum_jacobian_replay_difference=max(a['maximum_derivative_difference'] for a in replays),
        scope='Exposed validation evidence, recorded scalar-map probes and saved projection matrices. No fresh holdout or generated sampling-path replay; no full-network stability conclusion.'))
    return dict(state='PASS',rows=len(rows),curve_rows=len(curves),diagonal_rows=len(diagonal),eigenvalue_channel_rows=len(channels),projection_rows=len(projections),baseline_saved_states_equal=len(baselines),snapshot_rng_comparisons=rng_checks,nominated=nominated)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(analyze(args.current,args.output)))
