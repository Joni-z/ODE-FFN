"""Audit the unchanged plain-SiLU reference and its immutable Stage17 controls."""
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
OLD=('vector_parent','vector_shared','vector_untied64','swiglu_budget','swiglu','swiglu_large')
ARMS=(*OLD,'ffn_silu')
LABELS=('ODE parent32','ODE shared64','ODE untied64','SwiGLU96','SwiGLU128','SwiGLU192','SiLU FFN128')
COLORS=('#777777','#c95e58','#cc9846','#8eb5c7','#28638c','#53417b','#19834e')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def csv_write(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def analyze(prior,current,output):
    prior,current,output=prior.resolve(),current.resolve(),output.resolve()
    output.mkdir(parents=True,exist_ok=True);source=current/'source';archived=output/'source';archived.mkdir(exist_ok=True)
    sys.path.insert(0,str(source));torch.set_num_threads(1)
    from synthetic.models import VelocityNet
    from scripts.run_architecture_study import macs
    hashes={};rows=[];curves=[];initials=[];snapshots=[];monitors=[];native=[];replays=[]
    def read(path):hashes[str(path)]=sha(path);return json.loads(path.read_text())
    def load(path):hashes[str(path)]=sha(path);return torch.load(path,map_location='cpu',weights_only=False)
    def equal(a,b):torch.testing.assert_close(a,b,atol=0,rtol=0)
    def folder(study,cfg,ds,kind):return study/'runs'/cfg['experiment_id']/ds/kind/'seed000'
    oldmanifest=read(prior/'source-manifest.json');newmanifest=read(current/'source-manifest.json')
    for name,digest in newmanifest['all_files'].items():assert sha(source/name)==digest,name
    for name,digest in oldmanifest['all_files'].items():assert sha(prior/'source'/name)==sha(source/name)==digest,name
    assert read(current/'output/silu_reference_queue_20261024_m1/status.json')['state']=='COMPLETE'
    repo=Path(__file__).resolve().parents[1];oldanalysis=repo/'docs/research-architecture-search/analysis-shared-features'
    replay_source=oldanalysis/'source/audit_architecture_endpoints.py';shutil.copy2(replay_source,archived/replay_source.name)
    shutil.copy2(__file__,archived/Path(__file__).name)
    comparisons=0
    for suffix in ('v1','product16'):
        study=current/'output'/f'silu_reference_{suffix}_20261024_m1';old=prior/'output'/f'shared_features_{suffix}_20261024_m1'
        data=Path('/Users/mr.z/ode-workspace-prep/data/synthetic')/('v1' if suffix=='v1' else 'product16_v1')
        assert read(study/'status.json')['state']=='COMPLETE'
        commands=[('native',[source/'scripts/analyze_architecture_study.py','--study',study,'--source',source,'--data-root',data,'--output',output/f'{suffix}-native']),
                  ('replay',[archived/replay_source.name,'--study',study,'--source',source,'--data-root',data,'--output',output/f'{suffix}-endpoint-replay.json'])]
        for label,args in commands:
            result=subprocess.run([sys.executable,*map(str,args)],cwd=source,capture_output=True,text=True)
            (output/f'{suffix}-{label}.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
        audit=read(output/f'{suffix}-native/audit.json');native.append(audit);assert audit['state']=='PASS'
        replay=read(output/f'{suffix}-endpoint-replay.json');replays.append(replay);assert replay['state']=='PASS'
        oldaudit=read(oldanalysis/f'{suffix}-native/audit.json');assert oldaudit['state']=='PASS'
        oldreplay=read(oldanalysis/f'{suffix}-endpoint-replay.json');assert oldreplay['state']=='PASS'
        newprov=read(study/'provenance.json');oldprov=read(old/'provenance.json')
        assert newprov['dataset_manifest_sha256']==oldprov['dataset_manifest_sha256']
        for group,s,record in [('historical_stage17',old,oldaudit),('new_silu_reference',study,audit)]:
            for row in record['rows']:
                rows.append(dict(group=group,**row,study=str(s)))
                cfg=read(s/'configs'/f"{row['label']}.json");run=folder(s,cfg,row['dataset'],row['model'])
                path=run/'metrics.jsonl';hashes[str(path)]=sha(path)
                trace=[json.loads(line) for line in path.read_text().splitlines()]
                assert [r['step'] for r in trace]==list(range(0,5001,500))
                for metric in trace:curves.append(dict(group=group,dataset=row['dataset'],label=row['label'],step=metric['step'],
                    train_pair_mse=metric['train_pair_mse'],validation_pair_mse=metric['validation_pair_mse'],
                    validation_oracle_mse=metric['validation_oracle_mse'],wall_seconds=metric['wall_seconds']))
        log=study/'telemetry.jsonl';hashes[str(log)]=sha(log);history=[json.loads(s) for s in log.read_text().splitlines()]
        assert history and any(r['processes'] for r in history)
        monitors.append(dict(study=study.name,actual_telemetry=history,minimum_available_gib=min(r['memory_available_gib'] for r in history),terminal=read(study/'status.json')))
        for ds in newprov['plan']['datasets']:
            cfg=read(study/'configs/ffn_silu.json');run=folder(study,cfg,ds,'ffn');ref=cfg['backbone_initialization'][ds]
            assert sha(ref['checkpoint'])==ref['sha256'];reference=load(Path(ref['checkpoint']))
            initial=load(run/'checkpoints/step_0000000.pt');dim=16 if ds.startswith('rotated') else 2
            torch.manual_seed(0);model=VelocityNet(dim,'ffn',**cfg['model_options']);native_rng=torch.get_rng_state().clone()
            shared=[];ffn=[]
            for name,value in initial['model'].items():
                if '.ffn.' in name:equal(value,model.state_dict()[name]);ffn.append(name)
                else:equal(value,reference['model'][name]);shared.append(name)
            equal(initial['torch_rng'],native_rng)
            for name,value in initial['runtime_buffers'].items():equal(value,reference['runtime_buffers'][name])
            receipt=read(run/'provenance.json')['backbone_initialization']
            assert receipt['source_sha256']==ref['sha256'] and receipt['copied_state_keys']==sorted(shared)
            assert receipt['copied_runtime_buffer_keys']==sorted(initial['runtime_buffers'])
            budget=next(r for r in rows if r['dataset']==ds and r['label']=='vector_parent')
            assert model.parameter_count()==budget['parameters'] and macs(model,dim)==budget['forward_macs']
            assert sum(p.numel() for p in model.blocks[0].ffn.parameters())==16576
            old_cfgs={label:read(old/'configs'/f'{label}.json') for label in OLD}
            for label,oldcfg in old_cfgs.items():
                for key in ('steps','batch_size','learning_rate','weight_decay','grad_clip','eval_interval','sample_interval','checkpoint_interval','sample_count','sliced_directions','heun_steps','sampling_split'):
                    assert cfg[key]==oldcfg[key],key
                assert cfg['model_options']['width']==oldcfg['model_options']['width']==64
                assert cfg['model_options']['depth']==oldcfg['model_options']['depth']==3
                assert cfg['backbone_initialization'][ds]==oldcfg['backbone_initialization'][ds]
            for step in range(0,5001,1000):
                cp=load(run/'checkpoints'/f'step_{step:07d}.pt')
                for label,oldcfg in old_cfgs.items():
                    kind='ode_vector' if label.startswith('vector') else 'swiglu'
                    oldcp=load(folder(old,oldcfg,ds,kind)/'checkpoints'/f'step_{step:07d}.pt')
                    equal(cp['data_rng'],oldcp['data_rng']);comparisons+=1
                    if step==0:
                        for name in shared:equal(cp['model'][name],oldcp['model'][name])
                        for name,value in cp['runtime_buffers'].items():equal(value,oldcp['runtime_buffers'][name])
                snapshots.append(dict(dataset=ds,step=step,data_rng_matches_all_six_historical_arms=True))
            initials.append(dict(dataset=ds,reference=ref,non_ffn_state_tensors=len(shared),native_ffn_state_tensors=len(ffn),
                runtime_buffers=sorted(initial['runtime_buffers']),native_ffn_and_torch_rng_unchanged=True,
                parameters=model.parameter_count(),forward_macs=macs(model,dim),exact_budget_match_to_vector_parent=True))
    assert len(rows)==21 and comparisons==108 and len(initials)==3
    rows.sort(key=lambda r:(DATASETS.index(r['dataset']),ARMS.index(r['label'])))
    csv_write(output/'comparison.csv',rows);csv_write(output/'learning-curves.csv',curves)
    write(output/'pairing-audit.json',dict(state='PASS',initials=initials,snapshots=snapshots,data_rng_exact_comparisons=comparisons,
        inherited_source_files_byte_identical=len(oldmanifest['all_files']),new_source_files_verified=len(newmanifest['all_files']),
        scope='Historical controls reused after identical implementation hashes, matching configuration/common initialization and every saved data RNG state. No training-trajectory regeneration.'))
    write(output/'monitoring-evidence.json',monitors)
    def get(ds,label):return next(r for r in rows if r['dataset']==ds and r['label']==label)
    metrics=('validation_oracle_mse','jacobian_error','validation_sw2')
    ratios=[dict(dataset=ds,reference=label,**{m+'_ratio':get(ds,'ffn_silu')[m]/get(ds,label)[m] for m in metrics}) for ds in DATASETS for label in OLD]
    write(output/'reference-ratios.json',dict(state='REFERENCE_ONLY_NO_ODE_NOMINATION',rows=ratios,
        scope='Descriptive necessary baseline. Previous architecture decisions remain unchanged; plain SiLU performance does not fulfill the ODELayer goal.'))
    figures=output/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(3,3,figsize=(13,9.2),layout='constrained')
    for col,(ds,title) in enumerate(zip(DATASETS,TITLES)):
        for row,(metric,label) in enumerate(zip(metrics,('Oracle MSE','Jacobian error','Validation SW2'))):
            ax=axes[row,col];best=min(get(ds,b)[metric] for b in ('swiglu_budget','swiglu'))
            ax.bar(range(7),[get(ds,a)[metric]/best for a in ARMS],color=COLORS);ax.axhline(1,color='black',ls='--',lw=.8)
            ax.set_xticks(range(7),LABELS,rotation=48,ha='right',fontsize=7)
            ax.set(ylabel=label+' / better SwiGLU96,128',title=title if row==0 else '');ax.grid(axis='y',alpha=.15)
    for ext in ('png','pdf'):fig.savefig(figures/f'silu-reference-endpoints.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    for ax,ds,title in zip(axes,DATASETS,TITLES):
        for arm,label,color in zip(ARMS,LABELS,COLORS):
            trace=[r for r in curves if r['dataset']==ds and r['label']==arm]
            ax.plot([r['step'] for r in trace],[r['validation_oracle_mse'] for r in trace],label=label,color=color,
                ls='-' if arm=='ffn_silu' else '--',lw=2 if arm=='ffn_silu' else 1.25)
        ax.set(title=title,xlabel='Training update',ylabel='Validation oracle MSE',yscale='log');ax.grid(alpha=.15)
    axes[-1].legend(fontsize=7,frameon=False)
    for ext in ('png','pdf'):fig.savefig(figures/f'silu-reference-learning.{ext}',dpi=180)
    plt.close(fig)
    for name,digest in newmanifest['all_files'].items():assert sha(source/name)==digest,name
    write(output/'provenance.json',dict(state='PASS',analysis_sha256=sha(__file__),input_sha256=hashes,
        new_checkpoints_native_audited=sum(a['checkpoints_checked'] for a in native),new_arrays_native_audited=sum(a['arrays_checked'] for a in native),
        endpoint_replays=sum(a['final_checkpoints_reloaded'] for a in replays),maximum_forward_replay_difference=max(a['maximum_forward_difference'] for a in replays),
        maximum_jacobian_replay_difference=max(a['maximum_derivative_difference'] for a in replays),
        scope='Three existing-FFN reference runs plus18 immutable Stage17 controls. No fresh holdout or sampling-path replay.'))
    return dict(state='PASS',comparison_rows=len(rows),curve_rows=len(curves),new_initializations=len(initials),snapshot_rng_comparisons=comparisons)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prior','current','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(analyze(args.prior,args.current,args.output)))
