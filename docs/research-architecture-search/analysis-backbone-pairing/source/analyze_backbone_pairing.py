"""Audit the shared-backbone control and retain original/paired comparisons.

Read-only with respect to studies and checkpoints. Writes only the requested
analysis directory; no training, checkpoint selection, or generated samples.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

DATASETS=('gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d')
TITLES=('Gaussian2D','Ring8','Product16')
OLD_ARMS=('scalar_mul','scalar_add','vector_mul','vector_add','swiglu_budget','swiglu')
PLOT_ARMS=tuple(('original',k) for k in OLD_ARMS)+(('paired','swiglu_budget'),('paired','swiglu'))
PLOT_LABELS=('S mul','S add','V mul','V add','Swi small old','Swi full old','Swi small paired','Swi full paired')
COLORS=('#d2a066','#b97a34','#e78376','#bf4937','#8797a8','#3e5874','#49a1b1','#166477')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def analyze(prior:Path,current:Path,output:Path):
    torch.set_num_threads(1);output.mkdir(parents=True,exist_ok=True)
    figures=output/'figures';figures.mkdir(exist_ok=True)
    hashes={};rows=[];curves={};checks=[];telemetry=[]
    def record(path):hashes[str(path)]=sha(path)
    def read(path):record(path);return json.loads(path.read_text())
    def load(path):record(path);return torch.load(path,map_location='cpu',weights_only=False)
    def eq(a,b):torch.testing.assert_close(a,b,atol=0,rtol=0)
    def folder(study,cfg,ds,kind):return study/'runs'/cfg['experiment_id']/ds/kind/'seed000'
    queue=read(current/'output/backbone_pairing_queue_20261024_m1/status.json')
    assert queue['state']=='COMPLETE' and queue['completed']==queue['total']==2
    for frozen in (prior,current):
        manifest=read(frozen/'source-manifest.json')
        for name,digest in manifest['all_files'].items():assert sha(frozen/'source'/name)==digest,name
    data_rng_comparisons=0
    for suffix in ('v1','product16'):
        old=prior/'output'/f'additive_generator_{suffix}_20261024_m1'
        new=current/'output'/f'backbone_pairing_{suffix}_20261024_m1'
        oldaudit=read(old/'audit/audit.json');newaudit=read(new/'audit/audit.json')
        assert oldaudit['state']==newaudit['state']=='PASS'
        shutil.copy2(new/'audit/audit.json',output/f'{suffix}-native-audit.json')
        oldprov=read(old/'provenance.json');newprov=read(new/'provenance.json')
        assert oldprov['dataset_manifest_sha256']==newprov['dataset_manifest_sha256']
        status=read(new/'status.json');assert status['state']=='COMPLETE'
        log=new/'telemetry.jsonl';record(log)
        history=[json.loads(l) for l in log.read_text().splitlines()]
        assert history and all(r['memory_available_gib']>0 for r in history)
        monitor=read(new/'monitor.json')
        telemetry.append(dict(study=new.name,records=len(history),first=history[0],last_actual_sample=history[-1],
                              final_monitor_state=monitor['state'],minimum_available_gib=min(r['memory_available_gib'] for r in history)))
        for group,study,audit in (('original',old,oldaudit),('paired',new,newaudit)):
            for item in audit['rows']:
                cfg=read(study/'configs'/f"{item['label']}.json")
                run=folder(study,cfg,item['dataset'],item['model'])
                metric=run/'metrics.jsonl';record(metric)
                trace=[json.loads(l) for l in metric.read_text().splitlines()]
                assert trace[-1]['step']==5000
                curves[(item['dataset'],group,item['label'])]=trace
                rows.append(dict(group=group,**item,study=str(study)))
        for ds in newprov['plan']['datasets']:
            cache={}
            def oldcp(label,step):
                key=(label,step)
                if key not in cache:
                    cfg=read(old/'configs'/f'{label}.json')
                    kind='ode_original' if label.startswith('scalar') else ('ode_vector' if label.startswith('vector') else 'swiglu')
                    cache[key]=load(folder(old,cfg,ds,kind)/'checkpoints'/f'step_{step:07d}.pt')
                return cache[key]
            for label in ('swiglu_budget','swiglu'):
                cfg=read(new/'configs'/f'{label}.json');run=folder(new,cfg,ds,'swiglu')
                reference=cfg['backbone_initialization'][ds]
                assert sha(reference['checkpoint'])==reference['sha256']
                original=load(Path(reference['checkpoint']));native=oldcp(label,0)
                provenance=read(run/'provenance.json');receipt=provenance['backbone_initialization']
                assert receipt['source_sha256']==reference['sha256']
                initial=load(run/'checkpoints/step_0000000.pt')
                assert set(initial['model'])==set(native['model'])
                shared=[];ffn=[]
                for name,value in initial['model'].items():
                    if '.ffn.' in name:eq(value,native['model'][name]);ffn.append(name)
                    else:eq(value,original['model'][name]);shared.append(name)
                assert sorted(shared)==receipt['copied_state_keys']
                assert set(initial['runtime_buffers'])==set(original['runtime_buffers'])
                for name,value in initial['runtime_buffers'].items():eq(value,original['runtime_buffers'][name])
                assert sorted(initial['runtime_buffers'])==receipt['copied_runtime_buffer_keys']
                eq(initial['torch_rng'],native['torch_rng'])
                snapshots=sorted((run/'checkpoints').glob('step_*.pt'))
                assert [p.stem for p in snapshots]==[f'step_{step:07d}' for step in range(0,5001,1000)]
                for p in snapshots:
                    checkpoint=load(p)
                    for oldlabel in OLD_ARMS[:4]:
                        eq(checkpoint['data_rng'],oldcp(oldlabel,checkpoint['step'])['data_rng'])
                        data_rng_comparisons+=1
                checks.append(dict(dataset=ds,label=label,state='PASS',reference=reference,
                                   non_ffn_tensor_count=len(shared),unchanged_ffn_tensor_count=len(ffn),
                                   runtime_buffer_names=sorted(initial['runtime_buffers']),
                                   native_initial_torch_rng_unchanged=True,snapshot_data_rng_matches_four_reused_ODE_arms=True))
    assert len(rows)==24 and len(checks)==6 and data_rng_comparisons==144
    rows.sort(key=lambda r:(DATASETS.index(r['dataset']),PLOT_ARMS.index((r['group'],r['label']))))
    with (output/'comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def get(ds,group,label):return next(r for r in rows if r['dataset']==ds and r['group']==group and r['label']==label)
    shifts=[]
    for ds in DATASETS:
        for label in ('swiglu_budget','swiglu'):
            a,b=get(ds,'original',label),get(ds,'paired',label)
            shifts.append(dict(dataset=ds,label=label,**{metric+'_ratio':b[metric]/a[metric] for metric in ('validation_oracle_mse','jacobian_error','validation_sw2')}))
    screens={}
    for label,parent in (('scalar_add','scalar_mul'),('vector_add','vector_mul')):
        groups={}
        for group in ('original','paired'):
            details={};leads=[]
            for ds in DATASETS:
                item=get(ds,'original',label);p=get(ds,'original',parent)
                best={m:min(get(ds,group,l)[m] for l in ('swiglu_budget','swiglu')) for m in ('validation_oracle_mse','jacobian_error','validation_sw2')}
                ratio={m:item[m]/best[m] for m in best}
                if ds!=DATASETS[0] and ratio['validation_oracle_mse']<=.9 and item['validation_oracle_mse']<p['validation_oracle_mse']:leads.append(ds)
                details[ds]=dict(ratio_to_better_swiglu=ratio,parent_oracle_ratio=item['validation_oracle_mse']/p['validation_oracle_mse'])
            passes=bool(leads) and all(details[d]['ratio_to_better_swiglu']['validation_oracle_mse']<=1.05 for d in DATASETS) and all(details[d]['ratio_to_better_swiglu'][m]<=1.1 for d in DATASETS[1:] for m in ('jacobian_error','validation_sw2'))
            groups[group]=dict(passes=passes,leads=leads,details=details)
        screens[label]=groups
    assert not any(v['paired']['passes'] for v in screens.values())
    write(output/'screen-decision.json',dict(state='NO_CANDIDATE',comparisons=screens,scope='Descriptive reapplication of the unchanged screen; does not revise earlier STOP decisions or select a winner through baseline weakening.'))
    write(output/'initialization-audit.json',dict(state='PASS',new_initial_checkpoints=6,shared_tensor_checks=checks,data_rng_snapshot_comparisons=data_rng_comparisons,
        scope='All six new initializations and all36 snapshots compared with each of four reused ODE data-generator states. Native SwiGLU FFN and initial torch RNG are unchanged.'))
    write(output/'baseline-shifts.json',shifts);write(output/'monitoring-evidence.json',telemetry)
    fig,axes=plt.subplots(3,3,figsize=(13,9.4),layout='constrained')
    for j,(ds,title) in enumerate(zip(DATASETS,TITLES)):
        for i,(metric,name) in enumerate((('validation_oracle_mse','Oracle MSE'),('jacobian_error','Jacobian error'),('validation_sw2','Validation SW2'))):
            ax=axes[i,j];best=min(get(ds,'paired',l)[metric] for l in ('swiglu_budget','swiglu'))
            values=[get(ds,g,a)[metric]/best for g,a in PLOT_ARMS]
            ax.bar(range(8),values,color=COLORS);ax.axhline(1,color='black',ls='--',lw=.8)
            ax.set_xticks(range(8),PLOT_LABELS,rotation=52,ha='right',fontsize=7)
            ax.set(ylabel=name+' / better paired SwiGLU',title=title if i==0 else '');ax.grid(axis='y',alpha=.15)
    for ext in ('png','pdf'):fig.savefig(figures/f'paired-endpoints.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
    for ax,ds,title in zip(axes,DATASETS,TITLES):
        for (group,label),display,color in zip(PLOT_ARMS,PLOT_LABELS,COLORS):
            trace=curves[ds,group,label]
            ax.plot([r['step'] for r in trace],[r['validation_oracle_mse'] for r in trace],color=color,
                    ls='--' if group=='original' and label.startswith('swiglu') else '-',label=display,lw=1.4)
        ax.set(xlabel='Training update',ylabel='Validation oracle MSE',title=title,yscale='log');ax.grid(alpha=.15)
    axes[2].legend(loc='upper right',fontsize=7,frameon=False)
    for ext in ('png','pdf'):fig.savefig(figures/f'paired-learning.{ext}',dpi=180)
    plt.close(fig)
    source=output/'source';source.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__),source/Path(__file__).name)
    replay=[]
    for suffix in ('v1','product16'):
        result=read(output/f'{suffix}-endpoint-replay.json');assert result['state']=='PASS';replay.append(result)
    write(output/'provenance.json',dict(state='PASS',script_sha256=sha(__file__),input_sha256=hashes,new_checkpoints_native_audited=36,new_diagnostic_arrays_native_audited=30,
        endpoint_replays=sum(r['final_checkpoints_reloaded'] for r in replay),maximum_forward_replay_difference=max(r['maximum_forward_difference'] for r in replay),
        maximum_derivative_replay_difference=max(r['maximum_derivative_difference'] for r in replay),scope='Existing validation endpoints/curves and explicit initialization/RNG checks. No new training, sampling-trajectory replay, or fresh holdout.'))
    return dict(state='PASS',rows=len(rows),initializations=len(checks),snapshot_data_rng_comparisons=data_rng_comparisons,paired_candidate_passes=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prior','current','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(analyze(args.prior,args.current,args.output)))
