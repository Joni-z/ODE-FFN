"""Replay and apply the fixed temporal-generator Phase A continuation rule."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_time_generator as runner
from scripts.analyze_anchored_corrector import recompute
from synthetic.recurrent_depth import build as parent_build
from synthetic.train import restore_model


def load(path):return json.loads(path.read_text())
def write(path,value):runner.base.write(path,value)
def equal(a,b):
    assert set(a)==set(b)
    assert all(torch.equal(value,b[key]) for key,value in a.items())
def csv_write(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(group: Path,prior: Path,source: Path,output: Path):
    torch.set_num_threads(1)
    dataset='rotated_product_bimodal_16d';status=load(group/'status.json');config=load(group/'study.json')
    assert status['state']=='COMPLETE' and status['completed']==status['total']==4
    assert config['arms']==runner.ARMS and config['datasets']==[dataset]
    hashes=load(source/'source-sha256.json')
    assert load(group/'provenance.json')['source_sha256']==hashes
    for name,value in hashes.items():
        assert runner.base.sha256(source/name)==value
        assert runner.base.sha256(ROOT/name)==value
    old_hashes=load(prior/'provenance.json')['source_sha256']
    for name in ('synthetic/models.py','synthetic/train.py','synthetic/recurrent_depth.py'):
        assert hashes[name]==old_hashes[name]
    banks=group/'banks'/dataset;old_banks=prior/'banks'/dataset
    for path in old_banks.glob('*.npz'):
        with np.load(path) as a,np.load(banks/path.name) as b:
            assert a.files==b.files
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    rows,states,spectra=[],{},[]
    streams={}
    for arm,setting in runner.ARMS.items():
        run=group/'runs'/dataset/arm;native=load(run/'result.json')
        assert load(run/'status.json')['state']=='COMPLETE' and native['step']==5000
        assert all(native[k]==v for k,v in setting.items())
        old_parent=prior/'runs'/dataset/(setting['family']+'_native_final')
        assert load(run/'provenance.json')['manifest_sha256']==load(old_parent/'provenance.json')['manifest_sha256']
        for step in (0,1000,2500,5000):
            checkpoint=torch.load(run/f'step_{step:07d}.pt',map_location='cpu',weights_only=False)
            assert checkpoint['step']==step and all(torch.isfinite(v).all() for v in checkpoint['model'].values())
            parent=torch.load(old_parent/f'step_{step:07d}.pt',map_location='cpu',weights_only=False)
            assert torch.equal(checkpoint['data_rng'],parent['data_rng'])
            assert checkpoint['input_stream_digest']==parent['input_stream_digest']
            if step in streams:assert checkpoint['input_stream_digest']==streams[step]
            streams[step]=checkpoint['input_stream_digest']
            assert len(load(run/f'visits_{step:07d}.json'))==(6 if setting['family']=='ode_k2' else 0)
            if step==0:
                states[arm]=checkpoint
                for key,value in parent['model'].items():assert torch.equal(value,checkpoint['model'][key])
                equal(parent['runtime_buffers'],checkpoint['runtime_buffers'])
            if setting['family']=='ode_k2':
                effective=load(run/'effective-generator-trajectory.json')[str(step)]
                assert len(effective)==6
                for name,record in effective.items():
                    base=checkpoint['model'][name+'.proj.weight'].double()
                    delta=(checkpoint['model'][name+'.adapter.up.weight'].double()
                           @checkpoint['model'][name+'.adapter.down.weight'].double())
                    np.testing.assert_allclose(record['delta_frobenius'],float(delta.norm()),rtol=1e-12)
                    for trow in record['times']:
                        coefficient=float(runner.time_coordinate(torch.tensor(trow['time'],dtype=torch.float64))) if setting['mode']=='time' else 1.
                        matrix=base+coefficient*delta
                        singular=torch.linalg.svdvals(matrix)
                        np.testing.assert_allclose(singular.numpy(),trow['singular_values'],rtol=1e-12,atol=1e-13)
                        spectra.append(dict(arm=arm,step=step,layer=name,time=trow['time'],
                            base_frobenius=float(base.norm()),delta_frobenius=float(delta.norm()),
                            effective_sigma_max=float(singular.max()),effective_sigma_min=float(singular.min())))
        model=runner.build(16,setting['family'],setting['mode'])
        restore_model(model,checkpoint)
        rows.append(dict(arm=arm,family=setting['family'],mode=setting['mode'],origin='new adapter screen',
                         **recompute(run,banks,model)))
    for family in ('ode_k2','swiglu192'):
        a,b=[states[f'{family}_{mode}_final'] for mode in ('static','time')]
        equal(a['model'],b['model']);equal(a['runtime_buffers'],b['runtime_buffers'])
    prior_arms=load(prior/'study.json')['arms']
    for arm,setting in prior_arms.items():
        if (setting['family'].startswith('swiglu') and setting['mode']!='shared') or arm=='ode_k2_native_final':
            run=prior/'runs'/dataset/arm
            state=torch.load(run/'step_0005000.pt',map_location='cpu',weights_only=False)
            model=parent_build(16,setting['family'],setting['mode']);restore_model(model,state)
            rows.append(dict(arm=arm,family=setting['family'],mode=setting['mode'],origin='verified reused control',
                             **recompute(run,old_banks,model)))
    assert len(rows)==11
    generic=[r for r in rows if r['family'].startswith('swiglu')]
    byname={r['arm']:r for r in rows}
    candidate=byname['ode_k2_time_final'];parent=byname['ode_k2_native_final'];static=byname['ode_k2_static_final']
    best=min(generic,key=lambda r:r['primary_mse'])
    ratios=dict(primary=candidate['primary_mse']/best['primary_mse'],
        to_parent=candidate['primary_mse']/parent['primary_mse'],to_static=candidate['primary_mse']/static['primary_mse'],
        jacobian=candidate['jacobian_error']/min(r['jacobian_error'] for r in generic),
        sw2=candidate['sw2']/min(r['sw2'] for r in generic),
        macs=candidate['dense_forward_macs']/byname['swiglu192_time_final']['dense_forward_macs'])
    advance=ratios['primary']<=.9 and ratios['to_parent']<=.95 and ratios['to_static']<=.95 and ratios['jacobian']<=1.1 and ratios['sw2']<=1.1 and ratios['macs']<=1.05
    output.mkdir(parents=True,exist_ok=False);csv_write(output/'comparison.csv',rows);csv_write(output/'effective-generators.csv',spectra)
    write(output/'decision.json',dict(advance=bool(advance),ratios=ratios,primary_baseline=best['arm'],
        action='same four-arm Ring8/Gaussian comparison' if advance else 'stop this fixed time-generator setting'))
    write(output/'audit.json',dict(status='PASS',new_cells=4,new_checkpoints=16,reused_controls=7,
        full_field_jacobian_sampling_replays=11,initial_core_buffers_and_adapter_pairing='PASS',
        actual_training_stream_matches_previous='PASS',effective_generator_spectra='PASS',
        source_sha256=hashes,analysis_source_sha256=runner.base.sha256(Path(__file__)),
        metric_replay_helper_sha256=runner.base.sha256(ROOT/'scripts/analyze_anchored_corrector.py')))
    fig,axes=plt.subplots(1,3,figsize=(11,3.7))
    display=[r for r in rows if r['origin']=='new adapter screen']+[parent,byname['swiglu192_native_final'],best]
    labels=[r['arm'].replace('_final','').replace('_native_deep',' deep').replace('_',' ') for r in display]
    for ax,metric in zip(axes,('primary_mse','jacobian_error','sw2')):
        ax.barh(labels,[r[metric] for r in display],color=['#D55E00' if r['family']=='ode_k2' else '#0072B2' for r in display])
        ax.axvline(min(r[metric] for r in generic),color='black',linestyle='--',linewidth=1);ax.invert_yaxis();ax.set_title(metric)
        if ax is not axes[0]:ax.set_yticklabels([])
    fig.suptitle('Product16: temporal/static adapters and verified parent/frontier controls',fontsize=10);fig.tight_layout()
    fig.savefig(output/'comparison.png',dpi=160,bbox_inches='tight');fig.savefig(output/'comparison.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    for arm,color in (('ode_k2_static_final','#0072B2'),('ode_k2_time_final','#D55E00')):
        data=[r for r in spectra if r['arm']==arm]
        steps=(0,1000,2500,5000)
        axes[0].plot(steps,[np.mean([r['delta_frobenius']/r['base_frobenius'] for r in data if r['step']==step]) for step in steps],marker='o',color=color,label=arm)
        for time,style in ((.1,'--'),(.5,'-'),(.9,':')):
            axes[1].plot(steps,[max(r['effective_sigma_max'] for r in data if r['step']==step and r['time']==time) for step in steps],style,color=color,label=arm+f' t={time}')
    axes[0].set(xlabel='Update',ylabel='Mean ||UV||F / ||P0||F',title='Learned adapter magnitude')
    axes[1].set(xlabel='Update',ylabel='Max singular value across generators',title='Actual P(t), not only base P0')
    axes[0].legend(fontsize=7);axes[1].legend(fontsize=6);fig.tight_layout()
    fig.savefig(output/'generator-trajectories.png',dpi=160,bbox_inches='tight');fig.savefig(output/'generator-trajectories.pdf',bbox_inches='tight');plt.close(fig)
    lines=['# Time-dependent generator Phase A','','| Arm | Origin | Field MSE | Jacobian error | SW2 | Params | Dense MACs |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['arm']} | {r['origin']} | {r['primary_mse']:.7f} | {r['jacobian_error']:.5f} | {r['sw2']:.5f} | {r['parameters']} | {r['dense_forward_macs']} |")
    lines += ['','Decision: '+('advance' if advance else 'stop this setting')+'. All four new endpoints and all seven reused controls were replayed from actual weights for full field predictions, five Jacobian panels and generated samples. Initial common tensors, static/time adapter draws and all saved batch streams match their declared references.','','![Endpoint comparison](comparison.png)','','![Generator trajectories](generator-trajectories.png)','','The base P0 and time-dependent P(t) are different objects. Effective spectra are independently recomputed from actual saved factors at every declared checkpoint/time. Neither their norms nor adapter activity is a full-model stability or accuracy certificate. No novelty is claimed for time-aware low-rank adaptation.']
    (output/'analysis-report.md').write_text('\n'.join(lines)+'\n')
    (output/'stats-appendix.md').write_text('# Statistical scope\n\nTraining seed0 and the fixed5000-update endpoint. Descriptive checkpoint comparisons; times, coordinates and spectra are not independent training repetitions. Reused controls are not newly repeated trajectories. No p-values or significance claims. No fresh holdout evaluation is used in Phase A.\n')
    (output/'figure-catalog.md').write_text('# Figures\n\n- comparison: field, derivative and generation errors for every new arm, two parents and the best practical field comparator. Each dashed frontier can come from a different generic arm. Lower is better; no error bars. The continuation decision uses all verified controls, not only the displayed subset.\n- generator-trajectories: adapter/base Frobenius ratio averaged across six generators, plus maximum effective P(t) singular value at three illustrative fixed times. All five registered times and individual layers remain in the CSV. Shows whether the new factors learn; activity alone cannot justify further training.\n')
    print(json.dumps(dict(audit='PASS',advance=bool(advance),ratios=ratios)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('group','prior','source','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();analyze(a.group.resolve(),a.prior.resolve(),a.source.resolve(),a.output.resolve())
