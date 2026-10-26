"""Independent full-model hook replay of normalized-state interventions."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.recurrent_depth import build
from synthetic.train import restore_model

ARMS={'ode_k2_native_final':'ode_k2','swiglu192_native_final':'swiglu192','swiglu128_native_deep':'swiglu128'}
TIMES=(.1,.3,.5,.7,.9)
VARIANTS=('native','drop_uniform','drop_radial','drop_tangent')


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def csv_read(path):
    with Path(path).open() as handle:return list(csv.DictReader(handle))
def csv_write(path,rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class HookProjection:
    """Use module hooks and unit-vector projections, independently of GeometryNet."""
    def __init__(self,model):
        self.mode='native';self.index=None;self.hidden={};self.records=[];self.capture=False;self.hooks=[]
        for i,block in enumerate(model.raw.blocks):
            self.hooks.append(block.register_forward_pre_hook(lambda module,args,i=i:self.before(i,args)))
            self.hooks.append(block.ffn.register_forward_hook(lambda module,args,value,i=i:self.after(i,value)))

    def before(self,index,args):self.hidden[index]=args[0]

    def after(self,index,value):
        if self.mode=='native' and not self.capture:return None
        h=self.hidden[index];d=h.shape[-1]
        ones=torch.ones(d,dtype=h.dtype,device=h.device)/math.sqrt(d)
        centered=h-(h@ones)[:,None]*ones
        norm=centered.norm(dim=-1,keepdim=True)
        unit=centered/torch.where(norm>0,norm,torch.ones_like(norm))
        uniform=(value@ones)[:,None]*ones
        radial=((value-uniform)*unit).sum(-1,keepdim=True)*unit
        tangent=value-uniform-radial
        if self.capture:
            self.records.append(dict(block=index,hidden_energy=float(h.detach().square().mean()),
                centered_energy=float(centered.detach().square().mean()),branch_energy=float(value.detach().square().mean()),
                uniform_energy=float(uniform.detach().square().mean()),radial_energy=float(radial.detach().square().mean()),
                tangent_energy=float(tangent.detach().square().mean())))
        selected=self.index is None or self.index==index
        if not selected:return None
        if self.mode=='drop_uniform':return value-uniform
        if self.mode=='drop_radial':return value-radial
        if self.mode=='drop_tangent':return uniform+radial
        return None

    def close(self):
        for hook in self.hooks:hook.remove()


def derivative(model,x,t):
    x=x.detach().clone().requires_grad_(True);value=model(x,t)
    jac=torch.stack([torch.autograd.grad(value[:,i].sum(),x,retain_graph=i+1<value.shape[1])[0]
                     for i in range(value.shape[1])],dim=1)
    return value.detach().numpy(),jac.detach().numpy()


def risks(native,changed,truth):
    before=np.mean((native-truth)**2,axis=1);after=np.mean((changed-truth)**2,axis=1)
    alignment=2*np.mean((native-truth)*(changed-native),axis=1);energy=np.mean((changed-native)**2,axis=1)
    np.testing.assert_allclose(after-before,alignment+energy,rtol=1e-9,atol=1e-11)
    return dict(native_error=before,changed_error=after,difference=after-before,alignment=alignment,change_energy=energy)


def analyze(raw,groups,output):
    torch.set_num_threads(1)
    state=load(raw/'status.json');assert state['state']=='COMPLETE' and state['completed']==9
    assert (state['metric_rows'],state['panel_rows'],state['trace_rows'])==(36,450,276480)
    provenance=load(raw/'provenance.json')
    for path,value in provenance['inputs'].items():assert sha(path)==value
    for name in ('synthetic/models.py','synthetic/recurrent_depth.py','synthetic/train.py'):
        assert sha(ROOT/name)==provenance['source_sha256'][name]
    metrics=csv_read(raw/'metrics.csv');panels=csv_read(raw/'panels.csv')
    native_trace={};group_trace={}
    selected_case=None
    with (raw/'traces.csv').open() as handle:
        for row in csv.DictReader(handle):
            if row['variant']!='native':continue
            key=(row['dataset'],row['arm'],int(row['block']))
            time_key=key+(float(row['time']),)
            for target,key2 in ((native_trace,key),(group_trace,time_key)):
                record=target.setdefault(key2,dict(n=0,hidden_energy=0.,centered_energy=0.,branch_energy=0.,uniform_energy=0.,radial_energy=0.,tangent_energy=0.,angle=0.,negative=0,eps=0.,variance_min=float('inf')))
                record['n']+=1
                for name in ('hidden_energy','centered_energy','branch_energy','uniform_energy','radial_energy','tangent_energy'):
                    record[name]+=float(row[name])
                record['angle']+=float(row['angle_degrees']);record['negative']+=float(row['radial_multiplier'])<0
                record['eps']+=float(row['epsilon_fraction']);record['variance_min']=min(record['variance_min'],float(row['centered_energy']))
            if row['dataset']=='rotated_product_bimodal_16d' and row['arm']=='ode_k2_native_final' and row['time']=='0.5' and row['block']=='1' and row['input_index']=='0':selected_case=row
    # The illustrative case is fixed by index/time/block, never chosen for a favorable effect.
    assert selected_case is not None
    output.mkdir(parents=True,exist_ok=False);(output/'per-input-risks').mkdir()
    counter=0;max_field_error=max_jacobian_error=0.;risk_records=[]
    for group in groups:
        config=load(group/'study.json')
        for dataset in config['datasets']:
            bank=group/'banks'/dataset
            with np.load(bank/'field.npz') as data:x,t,oracle=[data[k].astype(np.float64) for k in ('x','t','oracle')]
            for arm,family in ARMS.items():
                run=raw/dataset/arm
                native_path=group/'runs'/dataset/arm/'step_0005000.pt'
                model=build(x.shape[1],family,'native');restore_model(model,torch.load(native_path,map_location='cpu',weights_only=False));model=model.double().eval()
                for p in model.parameters():p.requires_grad_(False)
                editor=HookProjection(model)
                try:
                    with np.load(run/'native-field.npz') as data:native=data['prediction']
                    for variant in VARIANTS:
                        editor.mode=variant;editor.index=None;prediction=[]
                        with torch.no_grad():
                            for start in range(0,len(x),1024):prediction.append(model(torch.from_numpy(x[start:start+1024]),torch.from_numpy(t[start:start+1024])).numpy())
                        prediction=np.concatenate(prediction)
                        with np.load(run/(variant+'-field.npz')) as data:saved=data['prediction']
                        max_field_error=max(max_field_error,float(np.max(np.abs(prediction-saved))))
                        np.testing.assert_allclose(prediction,saved,rtol=1e-7,atol=1e-8)
                        row=next(r for r in metrics if r['dataset']==dataset and r['arm']==arm and r['variant']==variant)
                        np.testing.assert_allclose(np.mean((saved-oracle)**2),float(row['primary_mse']),rtol=1e-12,atol=1e-14)
                        values=risks(native,saved,oracle)
                        np.savez_compressed(output/'per-input-risks'/f'{dataset}-{arm}-{variant}.npz',**values)
                        risk_records.append(dict(dataset=dataset,arm=arm,variant=variant,
                            field_difference=float(values['difference'].mean()),alignment=float(values['alignment'].mean()),
                            finite_change_energy=float(values['change_energy'].mean())))
                        jac_errors=[]
                        for time in TIMES:
                            with np.load(bank/f'panel_{time:.1f}.npz') as data:
                                px,pt,truth,jtruth=[data[k].astype(np.float64) for k in ('x','t','oracle','oracle_jacobian')]
                            editor.capture=variant=='native';editor.records=[]
                            value,jac=derivative(model,torch.from_numpy(px),torch.from_numpy(pt));editor.capture=False
                            with np.load(run/f'{variant}-panel-{time:.1f}.npz') as data:
                                saved_value,saved_jac=data['prediction'],data['jacobian']
                            max_field_error=max(max_field_error,float(np.max(np.abs(value-saved_value))))
                            max_jacobian_error=max(max_jacobian_error,float(np.max(np.abs(jac-saved_jac))))
                            np.testing.assert_allclose(value,saved_value,rtol=1e-7,atol=1e-8)
                            np.testing.assert_allclose(jac,saved_jac,rtol=1e-6,atol=1e-6)
                            jac_errors.append(float(np.mean(np.sum((saved_jac-jtruth)**2,axis=(1,2)))))
                            if variant=='native':
                                for record in editor.records:
                                    expected=group_trace[dataset,arm,record['block'],time]
                                    for key in ('hidden_energy','centered_energy','branch_energy','uniform_energy','radial_energy','tangent_energy'):
                                        np.testing.assert_allclose(record[key],expected[key]/expected['n'],rtol=1e-11,atol=1e-12)
                            with np.load(run/f'native-panel-{time:.1f}.npz') as data:base=data['prediction']
                            risks(base,saved_value,truth)
                        np.testing.assert_allclose(np.mean(jac_errors),float(row['jacobian_error']),rtol=1e-12,atol=1e-12)
                    for variant in ('drop_radial','drop_tangent'):
                        editor.mode=variant
                        for index in (0,1,2):
                            editor.index=index
                            for time in TIMES:
                                with np.load(bank/f'panel_{time:.1f}.npz') as data:
                                    px,pt,truth=[data[k].astype(np.float64) for k in ('x','t','oracle')]
                                with torch.no_grad():value=model(torch.from_numpy(px),torch.from_numpy(pt)).numpy()
                                with np.load(run/f'{variant}-block-{index}-panel-{time:.1f}.npz') as data:saved=data['prediction']
                                np.testing.assert_allclose(value,saved,rtol=1e-7,atol=1e-8)
                                expected=next(r for r in panels if r['dataset']==dataset and r['arm']==arm and r['variant']==variant and r['block']==str(index) and float(r['time'])==time)
                                np.testing.assert_allclose(np.mean((saved-truth)**2),float(expected['field_mse']),rtol=1e-12,atol=1e-12)
                finally:editor.close()
                counter+=1;print(json.dumps(dict(replayed_models=counter,total=9,dataset=dataset,arm=arm)),flush=True)
    assert counter==9
    bytask={d:[r for r in metrics if r['dataset']==d] for d in {r['dataset'] for r in metrics}}
    comparisons={}
    for dataset,values in bytask.items():
        own=next(r for r in values if r['arm']=='ode_k2_native_final' and r['variant']=='drop_radial')
        parent=next(r for r in values if r['arm']=='ode_k2_native_final' and r['variant']=='native')
        generic=[r for r in values if r['arm'].startswith('swiglu')]
        comparisons[dataset]=dict(primary_ratio=float(own['primary_mse'])/min(float(r['primary_mse']) for r in generic),
            parent_ratio=float(own['primary_mse'])/float(parent['primary_mse']),
            jacobian_ratio=float(own['jacobian_error'])/min(float(r['jacobian_error']) for r in generic))
    nonlinear=('gmm_ring8_2d','rotated_product_bimodal_16d')
    candidate=all(v['primary_ratio']<=1.05 for v in comparisons.values()) and any(comparisons[d]['primary_ratio']<=.9 and comparisons[d]['parent_ratio']<1 for d in nonlinear) and all(comparisons[d]['jacobian_ratio']<=1.1 for d in nonlinear)
    write(output/'decision.json',dict(candidate=bool(candidate),comparisons=comparisons,
        action='registered sampling checks' if candidate else 'stop this geometric projection setting; no sampling or training'))
    geometry=[]
    for (dataset,arm,index),record in native_trace.items():
        geometry.append(dict(dataset=dataset,arm=arm,block=index,inputs=record['n'],
            hidden_rms=math.sqrt(record['hidden_energy']/record['n']),centered_rms=math.sqrt(record['centered_energy']/record['n']),
            uniform_energy_fraction=record['uniform_energy']/record['branch_energy'],radial_energy_fraction=record['radial_energy']/record['branch_energy'],
            tangent_energy_fraction=record['tangent_energy']/record['branch_energy'],mean_angle_degrees=record['angle']/record['n'],
            radial_reversal_fraction=record['negative']/record['n'],mean_epsilon_fraction=record['eps']/record['n'],minimum_centered_variance=record['variance_min']))
    csv_write(output/'native-geometry.csv',geometry);csv_write(output/'risk-polarizations.csv',risk_records)
    csv_write(output/'comparison.csv',metrics);csv_write(output/'single-and-global-panels.csv',panels)
    write(output/'audit.json',dict(status='PASS',models=9,full_validation_arrays=36,panel_predictions=450,
        complete_panel_jacobians=180,native_geometry_groups=135,max_field_absolute_replay_error=max_field_error,
        max_jacobian_absolute_replay_error=max_jacobian_error,mechanism='independent normalized-unit projections through native FFN hooks',
        per_input_risk_polarization='PASS',input_provenance_sha256=sha(raw/'provenance.json'),script_sha256=sha(Path(__file__))))
    datasets=('gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d');titles=('Gaussian2D','Ring8','Product16')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,dataset,title in zip(axes,datasets,titles):
        values=[r for r in geometry if r['dataset']==dataset];left=np.zeros(len(values))
        for key,label,color in (('uniform_energy_fraction','Uniform','#999999'),('radial_energy_fraction','Radial','#E69F00'),('tangent_energy_fraction','Tangent','#009E73')):
            v=[r[key] for r in values];ax.barh(range(9),v,left=left,label=label,color=color);left+=v
        labels=[('ODE' if r['arm'].startswith('ode') else 'Swi192' if '192' in r['arm'] else 'Swi128deep')+f" b{r['block']}" for r in values]
        ax.set_yticks(range(9),labels);ax.invert_yaxis();ax.set(title=title,xlabel='Fraction of total branch energy',xlim=(0,1))
    axes[-1].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,fontsize=8);fig.tight_layout()
    fig.savefig(output/'branch-components.png',dpi=160,bbox_inches='tight');fig.savefig(output/'branch-components.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.5))
    for ax,dataset,title in zip(axes,datasets,titles):
        for arm,color in zip(ARMS,('#D55E00','#0072B2','#009E73')):
            values=sorted((r for r in geometry if r['dataset']==dataset and r['arm']==arm),key=lambda r:r['block'])
            ax.plot(range(3),[r['centered_rms'] for r in values],'-o',color=color,label=arm+' centered')
            ax.plot(range(3),[r['hidden_rms'] for r in values],'--',color=color)
        ax.set(title=title,xlabel='Before block',ylabel='State RMS',xticks=(0,1,2))
    axes[-1].legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=7);fig.suptitle('Solid: centered state; dashed: full state',fontsize=10);fig.tight_layout()
    fig.savefig(output/'hidden-radii.png',dpi=160,bbox_inches='tight');fig.savefig(output/'hidden-radii.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.7));labels=[];matrix=[]
    for arm in ARMS:
        for variant in ('drop_uniform','drop_radial','drop_tangent'):
            ratios=[]
            for dataset in datasets:
                before=next(float(r['primary_mse']) for r in metrics if r['dataset']==dataset and r['arm']==arm and r['variant']=='native')
                after=next(float(r['primary_mse']) for r in metrics if r['dataset']==dataset and r['arm']==arm and r['variant']==variant)
                ratios.append(after/before)
            labels.append(('ODE' if arm.startswith('ode') else 'Swi192' if '192' in arm else 'Swi128deep')+' '+variant);matrix.append(ratios)
    matrix=np.asarray(matrix);im=ax.imshow(np.log10(matrix),aspect='auto',cmap='YlOrRd',vmin=0)
    ax.set_xticks(range(3),titles);ax.set_yticks(range(9),labels)
    for i in range(9):
        for j in range(3):ax.text(j,i,f'{matrix[i,j]:.2f}x',ha='center',va='center',color='white' if matrix[i,j]>100 else 'black')
    ax.set_title('Field MSE / same frozen native model (lower is better)');fig.colorbar(im,ax=ax,label='log10 ratio');fig.tight_layout()
    fig.savefig(output/'intervention-errors.png',dpi=160,bbox_inches='tight');fig.savefig(output/'intervention-errors.pdf',bbox_inches='tight');plt.close(fig)
    # A fixed illustrative point in the exact(c,q)plane; never selected by its outcome.
    row=selected_case;a=1/math.sqrt(3);variance=float(row['centered_energy']);beta=float(row['beta'])
    vertical=a*math.sqrt(float(row['tangent_energy'])/variance);native_point=(1+a*beta,vertical);projected_point=(1,vertical)
    fig,ax=plt.subplots(figsize=(5,4))
    for point,label,color in (((1,0),'Current state','#333333'),(native_point,'Native update','#0072B2'),(projected_point,'Radial removed','#D55E00')):
        ax.annotate('',xy=point,xytext=(0,0),arrowprops=dict(arrowstyle='->',color=color,lw=2));ax.scatter(*point,color=color,label=label)
    ax.set(xlabel='Centered-state direction / original radius',ylabel='Orthogonal feature direction / original radius',title='Fixed example: Product16 ODE,t=.5,input0,block1')
    ax.axhline(0,color='gray',linewidth=.5);ax.axvline(0,color='gray',linewidth=.5);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(output/'case-study.png',dpi=160,bbox_inches='tight');fig.savefig(output/'case-study.pdf',bbox_inches='tight');plt.close(fig)
    source_group=next(g for g in groups if 'rotated_product_bimodal_16d' in load(g/'study.json')['datasets'])
    with np.load(source_group/'banks/rotated_product_bimodal_16d/panel_0.5.npz') as data:truth=data['oracle'][0]
    case_dir=raw/'rotated_product_bimodal_16d/ode_k2_native_final'
    with np.load(case_dir/'native-panel-0.5.npz') as data:before=float(np.mean((data['prediction'][0]-truth)**2))
    with np.load(case_dir/'drop_radial-block-1-panel-0.5.npz') as data:after=float(np.mean((data['prediction'][0]-truth)**2))
    write(output/'case-study.json',dict(selection='fixed input0,time.5,block1,Product16 ODE',
        beta=beta,normalized_native_next=native_point,normalized_radial_removed_next=projected_point,
        native_field_squared_error=before,changed_field_squared_error=after,scope='one illustrative point,not an aggregate result'))
    print(json.dumps(dict(audit='PASS',candidate=bool(candidate),field_error=max_field_error,jacobian_error=max_jacobian_error)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--inputs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.raw.resolve(),[q.resolve() for q in a.inputs],a.output.resolve())
