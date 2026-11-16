"""Figures and descriptive analysis of verified frozen-stage decompositions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p=argparse.ArgumentParser()
    for name in ['verified','probe','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();verified,probe,out=map(Path,[a.verified,a.probe,a.output])
    result=json.loads(verified.read_text());raw=json.loads(probe.read_text())
    assert result['passed'] and result['models']==6 and result['layer_clock_records']==180
    assert result['source_probe_sha256']==sha(probe)
    out.mkdir(exist_ok=False,parents=True);figures=out/'figures';figures.mkdir()
    summaries=[]
    for c in result['cases']:
        e=c['entries'];orders=[x['small_remainder_order'] for x in e if x['small_remainder_order'] is not None]
        cosine=[x['unit_cosine'] for x in e if x['unit_cosine'] is not None]
        stats={}
        for name in ['unit_relative_prediction_error','small_horizon_relative_prediction_error','unit_gap_over_increment']:
            v=np.array([x[name] for x in e]);stats[name]=dict(median=float(np.median(v)),iqr=np.quantile(v,[.25,.75]).tolist(),maximum=float(v.max()))
        stats['unit_cosine_median']=float(np.median(cosine)) if cosine else None
        stats['small_remainder_order_median']=float(np.median(orders)) if orders else None
        stats['unit_prediction_error_at_least_one']=sum(x['unit_relative_prediction_error']>=1 for x in e)
        summaries.append(dict(id=c['id'],dataset=c['dataset'],family=c['family'],field_mse=c['field_mse'],
            interventions=c['interventions'],statistics=stats))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    families=['state_midpoint4','stage_operator_midpoint4','contrast_operator_midpoint4']
    labels=['Shared P','Anchored stages','Mean-preserving']
    tasks=['gmm_ring8_2d','rotated_product_bimodal_16d']
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.1),constrained_layout=True)
    for j,task in enumerate(tasks):
        cases=[next(c for c in result['cases'] if c['dataset']==task and c['family']==f) for f in families]
        for mode,offset,color in [('native',-.25,'#2166ac'),('mean',0,'#8c510a'),('reverse',.25,'#777777')]:
            y=[c['field_mse'] if mode=='native' else next(v['field_mse'] for v in c['interventions'] if v['mode']==mode) for c in cases]
            axes[0,j].bar(np.arange(3)+offset,y,width=.23,color=color,label=mode)
        axes[0,j].set(yscale='log',xticks=range(3),xticklabels=labels,ylabel='Oracle field MSE (log scale)',title='Ring8' if j==0 else 'Product16')
        axes[0,j].yaxis.set_major_formatter(FormatStrFormatter('%.3f'));axes[0,j].tick_params(axis='x',rotation=15)
        axes[0,j].legend(frameon=False,fontsize=8)
        for i,c in enumerate(cases):
            v=np.array([e['unit_gap_over_increment'] for e in c['entries']]);q=np.quantile(v,[.25,.5,.75])
            axes[1,j].scatter(i+np.linspace(-.07,.07,len(v)),v,s=12,color=['#2166ac','#d6604d','#8c510a'][i],alpha=.65)
            axes[1,j].vlines(i,q[0],q[2],color='black',lw=2);axes[1,j].scatter(i,q[1],marker='_',s=150,color='black',zorder=3)
        axes[1,j].set(xticks=range(3),xticklabels=labels,ylabel='Mean-map change / native increment',title='Local sensitivity at internal horizon 1')
        axes[1,j].yaxis.set_major_formatter(FormatStrFormatter('%.3f'));axes[1,j].tick_params(axis='x',rotation=15)
    for ext in ['png','pdf']:fig.savefig(figures/('field-and-stage-action.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.2),constrained_layout=True)
    for j,task in enumerate(tasks):
        for family,color,label in [('stage_operator_midpoint4','#d6604d','Anchored stages'),('contrast_operator_midpoint4','#8c510a','Mean-preserving')]:
            c=next(c for c in raw['rows'] if c['dataset']==task and c['family']==family)
            taus=np.array([r['tau'] for r in c['layers'][0]['mean_flow_expansion']['rows']])
            values=np.array([[r['relative_mean_prediction_error'] for r in e['mean_flow_expansion']['rows']] for e in c['layers']])
            lo,mid,hi=np.quantile(values,[.25,.5,.75],axis=0)
            axes[j].plot(taus,mid,'o-',color=color,label=label);axes[j].fill_between(taus,lo,hi,color=color,alpha=.16)
        axes[j].plot(taus,taus,'--',color='.6',lw=1,label='Slope-one guide')
        axes[j].set(xscale='log',yscale='log',xlabel='Internal horizon',ylabel='Relative error of second-order prediction',title='Ring8' if j==0 else 'Product16',xticks=taus,xticklabels=['1','1/2','1/4','1/8','1/16'])
        axes[j].yaxis.set_major_formatter(FormatStrFormatter('%.3f'));axes[j].legend(frameon=False,fontsize=8)
    for ext in ['png','pdf']:fig.savefig(figures/('expansion-validity.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    table='| Model / task | Native MSE | Mean MSE | Reverse MSE | Relative local mean-map change | Unit prediction error | Error at 1/16 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for c in summaries:
        label=labels[families.index(c['family'])];task='Ring8' if c['dataset']==tasks[0] else 'Product16';s=c['statistics'];fields={v['mode']:v['field_mse'] for v in c['interventions']}
        table+=f"| {label} / {task} | {c['field_mse']:.3f} | {fields['mean']:.3f} | {fields['reverse']:.3f} | {s['unit_gap_over_increment']['median']:.3f} | {s['unit_relative_prediction_error']['median']:.3f} | {s['small_horizon_relative_prediction_error']['median']:.3f} |\n"
    (out/'table.md').write_text(table)
    summary=dict(passed=True,cases=summaries,verification_sha256=sha(verified),probe_sha256=sha(probe),analysis_source_sha256=sha(Path(__file__)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    receipt=dict(passed=True,figures=2,model_cases=6,scope='deterministic frozen-model diagnostics, not sampling quality or retraining gains',artifacts={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    (out/'COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'passed':True,'output':str(out),'figures':2}))


if __name__=='__main__':main()
