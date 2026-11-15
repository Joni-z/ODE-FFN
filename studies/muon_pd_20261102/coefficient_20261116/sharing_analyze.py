"""Completed-cohort analysis; refuses partial sharing evaluations."""
import argparse,csv,hashlib,itertools,json,shutil
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def csvrows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def holm(values):
    order=np.argsort(values);result=np.zeros(len(values));last=0.
    for rank,index in enumerate(order):last=max(last,min(1.,values[index]*(len(values)-rank)));result[index]=last
    return result.tolist()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root);source=root/'sharing-analysis';done=read(source/'COMPLETE.json')
    assert done['passed'] and done['training_cases']==2 and done['reference_cases']==6
    assert done['selection_cases']==done['confirmation_cases']==8
    for n,expected in [('verification.json',16),('confirmation-verification.json',8)]:
        verification=read(source/n);assert verification['passed'] and verification['cells']==8 and verification['rows']==expected and verification['max_numpy_difference']<1e-10
    probe=read(root/'sharing-probe.json');cost=read(root/'sharing-cost.json');pairing=read(root/'sharing-pairing.json')
    assert probe['passed'] and len(probe['rows'])==4 and cost['passed'] and len(cost['rows'])==8 and pairing['passed']
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);fig=out/'figures';fig.mkdir()
    for n in ['COMPLETE.json','verification.json','confirmation-verification.json','selection-metrics.csv','selection-banks.csv','confirmation-metrics.csv','confirmation-banks.csv','receipts.json']:
        shutil.copy2(source/n,out/n)
    for n in ['probe','cost','pairing']:shutil.copy2(root/('sharing-'+n+'.json'),out/('sharing-'+n+'.json'))
    selection=csvrows(source/'selection-metrics.csv');confirmation=csvrows(source/'confirmation-metrics.csv');banks=csvrows(source/'confirmation-banks.csv')
    assert len(selection)==16 and len(confirmation)==8 and len(banks)==64
    costs={r['id']:r for r in cost['rows']};byid={r['id']:r for r in confirmation};table=[]
    for r in confirmation:
        c=costs[r['id']];assert c['checkpoint']['checkpoint_sha256']==r['checkpoint_sha256'] and int(r['step'])==20000
        expected=next(s for s in read(source/'confirmation'/r['id']/'summary.json') if s['step']==20000)
        assert abs(float(r['pooled_sw2'])-expected['pooled_sw2'])<1e-10
        table.append(dict(id=r['id'],dataset=r['dataset'],family=c['family'],field_mse=float(r['field_mse']),pooled_sw2=float(r['pooled_sw2']),
            bank_mean=float(r['sw2_mean']),bank_sd=float(r['sw2_sd']),sector_l1=float(r['sector_l1']) if r['sector_l1'] else None,
            gap_error_max=float(r['gap_error_max']) if r['gap_error_max'] else None,parameters=c['parameters'],dense_macs=c['dense_macs_per_example'],
            forward_cpu_ms=1000*c['forward_cpu_seconds']['median'],heun512_cpu_seconds=c['sample_cpu_seconds']['median'],checkpoint_sha256=r['checkpoint_sha256']))
    tasks=['gmm_ring8_2d','rotated_product_bimodal_16d'];labels=['Shared P','Stage P','Free ISRU','SwiGLU'];names=['Ring8','Product16']
    def taskrows(task):
        rows=[r for r in table if r['dataset']==task]
        return [next(r for r in rows if r['family']==family) for family in ['state_midpoint4','stage_operator_midpoint4','free_isru']]+[next(r for r in rows if r['family'].startswith('swiglu'))]
    def bankvalues(id):return np.array([float(x['sw2']) for x in sorted([b for b in banks if b['id']==id],key=lambda x:int(x['bank']))])
    contrasts=[]
    for task in tasks:
        shared,released,ordinary,swiglu=taskrows(task)
        for candidate,reference,primary in [(released,shared,True),(shared,ordinary,False),(shared,swiglu,False)]:
            delta=bankvalues(candidate['id'])-bankvalues(reference['id']);mean=float(np.mean(delta));sd=float(np.std(delta,ddof=1));margin=float(stats.t.ppf(.975,7)*sd/np.sqrt(8))
            nonzero=delta[delta!=0];wins=int(np.count_nonzero(delta<0));raw=float(stats.binomtest(int(np.count_nonzero(nonzero<0)),len(nonzero),.5).pvalue)
            contrasts.append(dict(dataset=task,candidate=candidate['family'],reference=reference['family'],primary=primary,banks=8,
                bank_differences=delta.tolist(),mean_difference=mean,sd_difference=sd,approx_t95_interval=[mean-margin,mean+margin],
                candidate_better_banks=wins,pooled_difference=candidate['pooled_sw2']-reference['pooled_sw2'],
                paired_standardized_difference=mean/sd if sd else None,shapiro_p=float(stats.shapiro(delta).pvalue),conditional_sign_p=raw))
    primary=[r for r in contrasts if r['primary']]
    for r,corrected in zip(primary,holm([r['conditional_sign_p'] for r in primary])):r['conditional_sign_p_holm_two_tasks']=corrected
    dump(out/'comparison.json',dict(rows=table,contrasts=contrasts,scope='new sharing streams2026112000/2100 only; no pooled-bank/C8 mixing'))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    f,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for i,task in enumerate(tasks):
        rows=taskrows(task)
        for j,metric in enumerate(['field_mse','pooled_sw2']):
            ax=axes[i,j]
            for index,row in enumerate(rows):
                for step,offset,marker in [(15000,-.15,'x'),(20000,0.,'o')]:
                    v=next(r for r in selection if r['id']==row['id'] and int(r['step'])==step)
                    ax.scatter(index+offset,float(v[metric]),color='#888888',marker=marker,s=40,label=f'Selection {step//1000}k' if index==0 else None)
                ax.scatter(index+.15,row[metric],color='#007F78',s=45,label='Confirmation 20k' if index==0 else None)
            ax.set(xticks=np.arange(4),xticklabels=labels,ylabel='Oracle field MSE' if j==0 else 'Pooled SW2',title=names[i]);ax.set_ylim(bottom=0)
            if i==0 and j==0:ax.legend(fontsize=8)
    f.savefig(fig/'sharing-comparison.png',dpi=190);plt.close(f)
    f,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for i,task in enumerate(tasks):
        r=next(r for r in probe['rows'] if r['dataset']==task and r['family']=='stage_operator_midpoint4')
        values=[r['field_mse']]+[next(x for x in r['interventions'] if x['mode']==mode)['field_mse'] for mode in ['mean','reverse','base']]
        ax=axes[0,i];ax.bar(np.arange(4),values,color=['#007F78']+['#D55E00']*3);ax.set(xticks=np.arange(4),xticklabels=['Native','Mean P','Reverse','Base P'],yscale='log',ylabel='Oracle field MSE',title=names[i]+' / Stage P')
        layernames=sorted({v['layer'] for v in r['layers']});medians=[np.median([v['stage_drift_relative_variation'] for v in r['layers'] if v['layer']==name]) for name in layernames]
        ax=axes[1,i];ax.bar(np.arange(6),medians,color='#0072B2');ax.axhline(0,color='black',ls='--');ax.set(xticks=np.arange(6),xticklabels=[str(j+1) for j in range(6)],xlabel='Local solve',ylabel='Stage drift variation / mean drift')
    f.savefig(fig/'sharing-attribution.png',dpi=190);plt.close(f)
    f,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');colors=['#0072B2','#D55E00','#009E73','#444444']
    for ax,task,name in zip(axes,tasks,names):
        for index,(row,label,color) in enumerate(zip(taskrows(task),labels,colors)):
            x=row['heun512_cpu_seconds'];y=row['pooled_sw2'];ax.scatter(x,y,color=color,s=55)
            ax.annotate(label+f" ({row['parameters']/1000:.1f}k)",(x,y),xytext=(5,-14 if index==1 else 6),textcoords='offset points',fontsize=8)
        ax.set(xlabel='CPU seconds / 512 samples, Heun128',ylabel='Confirmation pooled SW2',title=name);ax.margins(x=.30,y=.25)
    f.savefig(fig/'sharing-cost-frontier.png',dpi=190);plt.close(f)
    f,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for i,(task,name) in enumerate(zip(tasks,names)):
        focal=[next(r for r in probe['rows'] if r['dataset']==task and r['family']==family) for family in ['state_midpoint4','stage_operator_midpoint4']]
        for r,label,color in zip(focal,labels[:2],colors[:2]):
            curve=r['training_curve'];axes[0,i].plot([p['step']/1000 for p in curve],[p['validation_oracle_mse'] for p in curve],label=label,color=color)
        left,right=focal;common=sorted(set(p['step'] for p in left['training_curve'])&set(p['step'] for p in right['training_curve'])-{0})
        loss=lambda r,step:next(p['train_pair_mse'] for p in r['training_curve'] if p['step']==step)
        axes[1,i].plot(np.array(common)/1000,[loss(right,s)-loss(left,s) for s in common],color='#D55E00');axes[1,i].axhline(0,color='black',ls='--')
        axes[0,i].set(yscale='log',ylabel='Native validation oracle MSE',title=name);axes[0,i].legend()
        axes[1,i].set(xlabel='Training updates (thousands)',ylabel='Pair loss: Stage P minus Shared P')
    f.savefig(fig/'sharing-training.png',dpi=190);plt.close(f)
    lines=['# Operator-sharing comparison','','Fresh sharing-cohort confirmation, lower is better.','', '| Task | Model | MSE | Pooled SW2 | Parameters | Dense MACs/example | CPU seconds/512 samples |','|---|---|---:|---:|---:|---:|---:|']
    for task,name in zip(tasks,names):
        for r,label in zip(taskrows(task),labels):lines.append(f"| {name} | {label} | {r['field_mse']:.3f} | {r['pooled_sw2']:.3f} | {r['parameters']:,} | {r['dense_macs']:,} | {r['heun512_cpu_seconds']:.3f} |")
    lines+=['','CPU times are process CPU time on Mac mini, not GPU throughput. Dense MACs exclude pointwise work. The two focal models have the same matrix work and different parameter counts.','', 'SwiGLU retains Ring8 width64/Muon0.02 and Product16 width192/AdamW0.0003. All focal models use AdamW0.0003.']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    dump(out/'analysis-verification.json',dict(passed=True,models=8,selection_rows=16,confirmation_rows=8,conditional_banks_per_model=8,
           primary_contrasts=2,descriptive_reference_contrasts=4,source_sha256=sha(__file__),artifacts=files))
    print(json.dumps(dict(passed=True,models=8,output=str(out))))

if __name__=='__main__':main()
