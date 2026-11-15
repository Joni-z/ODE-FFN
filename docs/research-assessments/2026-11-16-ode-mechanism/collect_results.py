"""Curate both completed mechanism stages; never treat partial scores as final."""
import csv,hashlib,json,math,shutil
from pathlib import Path
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=Path('/Users/mr.z/ode-workspace-prep/coefficient-20261116')

def read(p):return list(csv.DictReader(p.open()))
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=HERE/'results';out.mkdir(exist_ok=True);rows={};banks={};bindings={}
    for phase in ['analysis','state-analysis']:
        p=ROOT/phase
        for n in ['COMPLETE.json','verification.json','confirmation-verification.json']:
            assert json.loads((p/n).read_text())['passed'],(phase,n)
        for split in ['selection','confirmation']:
            rows.setdefault(split,[]).extend(read(p/(split+'-metrics.csv')))
            banks.setdefault(split,[]).extend(read(p/(split+'-banks.csv')))
        target=out/phase;target.mkdir(exist_ok=True)
        for n in ['COMPLETE.json','verification.json','confirmation-verification.json','selection-metrics.csv','selection-banks.csv','confirmation-metrics.csv','confirmation-banks.csv']:
            shutil.copyfile(p/n,target/n);bindings[phase+'/'+n]=sha(target/n)
    refinement=json.loads((ROOT/'state-analysis/refinement.json').read_text());assert refinement['passed']
    dump(out/'refinement.json',refinement)
    datasets=['gmm_ring8_2d','rotated_product_bimodal_16d']
    families=['isru_k1','isru_k8','free_coeff','state_isru','free_isru','swiglu']
    names=['Fixed K1','Fixed exponential K8','Free coefficients K8','State flow / midpoint4','Free ISRU','SwiGLU']
    def row(split,d,f,step=20000):
        return next(r for r in rows[split] if r['dataset']==d and int(r['step'])==step and
                    ((r['response']=='isru' and int(r['order'])==int(f[-1])) if f.startswith('isru_k') else
                     r['response'].startswith('swiglu') if f=='swiglu' else r['response']==f))
    lines=['# Completed mechanism comparison','',
           'Seed0;20k native training updates; new prebound selection/confirmation streams.',
           'All sampling uses Heun128, eight4096-sample banks and256directions.',
           'Displayed scores have three decimals; CSV/JSON retain full precision.','',
           '| Task | Model | Selection MSE | Selection SW2 | Confirmation MSE | Confirmation SW2 |',
           '|---|---|---:|---:|---:|---:|']
    for d in datasets:
        for f,name in zip(families,names):
            left,right=row('selection',d,f),row('confirmation',d,f)
            values=[float(r[k]) for r in [left,right] for k in ['field_mse','pooled_sw2']]
            lines.append('| '+('Ring8' if d==datasets[0] else 'Product16')+' | '+name+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines+=['','Ring8 SwiGLU64 uses Muon0.02; Product16 SwiGLU192 uses AdamW0.0003.',
            'All ODE/free-ISRU variants use AdamW0.0003. Comparisons to SwiGLU',
            'benchmark those labeled configurations, not isolated optimizer effects.',
            'Free coefficients add42scalars; state-flow and fixed ODE share parameter counts.','',
            '![Selection and confirmation](../2026-11-16-ode-mechanism/figures/final-comparison.png)']
    effects=[]
    for d in datasets:
        for candidate,reference in [('free_coeff','isru_k8'),('state_isru','isru_k8'),('state_isru','free_coeff'),('isru_k8','swiglu'),('free_coeff','swiglu'),('state_isru','swiglu')]:
            c,b=row('confirmation',d,candidate),row('confirmation',d,reference)
            values=lambda r:np.array([float(x['sw2']) for x in sorted(banks['confirmation'],key=lambda x:int(x['bank'])) if x['id']==r['id']])
            delta=values(c)-values(b);assert len(delta)==8
            mean=float(delta.mean());sd=float(delta.std(ddof=1));half=float(t.ppf(.975,7)*sd/math.sqrt(8))
            effects.append(dict(dataset=d,candidate=candidate,reference=reference,paired_delta=delta.tolist(),mean=mean,sd=sd,
                 conditional_mean_ci95=[mean-half,mean+half],banks_improved=int((delta<0).sum()),
                 pooled_relative_change=float(c['pooled_sw2'])/float(b['pooled_sw2'])-1))
    dump(out/'paired-effects.json',effects)
    lines+=['','Paired intervals in paired-effects.json describe conditional bank variation,',
            'not training variability or uncertainty of the pooled statistic.',
            'All four new trajectories and both independent metric checks completed.']
    (HERE/'completed-comparison.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for col,d in enumerate(datasets):
        for ri,metric in enumerate(['field_mse','pooled_sw2']):
            ax=axes[ri,col]
            for i,(split,color,marker) in enumerate([('selection','#999999','o'),('confirmation','#176e82','D')]):
                ax.plot(np.arange(6)+(-.1 if i==0 else .1),[float(row(split,d,f)[metric]) for f in families],
                        marker=marker,linestyle='none',color=color,label=split.capitalize())
            ax.set_xticks(np.arange(6),['Fixed K1','Fixed K8','Free coeff. K8','State flow','Free ISRU','SwiGLU'],rotation=30,ha='right')
            ax.set_title(('Ring8' if col==0 else 'Product16')+' /20k');ax.set_ylabel('Field MSE' if ri==0 else 'Pooled SW2');ax.grid(axis='y',alpha=.2)
            if ri==0 and col==0:ax.legend(frameon=False)
    (HERE/'figures').mkdir(exist_ok=True);fig.savefig(HERE/'figures/final-comparison.png',dpi=180);plt.close(fig)
    dump(HERE/'results-completion.json',dict(passed=True,selection_rows=len(rows['selection']),confirmation_rows=len(rows['confirmation']),artifacts=bindings,
          derived={'paired-effects':sha(out/'paired-effects.json'),'refinement':sha(out/'refinement.json')},collector_sha256=sha(Path(__file__))))
    print(json.dumps(dict(passed=True,selection_rows=len(rows['selection']),confirmation_rows=len(rows['confirmation']))))

if __name__=='__main__':main()
