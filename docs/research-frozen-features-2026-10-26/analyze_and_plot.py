"""Complete descriptive comparison of the registered frozen-feature probes."""
import argparse,collections,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ARMS=['ode_k2_direct','swiglu128_direct','swiglu192_direct','swiglu384_direct']
NAMES=['Original ODE','SwiGLU128','SwiGLU192','SwiGLU384']

def table(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(root,doc):
    out=root/'output';audit=json.loads((root/'independent-audit.json').read_text());assert audit['status']=='PASS'
    rows=json.loads((out/'rows.json').read_text());groups=collections.defaultdict(list)
    for r in rows:groups[r['arm'],r['site'],r['scope'],r['label'],r['evaluation_fold']].append(r)
    byfold=[]
    for key,values in groups.items():
        assert len(values)==5
        byfold.append(dict(zip(['arm','site','scope','label','evaluation_fold'],key))|{k:float(np.mean([v[k] for v in values])) for k in ['total','own','other']})
    lookup={(r['arm'],r['site'],r['scope'],r['label'],r['evaluation_fold']):r for r in byfold}
    pooled=collections.defaultdict(list)
    for r in byfold:pooled[r['arm'],r['site'],r['scope'],r['label']].append(r)
    aggregates=[]
    for key,values in pooled.items():
        assert len(values)==2
        aggregates.append(dict(zip(['arm','site','scope','label'],key))|{k:float(np.mean([r[k] for r in values])) for k in ['total','own','other']}|
            {'fold_min':min(r['total'] for r in values),'fold_max':max(r['total'] for r in values)})
    mean={(r['arm'],r['site'],r['scope'],r['label']):r for r in aggregates}
    contrasts=[]
    for ref in ARMS[1:]:
        for fold in [0,1]:
            native=lookup[ARMS[0],'native','native','native',fold]['total']-lookup[ref,'native','native','native',fold]['total']
            probe=lookup[ARMS[0],'post_norm','shared','mean',fold]['total']-lookup[ref,'post_norm','shared','mean',fold]['total']
            assert native>0
            contrasts.append(dict(reference=ref,evaluation_fold=fold,native_gap=native,probe_gap=probe,gap_closure=1-probe/native))
    decision=dict(primary='shared final-normalized features with mean labels',majority_accessibility_account=all(r['gap_closure']>=.5 for r in contrasts),
        architecture_selected=False,contrasts=contrasts)
    table(doc/'all-times-and-folds.csv',rows);table(doc/'fold-aggregates.csv',byfold);table(doc/'all-probes.csv',aggregates);table(doc/'primary-contrasts.csv',contrasts)
    (root/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,ax=plt.subplots(1,2,figsize=(13,5.4),gridspec_kw={'width_ratios':[1.3,1]})
    variants=[('native','native','native','Native head','#555555','o'),
        ('post_norm','shared','mean','Shared / mean labels','#D55E00','D'),
        ('post_norm','shared','pair','Shared / pair labels','#009E73','s'),
        ('post_norm','separate','mean','Separate time / mean labels','#0072B2','^')]
    yy=np.arange(4)
    for k,(site,scope,label,title,color,marker) in enumerate(variants):
        v=np.array([mean[arm,site,scope,label]['total'] for arm in ARMS]);lo=np.array([mean[arm,site,scope,label]['fold_min'] for arm in ARMS]);hi=np.array([mean[arm,site,scope,label]['fold_max'] for arm in ARMS])
        ax[0].errorbar(v,yy+(k-1.5)*.14,xerr=[v-lo,hi-v],fmt=marker,color=color,label=title,capsize=3,ms=5)
    ax[0].set_yticks(yy,NAMES);ax[0].invert_yaxis();ax[0].set_xscale('log');ax[0].set_xlabel('Five-time paired-redraw MSE');ax[0].set_title('Same frozen backbones; different affine readouts')
    ax[0].legend(frameon=False,fontsize=9,loc='lower right');ax[0].grid(axis='x',alpha=.15)
    for j,ref in enumerate(ARMS[1:]):
        vals=np.array([r['gap_closure']*100 for r in contrasts if r['reference']==ref])
        ax[1].barh(j,vals.mean(),color='#D55E00',height=.4)
        ax[1].errorbar(vals.mean(),j,xerr=[[vals.mean()-vals.min()],[vals.max()-vals.mean()]],color='black',fmt='none',capsize=3)
        ax[1].text(3,j,f'{vals.mean():.2f}%  [{vals.min():.2f}, {vals.max():.2f}]',va='center',fontsize=9)
    ax[1].axvline(50,color='.5',ls='--');ax[1].set_xlim(0,55);ax[1].set_yticks(range(3),['vs Swi128','vs Swi192','vs Swi384']);ax[1].invert_yaxis()
    ax[1].set_xlabel('Original-ODE excess removed (%)');ax[1].set_title('Primary comparison; required majority is 50%')
    fig.suptitle('A new linear head barely closes the original ODE deficit',fontsize=15)
    fig.text(.5,.015,'Whiskers and brackets span the two evaluation folds; they are not confidence intervals. Mean labels are privileged.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.95]);figures=doc/'figures';figures.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(figures/f'head-refit-comparison.{ext}',bbox_inches='tight')
    plt.close(fig)
    columns=[(site,scope,label) for site in ['input','pre_norm','post_norm'] for scope in ['shared','separate'] for label in ['mean','pair']]
    values=np.array([[mean[(arm,)+column]['total'] for column in columns] for arm in ARMS])
    fig,ax=plt.subplots(figsize=(14,5.5));im=ax.imshow(values,norm=LogNorm(vmin=values.min()*.9,vmax=values.max()*1.1),cmap='viridis_r',aspect='auto')
    for i in range(4):
        for j in range(12):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=9,color='white' if im.norm(values[i,j])>.6 else 'black')
    ax.set_yticks(range(4),NAMES);ax.set_xticks(range(12),[scope+'\n'+label for _,scope,label in columns],fontsize=8)
    for pos in [3.5,7.5]:ax.axvline(pos,color='white',lw=2)
    top=ax.secondary_xaxis('top');top.set_xticks([1.5,5.5,9.5],['Input embedding','Before final LayerNorm','After final LayerNorm'])
    ax.tick_params(axis='both',length=0);top.tick_params(length=0,pad=8)
    cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.025);cb.set_label('Five-time paired-redraw MSE (log color scale)')
    fig.suptitle('All 48 predetermined probes',fontsize=15)
    fig.text(.5,.015,'Shared: one head across time. Separate: five heads. Mean: conditional-mean labels. Pair: ordinary sampled velocity labels.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.91])
    for ext in ['png','pdf']:fig.savefig(figures/f'all-feature-probes.{ext}',bbox_inches='tight')
    plt.close(fig)
    receipt=dict(status='PASS',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        independent_audit_sha256=hashlib.sha256((root/'independent-audit.json').read_bytes()).hexdigest(),
        rows={'time_fold':len(rows),'fold':len(byfold),'pooled':len(aggregates),'contrasts':len(contrasts)},decision=decision)
    (root/'analysis-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();main(a.root,a.report)
