"""Complete descriptive tables and all-site visualizations."""
import argparse,collections,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize
from matplotlib.patches import Rectangle

ARMS=['ode_k2_direct','swiglu128_direct','swiglu192_direct','swiglu384_direct']
NAMES=['Original ODE','SwiGLU128','SwiGLU192','SwiGLU384']

def table(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(root,doc):
    out=root/'output';audit=json.loads((root/'independent-audit.json').read_text());assert audit['status']=='PASS'
    raw=json.loads((out/'rows.json').read_text());groups=collections.defaultdict(list)
    for row in raw:groups[row['arm'],row['site'],row['label'],row['evaluation_fold']].append(row)
    byfold=[]
    for key,rows in groups.items():
        assert len(rows)==5
        byfold.append(dict(zip(['arm','site','label','evaluation_fold'],key))|{m:float(np.mean([r[m] for r in rows])) for m in ['total','own','other']})
    folds={(r['arm'],r['site'],r['label'],r['evaluation_fold']):r for r in byfold};gg=collections.defaultdict(list)
    for r in byfold:gg[r['arm'],r['site'],r['label']].append(r)
    pooled=[]
    for key,rows in gg.items():
        assert len(rows)==2
        pooled.append(dict(zip(['arm','site','label'],key))|{m:float(np.mean([r[m] for r in rows])) for m in ['total','own','other']}|
            {'fold_min':min(r['total'] for r in rows),'fold_max':max(r['total'] for r in rows)})
    lookup={(r['arm'],r['site'],r['label']):r for r in pooled};contrasts=[]
    for site in ['b2.clock1','b2.clock2']:
        for ref in ARMS[1:]:
            for fold in [0,1]:
                initial=folds['ode_k2_direct','post_norm','mean',fold]['total'];generic=folds[ref,'post_norm','mean',fold]['total']
                current=folds['ode_k2_direct',site,'mean',fold]['total'];gap=initial-generic;assert gap>0
                contrasts.append(dict(site=site,reference=ref,evaluation_fold=fold,initial_risk=initial,reference_risk=generic,controller_risk=current,gap=gap,closure=(initial-current)/gap))
    decision=dict(late_controller_reservoir_account=all(r['closure']>=.5 for r in contrasts),architecture_selected=False,
        primary_sites=['b2.clock1','b2.clock2'],contrasts=contrasts)
    (root/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    table(doc/'all-times-and-folds.csv',raw);table(doc/'fold-aggregates.csv',byfold);table(doc/'all-sites.csv',pooled);table(doc/'primary-contrasts.csv',contrasts)
    figures=doc/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'savefig.dpi':180})
    local=['z','clock1','solve1','increment1','middle','clock2','increment2','ffn','residual']
    titles=['FFN\ninput','Controller1\nSiLU','Solve1\noutput','Solve1\nincrement','Middle\nSiLU','Controller2\nSiLU','Solve2\nincrement','FFN\noutput','Residual\nstate']
    ode={label:np.array([[lookup['ode_k2_direct',f'b{i}.{site}',label]['total'] for site in local] for i in range(3)]) for label in ['mean','pair']}
    norm=Normalize(vmin=min(x.min() for x in ode.values()),vmax=max(x.max() for x in ode.values()))
    fig,axes=plt.subplots(2,1,figsize=(14,7.4),layout='constrained')
    for ax,label in zip(axes,['mean','pair']):
        values=ode[label];im=ax.imshow(values,cmap='viridis_r',norm=norm,aspect='auto')
        for i in range(3):
            for j in range(9):ax.text(j,i,f'{values[i,j]:.4f}',ha='center',va='center',fontsize=9,color='white' if norm(values[i,j])>.6 else 'black')
        ax.set_yticks(range(3),['Block1','Block2','Block3']);ax.set_xticks(range(9),titles,fontsize=8);ax.tick_params(length=0)
        for j in [1,5]:ax.add_patch(Rectangle((j-.5,1.5),1,1,fill=False,edgecolor='#E69F00',lw=3))
        ax.set_title('Conditional-mean calibration' if label=='mean' else 'Ordinary pair-label calibration',fontsize=11)
        initial=lookup['ode_k2_direct','input',label]['total'];final=lookup['ode_k2_direct','post_norm',label]['total']
        ax.set_xlabel(f'Other fixed sites: input embedding {initial:.4f}; final normalized feature {final:.4f}',fontsize=9)
    fig.colorbar(im,ax=axes,shrink=.85,label='Five-time paired-redraw MSE')
    fig.suptitle('Original ODE: no large residual recovery at the measured internal sites',fontsize=14)
    fig.get_layout_engine().set(h_pad=.12,w_pad=.12)
    for ext in ['png','pdf']:fig.savefig(figures/f'ODE-internal-sites.{ext}',bbox_inches='tight')
    plt.close(fig)
    shared=['input']+[f'b{i}.{s}' for i in range(3) for s in ['z','ffn','residual']]+['post_norm']
    labels=['Input']+[f'B{i+1}\n'+s for i in range(3) for s in ['FFN in','FFN out','Residual']]+['Final\nnorm']
    values={label:np.array([[lookup[arm,site,label]['total'] for site in shared] for arm in ARMS]) for label in ['mean','pair']}
    norm=LogNorm(vmin=min(x.min() for x in values.values())*.95,vmax=max(x.max() for x in values.values())*1.05)
    fig,axes=plt.subplots(2,1,figsize=(14,8),layout='constrained')
    for ax,label in zip(axes,['mean','pair']):
        matrix=values[label];im=ax.imshow(matrix,cmap='viridis_r',norm=norm,aspect='auto')
        for i in range(4):
            for j in range(11):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if norm(matrix[i,j])>.65 else 'black')
        ax.set_xticks(range(11),labels,fontsize=8);ax.set_yticks(range(4),NAMES);ax.tick_params(length=0)
        ax.set_title('Conditional-mean calibration' if label=='mean' else 'Ordinary pair-label calibration',fontsize=11)
    fig.colorbar(im,ax=axes,shrink=.85,label='Five-time paired-redraw MSE (log color scale)')
    fig.suptitle('All corresponding 64-channel residual probes; native predictions remain fixed offsets',fontsize=14)
    for ext in ['png','pdf']:fig.savefig(figures/f'common-sites.{ext}',bbox_inches='tight')
    plt.close(fig)
    receipt=dict(status='PASS',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        independent_audit_sha256=hashlib.sha256((root/'independent-audit.json').read_bytes()).hexdigest(),
        table_rows={'raw':len(raw),'fold':len(byfold),'pooled':len(pooled),'contrasts':len(contrasts)},decision=decision)
    (root/'analysis-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();main(a.root,a.report)
