"""Figures and complete tables for the audited response diagnosis."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS=['ODE direct','ODE potential','Swi128 direct','Swi128 potential',
        'Swi192 direct','Swi192 potential','Swi384 direct']

def table(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(root,report):
    output=root/'output';raw=Path('/Users/mr.z/ode-workspace-prep/architecture-search/coordinate_dependence_20261026/output-v2')
    assert json.loads((root/'independent-verification.json').read_text())['status']=='PASS'
    d=json.loads((output/'summary.json').read_text());m=d['means'];arms=d['arms']
    pop=json.loads((raw/'population.json').read_text());mu=np.array(pop['locations']);var=np.array(pop['variances'])
    figures=report/'figures';figures.mkdir(exist_ok=True)
    rows=[]
    for j,arm in enumerate(arms):
        rows.append(dict(arm=arm,total=m['total'][j],conditional=m['own'][j],other=m['other'][j],
            affine_reduction=m['affine'][j],full_reduction=m['full'][j],
            conditional_after_full=m['own'][j]-m['full'][j],total_after_full=m['total'][j]-m['full'][j]))
    table(report/'aggregate.csv',rows);table(report/'contrasts.csv',d['decision']['comparisons'])
    times=[]
    for row in d['rows']:
        for j,arm in enumerate(arms):times.append(dict(time=row['time'],arm=arm,**{k:row[k][j] for k in ['total','own','other','affine','full']}))
    table(report/'all-times.csv',times)
    co_rows=[]
    for t in [.1,.3,.5,.7,.9]:
        dt=(1-t)**2+t*t*var;sd=np.sqrt(dt+t*t*mu*mu)
        alpha=(t*var-(1-t))/dt;beta=(1-t)*mu/dt
        with np.load(output/'fits'/f'time-{t:.1f}.npz') as f:co=f['full_coefficients']
        for i in range(16):
            for fold in range(2):
                for j,arm in enumerate(arms):
                    intercept,linear,nonlinear=co[i,fold,:,j]
                    co_rows.append(dict(time=t,coordinate=i,training_fold=fold,arm=arm,
                        error_intercept=intercept,error_linear_physical=linear/sd[i],error_nonlinear=nonlinear,
                        fitted_field_linear=alpha[i]+linear/sd[i],fitted_field_nonlinear=beta[i]+nonlinear,
                        oracle_linear=alpha[i],oracle_nonlinear=beta[i]))
    table(report/'all-coefficients.csv',co_rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,ax=plt.subplots(1,2,figsize=(13,5.1),gridspec_kw={'width_ratios':[1.6,1]})
    ys=np.arange(7)
    for key,label,color,marker in [('total','Raw error','#555555','o'),('affine','After affine fit','#0072B2','s'),('full','After full fit','#D55E00','D')]:
        values=np.array(m['total']) if key=='total' else np.array(m['total'])-np.array(m[key])
        ax[0].scatter(values,ys,label=label,color=color,marker=marker,s=35,zorder=3)
    ax[0].set_yticks(ys,LABELS);ax[0].invert_yaxis();ax[0].set_xscale('log')
    ax[0].set_xlabel('Five-time paired-redraw MSE');ax[0].set_title('Out-of-fold diagnostic correction')
    ax[0].legend(frameon=False,fontsize=9,loc='lower right');ax[0].grid(axis='x',alpha=.15)
    rr=d['decision']['comparisons'];y=np.arange(3)
    v=np.array([r['incremental_share'] for r in rr]);lo=np.array([r['incremental_lower'] for r in rr]);hi=np.array([r['incremental_upper'] for r in rr])
    ax[1].errorbar(v*100,y,xerr=np.array([v-lo,hi-v])*100,fmt='o',color='#D55E00',capsize=4)
    ax[1].set_yticks(y,['vs Swi128','vs Swi192','vs Swi384']);ax[1].invert_yaxis();ax[1].axvline(100,color='.5',ls=':')
    ax[1].set_xlabel('Incremental nonlinear share of ODE excess (%)');ax[1].set_title('Paired, refitted bootstrap intervals')
    ax[1].set_ylim(2.4,-.4);ax[1].set_xlim(99.8,104.4);ax[1].set_xticks([100,101,102,103,104])
    for i,value in enumerate(v):ax[1].annotate(f'{100*value:.2f}%',(100*value,i),xytext=(0,-17),textcoords='offset points',ha='center',fontsize=9)
    fig.suptitle('The large original-ODE deficit follows the posterior-response basis',fontsize=14)
    fig.text(.5,.015,'Corrections use known distribution information. These are diagnostic residuals, not trained-model results.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.07,1,.95])
    for ext in ['png','pdf']:fig.savefig(figures/f'response-localization.{ext}',bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(13,10),sharex=True)
    bins=np.linspace(-4,4,17);centers=(bins[:-1]+bins[1:])/2;bin_rows=[]
    for row,t in enumerate([.3,.5,.7]):
        dt=(1-t)**2+t*t*var;sd=np.sqrt(dt+t*t*mu*mu);k=t*mu/dt
        with np.load(raw/'banks'/f'time-{t:.1f}.npz') as bank:q=bank['A'];oracle=bank['oracle']
        with np.load(output/'fits'/f'time-{t:.1f}.npz') as f:fitted=f['full']
        for col,i in enumerate([0,7,15]):
            a=axes[row,col];z=q[:,i]/sd[i];counts,_=np.histogram(z,bins)
            countaxis=a.twinx();countaxis.bar(centers,counts,width=.46,color='.85',alpha=.6,zorder=0)
            countaxis.set_ylim(0,max(counts)*5);countaxis.tick_params(axis='y',labelsize=7,colors='.6')
            if col==2:countaxis.set_ylabel('Bin count (gray)',fontsize=8,color='.5')
            else:countaxis.set_yticklabels([])
            a.set_zorder(countaxis.get_zorder()+1);a.patch.set_visible(False)
            a.axhline(0,color='black',lw=.8,ls=':')
            for j,color,label in [(0,'#D55E00','ODE direct'),(4,'#0072B2','Swi192 direct')]:
                with np.load(raw/'fields'/arms[j]/f'time-{t:.1f}.npz') as f:
                    g=((f['native'][:,i]+f['hybrid_diagonal'][:,i])/2)-oracle[:,i]
                values=[];fitted_values=[]
                for b in range(16):
                    mask=(z>=bins[b])&(z<bins[b+1] if b<15 else z<=bins[b+1])
                    n=int(mask.sum());mean=float(g[mask].mean()) if n else np.nan
                    fitted_mean=float(fitted[mask,i,j].mean()) if n else np.nan
                    values.append(mean);fitted_values.append(fitted_mean)
                    bin_rows.append(dict(time=t,coordinate=i,arm=arms[j],left=bins[b],right=bins[b+1],count=n,mean_error=mean,mean_out_of_fold_fit=fitted_mean))
                a.plot(centers,fitted_values,color=color,lw=1.3,label=label+' basis fit')
                a.scatter(centers,values,s=14,color=color,zorder=5)
            a.set_xlim(-4,4);a.set_title(f't={t:.1f}, coordinate {i}, mu={mu[i]:.2f}; n={counts.sum()}',fontsize=9)
            if col==0:a.set_ylabel('Conditional velocity error')
            if row==2:a.set_xlabel('q / population SD(q)')
    axes[0,0].legend(frameon=False,fontsize=8,loc='lower left')
    fig.suptitle('Preselected conditional-response case studies',fontsize=15)
    fig.text(.5,.012,'Dots: binned paired errors. Lines: binned out-of-fold fits. No curves extrapolate into empty bins.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.04,1,.96])
    for ext in ['png','pdf']:fig.savefig(figures/f'conditional-cases.{ext}',bbox_inches='tight')
    plt.close(fig);table(report/'case-bins.csv',bin_rows)
    receipt=dict(status='PASS',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        independent_verification_sha256=hashlib.sha256((root/'independent-verification.json').read_bytes()).hexdigest(),
        rows={'aggregates':len(rows),'times':len(times),'coefficients':len(co_rows),'case_bins':len(bin_rows)})
    (root/'plot-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();main(a.root,a.report)
