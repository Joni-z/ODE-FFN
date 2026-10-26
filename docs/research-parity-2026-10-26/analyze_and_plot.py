"""Tables and figures from the completed, independently replayed frozen study."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/Users/mr.z/ode-workspace-prep/architecture-search/parity_20261026_v2')
DOC=Path(__file__).parent
ARMS=['ode_k2_direct','swiglu128_direct','swiglu192_direct','swiglu384_direct','sigmoid_glu128']
LABELS=['Original ODE','SwiGLU128','SwiGLU192','SwiGLU384','Sigmoid-GLU128']
TIMES=[.1,.3,.5,.7,.9]
def load(p):return json.loads(p.read_text())
def csvwrite(name,rows):
    names=list(dict.fromkeys(k for r in rows for k in r))
    with (DOC/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,names);writer.writeheader();writer.writerows(rows)

def main():
    out=ROOT/'output';assert load(out/'status.json')['state']=='COMPLETE'
    audit=load(ROOT/'independent-audit/audit.json');assert audit['status']=='PASS' and audit['decisions_reproduced']
    summary=load(out/'summary.json');decision=load(out/'decision.json');local=load(out/'local-summary.json')
    pooled=[dict(arm=a,**summary['aggregate'][a]) for a in ARMS];csvwrite('field-components.csv',pooled)
    bytime=[]
    for t in TIMES:
        with np.load(out/'projections'/f'time-{t:.1f}.npz') as data:
            for j,arm in enumerate(ARMS):
                row=dict(arm=arm,time=t,**{k:float(data[k][0,j]) for k in ['total','even','linear_odd','nonlinear_odd','nonlinear_alignment']})
                ratio=float(data['nonlinear_energy_ratio'][0,j]);assert ratio>=0
                row['nonlinear_relative_RMS']=float(np.sqrt(ratio));row['oracle_nonlinear_energy']=float(data['oracle_nonlinear_energy'][0]);bytime.append(row)
    csvwrite('field-by-time.csv',bytime);csvwrite('primary-contrasts.csv',decision['contrasts']);csvwrite('activation-statistics.csv',local)
    middle=[r for r in local if r['arm']==ARMS[0] and r['site'].endswith('.middle')];csvwrite('middle-response.csv',middle)
    clocks=[]
    for row in local:
        if row['kind']=='clock':
            clocks.append(dict(arm=row['arm'],time=row['time'],site=row['site'],mean_duration=row['even_value'],
                odd_RMS=float(np.sqrt(row['odd'])),even_RMS=float(np.sqrt(row['even_square'])),odd_to_even_RMS=float(np.sqrt(row['odd']/row['even_square']))))
    csvwrite('duration-statistics.csv',clocks)
    figures=DOC/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    colors=['#D55E00','#0072B2','#009E73','#CC79A7','#777777']
    fig,axes=plt.subplots(1,2,figsize=(12.3,4.5),layout='constrained')
    x=np.arange(len(ARMS));bottom=np.zeros(len(x))
    for key,title,color in [('even','Even leakage','#999999'),('linear_odd','Linear odd error','#56B4E9'),('nonlinear_odd','Nonlinear odd error','#D55E00')]:
        value=np.array([r[key] for r in pooled]);axes[0].bar(x,value,bottom=bottom,color=color,label=title,width=.7);bottom+=value
    axes[0].set_xticks(x,LABELS,rotation=22,ha='right');axes[0].set_ylabel('Paired population field MSE');axes[0].set_title('Error is dominated by the nonlinear component')
    axes[0].legend(fontsize=9);axes[0].set_ylim(0,.365)
    for j,value in enumerate(bottom):axes[0].text(j,value+.004,f'{value:.4f}',ha='center',fontsize=9)
    for arm,label,color in zip(ARMS,LABELS,colors):
        data=[r for r in bytime if r['arm']==arm];axes[1].plot(TIMES,[r['nonlinear_odd'] for r in data],marker='o',label=label,color=color)
    axes[1].set(xlabel='Sampling time',ylabel='Nonlinear odd fitting error',title='The largest deficit is at later times',xticks=TIMES);axes[1].legend(fontsize=9)
    for ext in ['png','pdf']:fig.savefig(figures/f'field-decomposition.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.7,3.8),layout='constrained')
    for ax,key,title in zip(axes,['eta','zeta'],['Measured extra-odd response / linear baseline','Upper bound / linear baseline']):
        array=np.array([[next(r[key] for r in middle if r['time']==t and r['site']==f'b{i}.middle') for t in TIMES] for i in range(3)])
        im=ax.imshow(array,aspect='auto',cmap='viridis',vmin=0,vmax=1 if key=='eta' else 4)
        ax.set_xticks(np.arange(5),[str(t) for t in TIMES]);ax.set_yticks(np.arange(3),['Block 1','Block 2','Block 3'])
        ax.set_xlabel('Sampling time');ax.set_title(title,fontsize=11)
        for i in range(3):
            for j in range(5):ax.text(j,i,f'{array[i,j]:.2f}',ha='center',va='center',color='black' if array[i,j]>(.6 if key=='eta' else 2.4) else 'white',fontsize=10)
        fig.colorbar(im,ax=ax,shrink=.85)
    fig.suptitle('Original ODE middle SiLUs: every measured ratio exceeds the 0.10 threshold',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(figures/f'middle-SiLU-response.{ext}')
    plt.close(fig)
    sample=lambda arm:next(r for r in bytime if r['time']==.7 and r['arm']==arm)
    table='\n'.join('| '+LABELS[j]+' | '+' | '.join(f'{r[k]:.6f}' for k in ['total','even','linear_odd','nonlinear_odd'])+' |' for j,r in enumerate(pooled))
    contrasts='\n'.join(f"| {r['reference'].replace('_direct','')} | {100*r['signed_fraction']:.2f}% | [{100*r['low95']:.2f}%, {100*r['high95']:.2f}%] |" for r in decision['contrasts'] if r['component']=='nonlinear_odd')
    eta=[r['eta'] for r in middle];zeta=[r['zeta'] for r in middle];case=sample(ARMS[0]);control=sample(ARMS[2])
    report=f'''# The nonlinear fitting deficit is real; the small-SiLU-response explanation fails

The original ODE's deficit is concentrated in its nonlinear response. The actual learned middle SiLUs depart substantially from the centered linear-odd formula, so that formula does not explain the deficit. This study selects no activation replacement or new architecture.

We evaluate five frozen final predictors on the same 4096 population rows at each of five times, pairing every physical input x with exactly -x. Product16's analytic velocity is odd. The models retain their original weights, buffers and training seed 0. No model is fitted or updated in this study; the linear projections describe saved fields.

For these input pairs, total field error splits exactly into even leakage, a linear part of the odd prediction error, and its residual nonlinear part. The projection uses the raw 16 input coordinates without a target-specific nonlinear basis. These are finite-panel components, not independently trained models.

| Predictor | Total MSE | Even leakage | Linear odd error | Nonlinear odd error |
|---|---:|---:|---:|---:|
{table}

![Full error decomposition and time dependence](figures/field-decomposition.png)

Against the three SwiGLU references, the nonlinear component accounts for 99.14–100.05% of the total excess error. Signed even-leakage contributions range from -0.60% to 0.45%. A fraction above 100% means another component partly offsets the nonlinear deficit; fractions are not clipped. Merely symmetrizing this ODE field would remove its 0.012117 even leakage and leave 0.304451 error, still far above the ordinary references.

| Reference | Nonlinear share of ODE excess | Central 95% refitted resample range |
|---|---:|---:|
{contrasts}

The registered majority criterion passes for nonlinear error and fails for even leakage. These are paired row-resample ranges for fixed predictors. The nonlinear percentile ranges lie below their full-panel point estimates; they are reported as registered sensitivity ranges rather than calibrated population confidence bounds. See [statistical details](stats-appendix.md).

The local mathematical question was narrower. Centered SiLU satisfies `SiLU(d)-SiLU(-d)=d`, making its odd part linear in d. For learned preactivations `m+d` and `m-d`, an additional term R appears. We measured `eta=R_RMS/(d/2)_RMS` and its analytic upper-bound ratio zeta, retaining the actual paired centers.

All 15 middle-SiLU cases exceed the predeclared 0.10 threshold: eta ranges from{min(eta):.3f} to{max(eta):.3f}. The bound ratios range from{min(zeta):.3f} to{max(zeta):.3f}, so the small-center certificate also fails. The learned network is not close to the bias-free centered special case in this measurement. A large R does not prove useful nonlinear features: d can itself be nonlinear in the raw input, and subsequent transformations can discard or misalign a response.

![Measured and bounded departures from the centered SiLU formula](figures/middle-SiLU-response.png)

At time 0.7, after removing each field's best raw-linear component, the original ODE's coefficient along the oracle's remaining signal is{case['nonlinear_alignment']:.3f}, versus{control['nonlinear_alignment']:.3f} for SwiGLU192. Their relative residual RMS values are{case['nonlinear_relative_RMS']:.3f} and{control['nonlinear_relative_RMS']:.3f}. This describes both weak magnitude and imperfect alignment in the ODE's nonlinear prediction. The complete [time table](field-by-time.csv) includes every time; at 0.1 the oracle's nonlinear energy is tiny, making normalized ratios especially sensitive.

Sigmoid-GLU128's 0.151666 total error lies between original ODE and SwiGLU128. Its idealized zero-center product has a linear odd part, yet its complete biased, conditioned network learns a nonlinear field. Thus the local SiLU/GLU/SwiGLU algebra does not rank the complete trained predictors by itself. The stored product moments are descriptive, not interventions that attribute causal contributions to gates or centers.

The six learned ODE clocks are also not uniformly constant: paired odd/even duration RMS ratios span{min(r['odd_to_even_RMS'] for r in clocks):.4f}–{max(r['odd_to_even_RMS'] for r in clocks):.4f}. This does not identify a better clock. It further limits using the constant-clock linear-map analogy as a description of every learned layer.

The evidence narrows the design question to useful nonlinear fitting, while rejecting a simple local activation explanation. It does not justify changing SiLU, imposing exact odd symmetry, or claiming that a new ODE operator will improve images. A next architecture needs a separate mechanism linking its internal response to prediction accuracy and must still beat strong ordinary FFN controls. Product16 symmetry is a property of this diagnostic population, not an assumption about natural images.

The full [protocol](protocol.md), [component table](field-components.csv), [activation statistics](activation-statistics.csv), [primary contrasts](primary-contrasts.csv), and [figure catalog](figure-catalog.md) preserve the measurements. Manual forwards replay all 50 paired field tensors bitwise; independent weighted QR in rotated coordinates reproduces the projections and decisions. The main calculation took 6.82 seconds and its independent audit 13.73 seconds on one Mac mini CPU thread. Lenovo SSH remained unreachable, and no GPU was reserved. No training, sampling run, commit or push was launched. Earlier results and Report1 are unchanged.
'''
    (DOC/'analysis-report.md').write_text(report)
    receipt=dict(status='PASS',runtime=str(ROOT),field_rows=len(pooled),time_rows=len(bytime),local_rows=len(local),middle_rows=len(middle),clock_rows=len(clocks),
        primary_contrast_rows=len(decision['contrasts']),figures=['field-decomposition','middle-SiLU-response'],eta_range=[min(eta),max(eta)],zeta_range=[min(zeta),max(zeta)],
        architecture_selected=False)
    (DOC/'analysis-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
