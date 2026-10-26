"""All six audited endpoints, learning curves and original P trajectories."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT=Path('/Users/mr.z/ode-workspace-prep/architecture-search/realizable_teacher_20261026');DOC=Path(__file__).parent
TEACHERS=['ode_k2','swiglu128'];STUDENTS=['ode_k2','swiglu128','swiglu192'];LABELS={'ode_k2':'Original ODE','swiglu128':'SwiGLU128','swiglu192':'SwiGLU192'}
def load(p):return json.loads(p.read_text())
def csvwrite(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (DOC/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)

def main():
    audit=ROOT/'independent-audit';assert load(audit/'audit.json')['status']=='PASS';results=load(audit/'results.json');curves=load(audit/'learning-curves.json');ptraces=load(audit/'P-trajectories.json');decision=load(audit/'decision.json')
    out=ROOT/'output';meta=load(out/'cache/metadata.json');summary=[];panels=[];training=[];teachers=[]
    for r in results:
        row={k:v for k,v in r.items() if k not in ['panels','actual_replay_digests']};summary.append(row)
        for p in r['panels']:panels.append(dict(teacher=r['teacher'],student=r['student'],**p))
        for line in (out/'runs'/(r['teacher']+'--'+r['student'])/'metrics.jsonl').read_text().splitlines():training.append(dict(teacher=r['teacher'],student=r['student'],**json.loads(line)))
    for name,values in meta['teacher_energy'].items():teachers.append(dict(teacher=name,field_energy=values['field_energy'],mean_panel_jacobian_energy=float(np.mean([r['jacobian_energy'] for r in values['panels']]))))
    csvwrite('comparison.csv',summary);csvwrite('all-panels.csv',panels);csvwrite('learning-curves.csv',curves);csvwrite('training-metrics.csv',training);csvwrite('P-trajectories.csv',ptraces);csvwrite('teacher-energy.csv',teachers)
    figdir=DOC/'figures';figdir.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15.6,4.1),layout='constrained');matrix=np.array([[100*next(r['relative_teacher_mse'] for r in results if r['teacher']==t and r['student']==s) for s in STUDENTS] for t in TEACHERS])
    image=axes[0].imshow(matrix,aspect='auto',norm=LogNorm(vmin=.2,vmax=20),cmap='viridis')
    axes[0].set_xticks(range(3),['ODE','Swi128','Swi192']);axes[0].set_yticks(range(2),['ODE teacher','Swi128 teacher']);axes[0].set_title('Final relative teacher MSE (%)')
    for i in range(2):
        for j in range(3):axes[0].text(j,i,f'{matrix[i,j]:.2f}%',ha='center',va='center',color='black' if matrix[i,j]>5 else 'white',fontsize=11)
    bar=fig.colorbar(image,ax=axes[0],shrink=.8);bar.set_label('% (log color scale)')
    colors=['#D55E00','#0072B2','#009E73']
    for ax,t in zip(axes[1:],TEACHERS):
        for s,color in zip(STUDENTS,colors):
            points=[r for r in curves if r['teacher']==t and r['student']==s]
            ax.semilogy([r['step'] for r in points],[100*r['relative_teacher_mse'] for r in points],marker='o',color=color,label=LABELS[s])
        ax.set(xlabel='Student updates',ylabel='Relative teacher MSE (%)',title=LABELS[t]+' teacher',ylim=(.2,130));ax.legend(fontsize=8)
    for ext in ['png','pdf']:fig.savefig(figdir/f'teacher-fitting.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.7,4),layout='constrained');layers=list(dict.fromkeys(r['layer'] for r in ptraces))
    for ax,t in zip(axes,TEACHERS):
        for name in layers:
            data=[r for r in ptraces if r['teacher']==t and r['layer']==name]
            parts=name.split('.');label=f"B{int(parts[1])+1} S{parts[-1][-1]}"
            ax.plot([r['step'] for r in data],[r['spectral_norm'] for r in data],marker='o',label=label)
        ax.set(xlabel='ODE student updates',ylabel='Raw P spectral norm',title=LABELS[t]+' teacher');ax.legend(fontsize=8,ncol=2)
    for ext in ['png','pdf']:fig.savefig(figdir/f'P-trajectories.{ext}',dpi=180)
    plt.close(fig)
    get=lambda t,s:next(r for r in results if r['teacher']==t and r['student']==s)
    own=get('ode_k2','ode_k2');hard=get('swiglu128','ode_k2');s128=get('swiglu128','swiglu128');s192=get('swiglu128','swiglu192')
    rows='\n'.join(f"| {LABELS[t]} | {LABELS[s]} | {get(t,s)['teacher_mse']:.6f} | {100*get(t,s)['relative_teacher_mse']:.3f}% | {get(t,s)['mean_panel_jacobian_error']:.4f} |" for t in TEACHERS for s in STUDENTS)
    teacher_table='\n'.join(f"| {LABELS[r['teacher']]} | {r['field_energy']:.4f} | {r['mean_panel_jacobian_energy']:.4f} |" for r in teachers)
    monitor=[json.loads(line) for line in (out/'monitor.jsonl').read_text().splitlines()];peak_rss=max(sum(p['rss_bytes'] for p in row['processes']) for row in monitor)/2**30
    total_seconds=load(out/'status.json')['elapsed_seconds'];audit_seconds=load(audit/'audit.json')['elapsed_seconds']
    last_clips={(t,s):next(r['clip_fraction'] for r in reversed(training) if r['teacher']==t and r['student']==s and r['step']==5000) for t in TEACHERS for s in STUDENTS}
    report=f'''# A cold ODE learns its own teacher, but fits the SwiGLU teacher poorly

The original ODE student reaches {100*own['relative_teacher_mse']:.2f}% relative squared error on a frozen teacher of its own architecture. On the SwiGLU teacher it remains at {100*hard['relative_teacher_mse']:.2f}%, compared with {100*s128['relative_teacher_mse']:.2f}% for SwiGLU128 and {100*s192['relative_teacher_mse']:.2f}% for SwiGLU192. The experiment establishes useful within-function contrast, but does not identify whether the harder target exceeds the ODE's finite function class or is harder to optimize.

This was six new cold seed-0 regressions on the same Product16 input stream, with unchanged architectures, 5000 updates, batch512, AdamW3e-4, weight_decay0 and clip1. The targets were complete frozen teacher outputs. The ODE teacher's weights and actual buffers are an explicit zero-risk representer in the ODE student's class; they were restored only in a separate procedural control, never copied into a cold student.

| Teacher | Cold student | Teacher MSE | Relative teacher MSE | Mean panel Jacobian error |
|---|---|---:|---:|---:|
{rows}

Teacher MSE uses all 20000 fixed uniform-time validation inputs. Relative MSE divides by the corresponding teacher's mean squared output. It is a squared-error percentage, not relative RMS error or a generation-quality score. Derivative error averages all five fixed-time512-input panels; all time-specific rows are in [all-panels.csv](all-panels.csv).

![Every teacher/student endpoint and the complete saved validation curves](figures/teacher-fitting.png)

The predeclared severe learning-difficulty pattern does **not** pass. The ODE self-teacher error is below the5% cutoff. In addition, same-budget SwiGLU128 is better but does not halve that self-teacher error. The near-compute SwiGLU192 does approximately halve it. All controls remain visible rather than selecting the more favorable comparison. On the SwiGLU teacher, the ODE's error is {hard['teacher_mse']/s128['teacher_mse']:.1f} times SwiGLU128's and {hard['teacher_mse']/s192['teacher_mse']:.1f} times SwiGLU192's. That is a measured fitting gap, not an impossibility theorem.

The teachers have different local variation. Their output energies are similar, but their fixed-panel derivative energies differ:

| Frozen teacher | Uniform-time field energy | Mean panel Jacobian energy |
|---|---:|---:|
{teacher_table}

These energies do not define a complete complexity measure or causal explanation. They prevent treating the targets as equally demanding merely because one can normalize their amplitudes. The larger SwiGLU192 student also fits the SwiGLU128 teacher worse than the same-width student at this endpoint, which limits a simple parameter-count explanation.

The two ODE students change all six P matrices during fitting. Their complete saved norm trajectories are below; [P-trajectories.csv](P-trajectories.csv) also retains Frobenius norms and symmetric minima. Different teacher risks coexist with modest P norms. Those internal norms do not certify full-network stability or determine the achievable nonlinear field.

![All six original P trajectories under each teacher](figures/P-trajectories.png)

The final-window clipping fractions are {last_clips['ode_k2','ode_k2']:.3f} on the ODE teacher and {last_clips['swiglu128','ode_k2']:.3f} on the SwiGLU teacher for ODE students. This describes the recorded last500 updates; it neither characterizes every earlier step nor selects a clipping or optimizer intervention.

The realizable control excludes the absence of the ODE teacher from its own hypothesis class as an explanation for residual error on that function. Its accurate self-fit does not prove that the analytic population target or the SwiGLU teacher is representable by the same finite ODE. Earlier [parity](../research-parity-2026-10-26/analysis-report.md) and [P-tangent](../research-generator-tangent-2026-10-26/analysis-report.md) measurements remain separate: a nonlinear-fitting deficit was observed, while a small centered-SiLU response and severe K2 derivative-route cancellation were not supported.

This study does not nominate a new architecture, use distillation as a rescue method, or change the ordinary FM acceptance criteria. It leaves a more specific question: what structural limitation or learning mechanism makes the ODE predictor lose accuracy on the more strongly varying target? A proposed answer still needs a mathematically specified operator, an isolated cold comparison, strong generic controls, and ordinary FM/transfer/cost evidence.

All six cells finished in {total_seconds:.1f} seconds including cache preparation, using two one-thread Mac CPU workers. The largest observed sum of parent/child RSS was {peak_rss:.2f} GiB. One training-label pass per teacher reduced batch-label evaluations from30000 to10000; no GPU was reserved. Lenovo SSH remained unreachable. The independent audit took {audit_seconds:.1f} seconds and replayed the full input/label stream, all24 checkpoints and all30 native endpoint panels. Separate explicit float64 forwards and Jacobians agree exactly. Submitted Report1 and every earlier runtime remain unchanged; no commit or push occurred.

The [protocol](protocol.md), [statistical and audit scope](stats-appendix.md), [comparison table](comparison.csv), [training metrics](training-metrics.csv), [figure catalog](figure-catalog.md), and [execution notes](notes.md) preserve the evidence. The useful-ODELayer objective remains active and unachieved.
'''
    # Keep prose readable while preserving names such as SwiGLU128.
    for a,b in {'batch512':'batch 512','AdamW3e-4':'AdamW 3e-4','weight_decay0':'weight_decay 0','clip1':'clip 1','fixed-time512':'fixed-time 512','the5%':'the 5%',
                'last500':'last 500','from30000':'from 30000','to10000':'to 10000','all24':'all 24','all30':'all 30'}.items():report=report.replace(a,b)
    (DOC/'analysis-report.md').write_text(report)
    receipt=dict(status='PASS',comparison_rows=6,panel_rows=len(panels),learning_rows=len(curves),training_rows=len(training),P_checkpoint_rows=len(ptraces),teacher_rows=2,
        registered_learning_difficulty_pattern=decision['registered_learning_difficulty_pattern'],own_teacher_relative_MSE=own['relative_teacher_mse'],
        other_teacher_relative_MSE=hard['relative_teacher_mse'],total_study_seconds=total_seconds,maximum_observed_process_RSS_GiB=peak_rss,figures=2,architecture_selected=False)
    (DOC/'analysis-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
