"""Complete descriptive curvature analysis and explicit exploratory branch screen."""
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

ROOT,DEST=map(Path,sys.argv[1:3])
sys.path.insert(0,str(ROOT/'source'))
import curvature_study as study
from curvature_math import local
from teacher_support import TEACHERS,STUDENTS,TIMES,load,write,sha


def csv_file(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (DEST/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def ratio(a,b):
    return float(a/b) if b>1e-18 else None


def fit_row(target,value,mask):
    answer={}
    for field,metric in [('value','value'),('jacobian','jacobian'),('hessian_total','hessian')]:
        a,b=value[field][mask],target[field][mask]
        error=float(np.sum((a-b)**2));energy=float(np.sum(b*b))
        answer[metric+'_error_energy']=error;answer[metric+'_target_energy']=energy
        answer[metric+'_relative_squared_error']=ratio(error,energy)
    local_error=value['hessian_local'][mask]-target['hessian_local'][mask]
    total_error=value['hessian_total'][mask]-target['hessian_total'][mask]
    embedding_error=total_error-local_error
    answer.update(local_error_energy=float(np.sum(local_error**2)),embedding_error_energy=float(np.sum(embedding_error**2)),
                  local_embedding_cross=float(2*np.sum(local_error*embedding_error)))
    np.testing.assert_allclose(answer['local_error_energy']+answer['embedding_error_energy']+answer['local_embedding_cross'],
                               answer['hessian_error_energy'],rtol=2e-10,atol=1e-12)
    return answer


def figure(fig,name):
    fig.savefig(DEST/'figures'/f'{name}.png',dpi=180);fig.savefig(DEST/'figures'/f'{name}.pdf');plt.close(fig)


torch.set_num_threads(1)
assert load(ROOT/'audit.json')['status']=='PASS'
(DEST/'figures').mkdir(exist_ok=True)
with np.load(ROOT/'output/cache/prefix.npz') as f:bank={k:f[k] for k in f.files}
approximation,anchors,grams,fits,fit_all,reduced_rows,reduced_all,branches=[],[],[],[],[],[],[],[]
for teacher in TEACHERS:
    for position in range(3):
        folder=ROOT/'output/cases'/f'{teacher}--{position}'
        arrays={}
        for key in ['target',*STUDENTS]:
            with np.load(folder/(key+'.npz')) as f:arrays[key]={k:f[k] for k in f.files}
            geometry=load(folder/(key+'-geometry.json'))
            for r in geometry:anchors.append(dict(teacher=teacher,position=position,subject=key,**r))
            if geometry:
                for time in TIMES:
                    rows=[r for r in geometry if r['time']==time]
                    den=sum(r['full_curvature_energy'] for r in rows)
                    row=dict(teacher=teacher,position=position,subject=key,time=time,full_curvature_energy=den)
                    for raw,metric in [('frozen_difference_energy','frozen'),('origin_omission_difference_energy','origin_omission')]:
                        value=sum(r[raw] for r in rows)
                        row[metric+'_difference_energy']=value;row[metric+'_relative_squared_error']=ratio(value,den)
                        row[metric+'_relative_RMS']=math.sqrt(value/den) if den>1e-18 else None
                    for raw in ['middle_energy','clock1_energy','clock2_energy','embedding_curvature_energy','local_curvature_energy']:
                        row[raw+'_ratio']=ratio(sum(r[raw] for r in rows),den)
                    approximation.append(row)
                    gram=arrays[key]['route_grams'][bank['time']==time].sum(0)
                    for i in range(4):
                        for j in range(4):grams.append(dict(teacher=teacher,position=position,subject=key,time=time,route_i=i,route_j=j,inner_product=float(gram[i,j]),relative_to_full_energy=ratio(float(gram[i,j]),den)))
        for family in STUDENTS:
            for time in TIMES:fits.append(dict(teacher=teacher,position=position,student=family,time=time,**fit_row(arrays['target'],arrays[family],bank['time']==time)))
            fit_all.append(dict(teacher=teacher,position=position,student=family,**fit_row(arrays['target'],arrays[family],np.ones(80,dtype=bool))))
        projected=load(folder/'projected-errors.json')
        for family in STUDENTS:
            for time in [None,*TIMES]:
                rows=projected if time is None else [r for r in projected if r['time']==time]
                target_energy=sum(r['teacher_retained_energy'] for r in rows)
                full_energy=sum(r['teacher_full_local_energy'] for r in rows)
                error=sum(r[family+'_error_energy'] for r in rows)
                row=dict(teacher=teacher,position=position,student=family,teacher_retained_energy=target_energy,
                         teacher_full_local_energy=full_energy,teacher_retained_fraction=ratio(target_energy,full_energy),
                         error_energy=error,relative_squared_error=ratio(error,target_energy))
                if time is None:reduced_all.append(row)
                else:reduced_rows.append(dict(time=time,**row))
        # Exploratory, declared after the registered curvature results.
        for key,(module,family) in study.subjects(teacher,position).items():
            if family!='ode_k2':continue
            matrices=[]
            for layer in [module.ode1,module.ode2]:
                p=layer.proj.weight.detach().numpy()
                values,right=np.linalg.eig(p);left_values,left_vectors=np.linalg.eig(p.T)
                order=np.array([np.argmin(np.abs(left_values-v)) for v in values])
                mu=float(np.linalg.eigvalsh((p+p.T)/2).min())
                matrices.append((p,values,right,left_vectors[:,order],mu,layer))
            with torch.no_grad():
                for anchor in range(80):
                    data=local(module,torch.from_numpy(bank['z'][anchor,position]),torch.from_numpy(bank['jacobian'][anchor,position]),family)
                    for number,(p,ev,right,left_vectors,mu,layer) in enumerate(matrices):
                        a=float(data['first' if number==0 else 'second']['a'])
                        eigenvalues=1+.5*a*(ev[:,None]+ev[None,:]);moduli=np.abs(eigenvalues)
                        i,j=np.unravel_index(np.argmin(moduli),moduli.shape)
                        h=np.outer(right[:,i],left_vectors[:,j]);h=h/np.linalg.norm(h)
                        actual=h+.5*a*(p@h+h@p)
                        residual=float(np.linalg.norm(actual-eigenvalues[i,j]*h))
                        assert residual<1e-10
                        sigmoid=(a-layer.shift)/layer.scale
                        branches.append(dict(teacher=teacher,position=position,subject=key,solve=number+1,anchor=anchor,
                            time=float(bank['time'][anchor]),duration=a,sigmoid_fraction=sigmoid,
                            logit_derivative_over_maximum=4*sigmoid*(1-sigmoid),
                            minimum_differential_eigenvalue_modulus=float(moduli.min()),
                            maximum_differential_eigenvalue_modulus=float(moduli.max()),
                            condition_number_lower_bound=float(moduli.max()/moduli.min()),
                            symmetric_minimum=mu,raw_symmetric_certificate=1+a*mu,
                            eigenmatrix_residual=residual))

exports={'clock-approximation.csv':approximation,'approximation-anchors.csv':anchors,'route-gram.csv':grams,
         'curvature-fit.csv':fits,'fit-aggregate.csv':fit_all,'reduced-fit.csv':reduced_rows,
         'reduced-aggregate.csv':reduced_all,'branch-screen.csv':branches,
         'promotion.csv':load(ROOT/'output/cache/metadata.json')['promotion']}
for name,rows in exports.items():csv_file(name,rows)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
identifiers=sorted({(r['teacher'],r['position'],r['subject']) for r in approximation})
labels=[f'{"ODE target" if t=="ode_k2" else "Swi target"} · pos {j+1} · {"teacher" if s=="target" else "student"}' for t,j,s in identifiers]
fig,axes=plt.subplots(1,2,figsize=(14.2,7),layout='constrained')
for ax,metric,title in zip(axes,['frozen','origin_omission'],['Freeze both clocks','Omit single-solve terms that vanish at zero']):
    matrix=np.array([[100*next(r for r in approximation if (r['teacher'],r['position'],r['subject'])==key and r['time']==time)[metric+'_relative_RMS'] for time in TIMES] for key in identifiers])
    maximum=max(100,float(matrix.max()))
    plot=ax.imshow(matrix,aspect='auto',norm=LogNorm(vmin=.3,vmax=maximum),cmap='viridis')
    for i in range(len(identifiers)):
        for j in range(5):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if matrix[i,j]<10 else 'black',fontsize=9)
    ax.set_xticks(range(5),[str(t) for t in TIMES]);ax.set_yticks(range(len(labels)),labels)
    ax.set_xlabel('Fixed generative time');ax.set_title(title)
    fig.colorbar(plot,ax=ax,label='Relative Hessian RMS discrepancy (%)',shrink=.8)
fig.suptitle('Approximation accuracy along actual data changes\nThe declared accuracy criterion is at most 10% RMS in every time panel',fontsize=13)
figure(fig,'clock-approximation')

colors={'ode_k2':'#D55E00','swiglu128':'#0072B2','swiglu192':'#009E73'}
labels_short={'ode_k2':'ODE','swiglu128':'Swi128','swiglu192':'Swi192'}
fig,axes=plt.subplots(2,3,figsize=(12.5,7.2),layout='constrained')
for row,teacher in enumerate(TEACHERS):
    for j in range(3):
        ax=axes[row,j]
        for family in STUDENTS:
            rows=sorted([r for r in fits if r['teacher']==teacher and r['position']==j and r['student']==family],key=lambda r:r['time'])
            ax.plot([r['time'] for r in rows],[100*math.sqrt(r['hessian_relative_squared_error']) for r in rows],marker='o',label=labels_short[family],color=colors[family])
        ax.set_yscale('log');ax.set_xlabel('Generative time');ax.set_ylabel('Relative Hessian RMS error (%)')
        ax.set_title(f'{labels_short[teacher]} teacher · position {j+1}');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=9)
fig.suptitle('Complete curvature of each local module composed with the common prefix',fontsize=13)
figure(fig,'data-composed-fit')

fig,axes=plt.subplots(2,3,figsize=(12.5,7),layout='constrained')
for row,teacher in enumerate(TEACHERS):
    for j in range(3):
        ax=axes[row,j]
        rows=[next(r for r in reduced_all if r['teacher']==teacher and r['position']==j and r['student']==f) for f in STUDENTS]
        values=[100*math.sqrt(r['relative_squared_error']) for r in rows]
        ax.bar(range(3),values,color=[colors[f] for f in STUDENTS])
        for i,value in enumerate(values):ax.annotate(f'{value:.2f}%',(i,value),xytext=(0,4),textcoords='offset points',ha='center')
        ax.set_xticks(range(3),[labels_short[f] for f in STUDENTS]);ax.set_ylabel('Reduced local Hessian RMS error (%)')
        ax.set_title(f'{labels_short[teacher]} teacher · position {j+1}\nTeacher energy retained: {100*rows[0]["teacher_retained_fraction"]:.1f}%')
        ax.set_ylim(0,max(values)*1.25);ax.grid(axis='y',alpha=.2)
fig.suptitle('The exact reduced identity does not remove the observed fitting gap\nFrames come from the ODE student; no target-dependent direction optimization',fontsize=13)
figure(fig,'reduced-local-fit')

receipt=dict(status='COMPLETE',audit_sha256=sha(ROOT/'audit.json'),source_sha256=sha(__file__),
             csv_rows={name:len(rows) for name,rows in exports.items()},figures=['clock-approximation','data-composed-fit','reduced-local-fit'])
write(DEST/'analysis-receipt.json',receipt)
print(receipt)
print('aggregate fit',[(r['teacher'],r['position'],r['student'],round(100*math.sqrt(r['hessian_relative_squared_error']),3)) for r in fit_all])
print('branch',{'min_eigenmod':min(r['minimum_differential_eigenvalue_modulus'] for r in branches),
               'max_condition_lower_bound':max(r['condition_number_lower_bound'] for r in branches),
               'severe_screen_rows':sum(r['minimum_differential_eigenvalue_modulus']<.01 and r['condition_number_lower_bound']>100 for r in branches),
               'max_eigenmatrix_residual':max(r['eigenmatrix_residual'] for r in branches)})
