"""Complete-cohort summaries and plots, after independent native/AD audit."""
import argparse,csv,json,shutil,sys
from pathlib import Path
import numpy as np

def table(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(x):
    a=np.asarray(x,dtype=np.float64).reshape(-1);a=a[np.isfinite(a)]
    return dict(zip(('min','p05','median','p95','max'),(float(v) for v in np.quantile(a,(0,.05,.5,.95,1))))) if len(a) else dict.fromkeys(('min','p05','median','p95','max'))
def describe(arrays):
    values={};a=arrays['sv_full'];values['full_gain']=a[:,0];values['full_smallest']=a[:,-1]
    if 'sv_ref' in arrays:
        r=arrays['sv_ref'];values['reference_gain']=r[:,0];values['reference_smallest']=r[:,-1]
        valid=r[:,0]>0;values['full_to_reference']=np.divide(a[:,0],r[:,0],out=np.full(len(a),np.nan),where=valid)
        fb=arrays['feedback_norm'] if 'feedback_norm' in arrays else arrays['sv_feedback'][:,0]
        values['feedback_norm']=fb;values['relative_feedback']=np.divide(fb,r[:,0],out=np.full(len(a),np.nan),where=valid)
    if 's' in arrays:
        for key in ('s','gradient_norm','chi'):values[key]=arrays[key]
    return values
def main(root):
    sys.path.insert(0,str(root/'source'));import study,feedback_math
    study.verify(root);assert study.load(root/'controller/status.json')['state']=='COMPLETE'
    assert all(r['exit_code']==0 for r in study.load(root/'controller/launches.json'))
    audit=study.load(root/'audit/result.json');assert audit['status']=='PASS' and audit['probes']==122880
    out=root/'curation';out.mkdir();figures=out/'figures';figures.mkdir();rows=[];variance=[];cases=[];groups={};native_errors=[]
    for cell in study.CELLS:
        arm,step=study.split(cell);folder=root/'output'/cell
        for r in study.load(folder/'result.json')['rows']:
            key=r['group'];domain,time_name,block,kind=key.split('-');index=int(time_name[1:]);local=kind!='core'
            with np.load(folder/f'{key}-spectra.npz') as f:arrays={k:f[k] for k in f.files}
            descriptors=describe(arrays);meta=dict(family=arm,step=step,domain=domain,time_index=index,time=index/128,layer=block+'-'+kind,kind='local' if local else 'core',probes=256)
            summary=dict(meta)
            for key2,value in descriptors.items():summary.update({key2+'_'+k:v for k,v in stats(value).items()})
            summary.update(base_inverse_excluded=int(np.sum(~arrays['base_inverse_valid'])) if local else 0,
                reference_zero_gain_count=int(np.sum(arrays['sv_ref'][:,0]==0)) if 'sv_ref' in arrays else 0,
                rank3_residual_max=float(arrays['sv_feedback'][:,2].max()) if 'sv_feedback' in arrays else 0,
                native_to_promoted_max=r['native_to_promoted_max'],native_to_promoted_RMS=r['native_to_promoted_RMS'])
            # Stable CSV columns for the ordinary rows, whose clock fields are NA.
            rows.append(summary);groups[(cell,domain,index,block,kind)]=arrays
            native_errors.append(dict(cell=cell,group=key,maximum=r['native_to_promoted_max'],RMS=r['native_to_promoted_RMS']))
            if local and block=='block0' and kind=='ode1':
                cases.append(dict(**meta,image_index=0,token_index=0,**{k:float(v[0]) for k,v in descriptors.items()}))
        if arm.startswith('ode'):
            for domain in ('paired','generated'):
                for block in range(3):
                    for layer in (1,2):
                        name=f'block{block}-ode{layer}';values=[]
                        for j in study.INDICES:
                            with np.load(folder/f'{domain}-t{j:03d}-{name}-capture.npz') as f:values.append(f['s_all'])
                        s=np.stack(values);parts=feedback_math.variance_parts(s)
                        variance.append(dict(family=arm,step=step,domain=domain,layer=name,clock_values=s.size,
                            mean=float(s.mean(dtype=np.float64)),minimum=float(s.min()),maximum=float(s.max()),**parts))
    assert len(rows)==480 and len(variance)==48 and len(cases)==40
    keys=list(dict.fromkeys(k for row in rows for k in row));table(out/'groups.csv',[{k:r.get(k) for k in keys} for r in rows])
    table(out/'clock-variance.csv',variance);table(out/'fixed-cases.csv',cases);table(out/'precision.csv',native_errors)
    pooled=[];profiles=[];variance_pooled=[]
    for cell in study.CELLS:
        arm,step=study.split(cell)
        for domain in ('paired','generated'):
            for kind in (('core','local') if arm.startswith('ode') else ('core',)):
                selected=[v for (c,d,j,b,k),v in groups.items() if c==cell and d==domain and (k=='core')==(kind=='core')]
                arrays={k:np.concatenate([v[k] for v in selected],axis=0) for k in selected[0] if k!='P_sigma_max'}
                record=dict(family=arm,step=step,domain=domain,kind=kind,probes=len(arrays['sv_full']))
                for k,v in describe(arrays).items():record.update({k+'_'+q:a for q,a in stats(v).items()})
                record['base_inverse_excluded']=int(np.sum(~arrays['base_inverse_valid'])) if kind=='local' else 0
                pooled.append(record)
            for j in study.INDICES:
                chosen=[groups[(cell,domain,j,f'block{block}','core')] for block in range(3)]
                arrays={k:np.concatenate([v[k] for v in chosen],axis=0) for k in chosen[0]}
                record=dict(family=arm,step=step,domain=domain,time_index=j,time=j/128,probes=768)
                for k,v in describe(arrays).items():record.update({k+'_'+q:a for q,a in stats(v).items()})
                profiles.append(record)
            if arm.startswith('ode'):
                selected=[r for r in variance if (r['family'],r['step'],r['domain'])==(arm,step,domain)]
                total=sum(r['variance_total'] for r in selected)
                variance_pooled.append(dict(family=arm,step=step,domain=domain,clock_layers=6,total_within_clock_variance=total,
                    **{k:sum(r['variance_'+k] for r in selected)/total for k in ('time','image','token')}))
    allkeys=list(dict.fromkeys(k for r in profiles for k in r));table(out/'time-profiles.csv',[{k:r.get(k) for k in allkeys} for r in profiles])
    allkeys=list(dict.fromkeys(k for r in pooled for k in r));table(out/'pooled.csv',[{k:r.get(k) for k in allkeys} for r in pooled])
    table(out/'variance-pooled.csv',variance_pooled)
    for source,target in [('preflight.json','preflight.json'),('documented-pass.json','agent-doc-pass.json')]:shutil.copyfile(root/source,out/target)
    study.write(out/'audit-summary.json',dict(status='PASS',groups=audit['groups'],probes=audit['probes'],native_endpoints_bitwise=audit['native_endpoints_bitwise'],
        max_AD_spectra_error=max(r['AD_spectra_max_error'] for r in audit['rows']),seconds=audit['seconds']))
    study.write(out/'summary.json',dict(status='PASS',groups=480,core_probes=61440,local_probes=61440,clock_groups=48,clock_values=983040,
        maximum_native_to_promoted=max(r['maximum'] for r in native_errors),base_inverse_excluded=sum(r['base_inverse_excluded'] for r in rows),
        maximum_FFN_rank3_residual=max(r['rank3_residual_max'] for r in rows),new_training=False,new_quality_result=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def export(fig,name):
        for ext in ('png','pdf'):fig.savefig(figures/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for ax,(arm,step) in zip(axes.flat,[(a,s) for a in ('ode_k1','ode_k2') for s in (5000,20000)]):
        selected=[r for r in variance_pooled if r['family']==arm and r['step']==step];bottom=np.zeros(2)
        for key,color in zip(('time','image','token'),('#3274a1','#e1812c','#3a923a')):
            v=np.array([r[key] for r in selected]);ax.bar([r['domain'] for r in selected],v,bottom=bottom,label=key,color=color);bottom+=v
        ax.set(title=f'{arm}, {step//1000}k',ylabel='Fraction of within-clock variance',ylim=(0,1))
    axes[0,0].legend(ncol=3,loc='upper center');export(fig,'clock-variance')
    labels=dict(ode_k1='K1',ode_k2='K2',swiglu86='Swi86',silu131='SiLU131');colors=('#2378a8','#d65b36','#4f6d7a','#9a8c98')
    fig,axes=plt.subplots(2,2,figsize=(11,6),layout='constrained')
    for ax,(step,domain) in zip(axes.flat,[(s,d) for s in (5000,20000) for d in ('paired','generated')]):
        for arm,color in zip(study.FAMILIES,colors):
            selected=[r for r in profiles if (r['family'],r['step'],r['domain'])==(arm,step,domain)];x=[r['time'] for r in selected]
            ax.plot(x,[r['full_gain_median'] for r in selected],'-o',label=labels[arm],color=color)
            ax.plot(x,[r['full_gain_p95'] for r in selected],':',color=color,alpha=.7)
        ax.set(title=f'{step//1000}k, {domain}',xlabel='External flow time',ylabel='FFN Jacobian largest singular value')
    axes[0,0].legend(ncol=4,fontsize=8);fig.suptitle('Solid: median; dotted: 95th percentile. Different native state coordinates.')
    export(fig,'complete-core-gain')
    fig,axes=plt.subplots(2,2,figsize=(11,6),layout='constrained')
    for ax,(arm,domain) in zip(axes.flat,[(a,d) for a in ('ode_k1','ode_k2') for d in ('paired','generated')]):
        for step,color in zip((5000,20000),('#2378a8','#d65b36')):
            selected=[r for r in profiles if (r['family'],r['step'],r['domain'])==(arm,step,domain)];x=[r['time'] for r in selected]
            ax.plot(x,[r['relative_feedback_median'] for r in selected],'-o',label=f'{step//1000}k',color=color)
            ax.plot(x,[r['relative_feedback_p95'] for r in selected],':',color=color)
        ax.set(title=f'{labels[arm]}, {domain}',xlabel='External flow time',ylabel='||J - R||2 / ||R||2')
    axes[0,0].legend();fig.suptitle('Native pointwise clock feedback; solid median, dotted 95th percentile')
    export(fig,'gate-feedback')
    study.verify(root);print(json.dumps(study.load(out/'summary.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
