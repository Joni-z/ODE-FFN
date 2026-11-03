"""Independent NumPy replay of saved native-feature sufficient arrays."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.evidence);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
all_rows=[];all_pairs=[];bindings=[];all_arrays={};errors=[];receipts=[]
for host in ['145','lenovo']:
    directory=root/host;receipt=json.loads((directory/'results.json').read_text());assert receipt['passed'] and len(receipt['rows'])==42 and len(receipt['bindings'])==7 and len(receipt['pairs'])==30
    assert len({b['probe_sha256'] for b in receipt['bindings']})==1
    receipts.append(receipt)
    for binding in receipt['bindings']:
        file=directory/binding['array_file'];assert hashlib.sha256(file.read_bytes()).hexdigest()==binding['array_sha256']
        with np.load(file,allow_pickle=False) as data:
            for row in [r for r in receipt['rows'] if r['cell']==binding['id']]:
                prefix=f'layer{row["index"]}_';array={key.removeprefix(prefix):data[key].copy() for key in data.files if key.startswith(prefix)}
                P,h,d=array['P'],array['h'],array['d'];q=h*d;first=np.einsum('nj,ij->ni',q,P,optimize=False)
                assert np.array_equal(q,array['q'])
                M=np.einsum('ni,nj->ij',q,q,optimize=False)/len(q);H=np.mean(np.einsum('ni,ni->n',h,h,optimize=False));energy=np.mean(np.einsum('ni,ni->n',first,first,optimize=False))
                assert np.allclose(M,array['M'],rtol=1e-10,atol=1e-12*max(abs(M).max(),1e-30))
                assert np.allclose(M/H,array['N'],rtol=1e-10,atol=1e-12*max(abs(M/H).max(),1e-30))
                assert np.allclose(array['B'],P.T@P,rtol=1e-10,atol=1e-12*max(abs(P.T@P).max(),1e-30))
                second=.5*np.einsum('nj,ij->ni',first,P,optimize=False)*d if row['orders']==2 else np.zeros_like(first)
                assert np.allclose(second,array['r2'],rtol=1e-9,atol=1e-10*max(abs(second).max(),1e-30))
                total_energy=np.mean(np.sum((first+second)**2,axis=1));cross=2*np.mean(np.sum(first*second,axis=1));e2=np.mean(np.sum(second*second,axis=1))
                checks=[abs(energy-row['increment_energy'])/max(energy,1e-30),abs(np.sqrt(energy/H)-row['first_term_rms_over_input'])/max(row['first_term_rms_over_input'],1e-30),abs(total_energy-row['full_increment_energy'])/max(total_energy,1e-30),abs(total_energy-energy-e2-cross)/max(total_energy,energy+e2,1e-30)]
                assert max(checks)<1e-9;errors+=checks
                for array_input,key in [(P,'p_singular_values'),(q/np.sqrt(len(q)),'feature_singular_values'),(first/np.sqrt(len(q)),'increment_singular_values')]:
                    s=np.linalg.svd(array_input,compute_uv=False);assert np.allclose(s,row[key],rtol=1e-9,atol=1e-11*max(s[0],1e-30))
                qenergy=float(np.mean(np.sum(q*q,axis=1)))
                row['feature_amplitude_over_input']=float(np.sqrt(qenergy/H))
                row['response_gain_on_feature']=float(np.sqrt(energy/qenergy))
                assert abs(row['feature_amplitude_over_input']*row['response_gain_on_feature']/row['first_term_rms_over_input']-1)<1e-9
                all_arrays[(row['cell'],row['layer'])]=array;all_rows.append(row)
        bindings.append(binding)
    all_pairs+=receipt['pairs']
assert len(all_rows)==84 and len(all_pairs)==60
assert len({r['analysis_source_sha256'] for r in receipts})==len({r['cases_sha256'] for r in receipts})==1
for pair in all_pairs:
    left=all_arrays[(pair['reference'],pair['layer'])];right=all_arrays[(pair['cell'],pair['layer'])]
    def response(array):return np.mean(np.sum((array['q']@array['P'].T)**2,axis=1))/np.mean(np.sum(array['h']**2,axis=1))
    delta=response(right)-response(left)
    param=np.einsum('ij,ji',right['B']-left['B'],(right['N']+left['N'])/2,optimize=False)
    feature=np.einsum('ij,ji',(right['B']+left['B'])/2,right['N']-left['N'],optimize=False)
    scale=max(response(left),response(right),1e-30);error=abs(delta-param-feature)/scale;assert error<1e-9;errors.append(error)
    assert abs(param-pair['parameter_accounting_term'])/scale<1e-9 and abs(feature-pair['feature_accounting_term'])/scale<1e-9
def save(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
save('layers.csv',[{k:v for k,v in r.items() if not isinstance(v,list)} for r in all_rows]);save('pairs.csv',all_pairs)
models=[]
for binding in bindings:
    rr=[r for r in all_rows if r['cell']==binding['id']]
    model={k:binding[k] for k in ['id','dataset','family','relaxation','step']}
    for key in ['p_sigma_max','p_stable_rank','first_term_rms_over_input','feature_stable_rank','increment_stable_rank','feature_mean_energy_fraction','increment_mean_energy_fraction','term_cancellation_factor','feature_amplitude_over_input','response_gain_on_feature']:
        model['mean_'+key]=float(np.mean([r[key] for r in rr]))
    models.append(model)
save('models.csv',models)
fig,axes=plt.subplots(2,2,figsize=(11,7));colors={'none':'#4477aa','shape':'#ee7733','scale':'#228833','native80k':'#aa3377'}
for axs,ds in zip(axes,['gmm_ring8_2d','rotated_product_bimodal_16d']):
    for ax,family in zip(axs,['scalar_k2','pd_direct_raw_k1']):
        for mode,color in colors.items():
            rr=sorted([r for r in all_rows if r['dataset']==ds and r['family']==family and r['relaxation']==mode],key=lambda r:r['index'])
            if rr:ax.plot(range(6),[r['first_term_rms_over_input'] for r in rr],'-o',label=mode,color=color)
        ax.set(title=f'{ds}: {family}',xlabel='Layer: block0/1/2, solve1/2',ylabel='First-term RMS / input RMS',yscale='log');ax.set_xticks(range(6),['0/1','0/2','1/1','1/2','2/1','2/2']);ax.legend();ax.grid(alpha=.2)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'layer-responses.{ext}',dpi=160,bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,ds in zip(axes,['gmm_ring8_2d','rotated_product_bimodal_16d']):
    for mode,color in colors.items():
        rr=sorted([r for r in all_rows if r['dataset']==ds and r['family']=='scalar_k2' and r['relaxation']==mode],key=lambda r:r['index'])
        if rr:ax.plot(range(6),[r['term_cancellation_factor'] for r in rr],'-o',label=mode,color=color)
    ax.axhline(1,color='gray',ls=':');ax.set(title=ds+' scalarK2',xlabel='Layer: block/solve',ylabel='Energy of sum / sum of term energies');ax.set_xticks(range(6),['0/1','0/2','1/1','1/2','2/1','2/2']);ax.legend();ax.grid(alpha=.2)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'finite-series-alignment.{ext}',dpi=160,bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(12,7))
for axs,ds in zip(axes,['gmm_ring8_2d','rotated_product_bimodal_16d']):
    for ax,family in zip(axs,['scalar_k2','pd_direct_raw_k1']):
        selected=next(b for b in bindings if b['dataset']==ds and b['family']==family and b['relaxation']=='shape')
        pp=sorted([r for r in all_pairs if r['cell']==selected['id']],key=lambda r:next(x['index'] for x in all_rows if x['cell']==r['cell'] and x['layer']==r['layer']))
        baseline=[response(all_arrays[(r['reference'],r['layer'])]) for r in pp]
        for shift,key,label in [(-.25,'parameter_accounting_term','P term'),(0,'feature_accounting_term','Feature term'),(.25,'normalized_energy_change','Total')]:
            ax.bar(np.arange(6)+shift,[r[key]/b for r,b in zip(pp,baseline)],width=.23,label=label)
        ax.axhline(0,color='black',lw=.6);ax.set(title=f'{ds}: {family}, native -> shape',xlabel='Layer: block/solve',ylabel='Change / native normalized energy');ax.set_xticks(range(6),['0/1','0/2','1/1','1/2','2/1','2/2']);ax.legend(fontsize=8);ax.grid(alpha=.15)
fig.suptitle('Symmetric algebraic accounting; neither component is a causal attribution')
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'response-accounting.{ext}',dpi=160,bbox_inches='tight')
verification=dict(passed=True,models=len(bindings),layers=len(all_rows),matched_layer_pairs=len(all_pairs),independent_relative_error_max=max(errors),native_layer_reconstruction_relative_rms_max=max(r['native_layer_reconstruction_relative_rms_error'] for r in all_rows),archived_first_term_relative_error_max=max(r['archived_term1_relative_error'] for r in all_rows),analysis_source_sha256=receipts[0]['analysis_source_sha256'],replay_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases_sha256=receipts[0]['cases_sha256'],scope='Saved-array moment/energy/spectrum replay, not a new model or independent re-generation of native hidden states.')
(out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n');(out/'bindings.json').write_text(json.dumps(bindings,indent=2)+'\n');(out/'spectra.json').write_text(json.dumps(all_rows,indent=2)+'\n')
print(json.dumps(verification));print(json.dumps(models))
