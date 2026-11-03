"""Independent saved-array checks and complete paired frozen-swap figures."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input-root',required=True);p.add_argument('--out',required=True);p.add_argument('--ordinary',required=True);a=p.parse_args()
raw=Path(a.input_root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);figs=out/'figures';figs.mkdir(exist_ok=True);cohort_file=Path(__file__).with_name('cases.json');cohort=json.loads(cohort_file.read_text());ordinary=list(csv.DictReader(Path(a.ordinary).open()))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();root=Path(__file__).resolve().parents[4]
for file,digest in cohort['source_csvs'].items():assert sha(root/file)==digest
families=['vector_pd_k2','pd_factorized_linear','pd_mlp_affine','pd_factorized_affine'];labels=dict(zip(families,['MLP / sigmoid','Linear / sigmoid','MLP / affine','Linear / affine']));tasks=['gmm_ring8_2d','rotated_product_bimodal_16d'];settings=[('adamw',.0003),('muon',.006),('muon',.02)]
rows=[];pairs=[];layer_rows=[];bindings=[];arrays={};manifests={};input_hashes={};sw_errors=[];native_field_errors=[];native_sw_errors=[];targets={}
def save(name,records):
    keys=list(dict.fromkeys(k for r in records for k in r))
    with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
for host in ['145','lenovo']:
    folder=raw/host;file=folder/'results.json';d=json.loads(file.read_text());finished=json.loads((folder/'finished.json').read_text());manifest_file=folder/'manifest.json';manifest=json.loads(manifest_file.read_text());manifests[host]=manifest
    assert d['passed'] and d['complete'] and not d['witness'] and d['host']==host and len(d['rows'])==24 and len(d['bindings'])==12
    assert finished['passed'] and finished['checkpoints']==12 and finished['evaluations']==24 and finished['results_sha256']==sha(file)
    assert d['cohort_sha256']==sha(cohort_file) and d['producer_sha256']==sha(Path(__file__).with_name('swap.py'))
    for file in [file,folder/'finished.json',manifest_file]:input_hashes[str(file.relative_to(raw))]=sha(file)
    for b in d['bindings']:
        case=next(c for c in cohort['cases'] if c['id']==b['id']);assert all(b[k]==v for k,v in case.items()) and case['host']==host
        assert b['round_trip_probe_exact'] and b['tensors_buffers_unchanged'] and b['manifest_sha256']==sha(manifest_file)
        file=folder/b['array_file'];assert sha(file)==b['array_sha256'];input_hashes[str(file.relative_to(raw))]=sha(file)
        with np.load(file,allow_pickle=False) as payload:arr={k:payload[k].copy() for k in payload.files}
        assert np.array_equal(arr['native_probe'],arr['restored_probe']) and len(arr['target'])==4096 and arr['directions'].shape==(256,arr['target'].shape[1])
        target_sha=hashlib.sha256(arr['target'].tobytes()).hexdigest();targets.setdefault(host,target_sha);assert targets[host]==target_sha
        actions=[r for r in d['rows'] if r['cell']==case['id']];assert len(actions)==2 and {r['action'] for r in actions}=={'native','swap'}
        native=next(r for r in actions if r['action']=='native');swap=next(r for r in actions if r['action']=='swap')
        assert native['transform']==case['native_transform'] and swap['transform']!=native['transform']
        assert native['generated_replay_max_error']==0 and native['finite_field'] and native['sampling_metric_finite']
        native_field_errors.append(abs(native['validation_field_mse']-case['native_field_mse']));native_sw_errors.append(abs(native['test_sw2']-case['native_test_sw2']))
        assert native_field_errors[-1]<2e-6 and native_sw_errors[-1]<2e-6
        for r in actions:
            generated=arr[r['action']+'_generated'];assert generated.shape==arr['target'].shape
            finite=bool(np.isfinite(generated).all());fraction=float(np.isfinite(generated).all(1).mean());assert fraction==r['finite_generated_fraction']
            numpy_sw2=None
            if finite:
                gx=np.sort(generated.astype(np.float64)@arr['directions'].astype(np.float64).T,axis=0);gt=np.sort(arr['target'].astype(np.float64)@arr['directions'].astype(np.float64).T,axis=0);numpy_sw2=float(np.sqrt(np.mean((gx-gt)**2)))
            if r['test_sw2'] is not None:
                assert finite and r['sampling_metric_finite'];error=abs(numpy_sw2-r['test_sw2']);assert error<2e-6;sw_errors.append(error)
            else:assert not r['sampling_metric_finite']
            clean={k:v for k,v in r.items() if k!='activation'};clean.update(numpy_test_sw2=numpy_sw2,checkpoint_sha256=b['checkpoint_sha256'],array_sha256=b['array_sha256']);rows.append(clean)
            for layer in r['activation'] or []:
                act=layer['activation'];rr=dict(cell=r['cell'],dataset=r['dataset'],family=r['family'],optimizer=r['optimizer'],lr=r['projection_lr'],action=r['action'],layer=layer['layer'],p_sigma_max=layer['raw']['sigma_max'],p_stable_rank=layer['raw']['stable_rank'])
                rr.update({k:v for k,v in act.items() if isinstance(v,(float,int,str))});layer_rows.append(rr)
        if native['activation'] is not None and swap['activation'] is not None:
            assert len(native['activation'])==len(swap['activation'])==6
            assert all(n['layer']==s['layer'] and n['raw']==s['raw'] for n,s in zip(native['activation'],swap['activation']))
        nf=native['validation_field_mse'];sf=swap['validation_field_mse'];disp=swap['field_prediction_displacement_mse']
        if sf is not None:
            assert swap['finite_field'] and sf>=0 and disp>=0;nn,ss,dd=map(math.sqrt,[nf,sf,disp]);tol=1e-7*max(nn,ss,dd,1);assert abs(nn-ss)<=dd+tol and dd<=nn+ss+tol
        baseline=next(r for r in ordinary if r['dataset']==case['dataset'] and r['family']=='swiglu128')
        pairs.append(dict(cell=case['id'],dataset=case['dataset'],family=case['family'],optimizer=case['optimizer'],lr=case['projection_lr'],native_transform=native['transform'],swapped_transform=swap['transform'],native_field_mse=nf,swapped_field_mse=sf,field_ratio=sf/nf if sf is not None else None,native_test_sw2=native['test_sw2'],swapped_test_sw2=swap['test_sw2'],test_sw2_ratio=swap['test_sw2']/native['test_sw2'] if swap['test_sw2'] is not None else None,swi128_field=float(baseline['replayed_field_mse']),swi128_test_sw2=float(baseline['numpy_test_sw2']),round_trip_probe_exact=True,tensors_buffers_unchanged=True))
        bindings.append(b);arrays[case['id']]=arr
assert len(rows)==48 and len(pairs)==len(bindings)==24 and {r['cell'] for r in pairs}=={c['id'] for c in cohort['cases']}
save('endpoints.csv',rows);save('paired-changes.csv',pairs);save('layer-statistics.csv',layer_rows)
def pair(ds,family,opt,lr):return next(r for r in pairs if r['dataset']==ds and r['family']==family and r['optimizer']==opt and r['lr']==lr)
def finish(fig,name):
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(figs/(name+'.'+ext),dpi=160,bbox_inches='tight')
    plt.close(fig)
z=[]
for ds in tasks:z.append(np.array([[math.log10(r['field_ratio']) if r['field_ratio'] is not None and r['field_ratio']>0 else np.nan for opt,lr in settings for r in [pair(ds,fam,opt,lr)]] for fam in families]))
extent=max(1e-3,max(float(np.nanmax(np.abs(v))) for v in z));fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,ds,matrix in zip(axes,tasks,z):
    im=ax.imshow(matrix,cmap='RdBu_r',vmin=-extent,vmax=extent,aspect='auto')
    for i,fam in enumerate(families):
        for j,(opt,lr) in enumerate(settings):r=pair(ds,fam,opt,lr);value=r['field_ratio'];ax.text(j,i,'nonfinite' if value is None else f'{value:.2f}x',ha='center',va='center',fontsize=9)
    ax.set(title=ds,xticks=range(3),xticklabels=['Adam .0003','Muon .006','Muon .02'],yticks=range(4),yticklabels=[labels[f] for f in families]);fig.colorbar(im,ax=ax,label='log10(swapped/native validation MSE)')
fig.suptitle('Every frozen outer-transform swap; ratio below 1 improves fitting');finish(fig,'field-changes')
product_spec=manifests['145']['datasets']['rotated_product_bimodal_16d']['distribution'];rotation=np.asarray(product_spec['rotation']);locations=np.asarray(product_spec['locations']);variances=np.asarray(product_spec['variances']);fig,axes=plt.subplots(4,3,figsize=(12,11));examples=[]
for axs,fam in zip(axes,families):
    r=pair(tasks[1],fam,'adamw',.0003);arr=arrays[r['cell']]
    for ax,j in zip(axs,[0,7,15]):
        limit=locations[j]+6*math.sqrt(variances[j]);edges=np.linspace(-limit,limit,61);x=np.linspace(-limit,limit,1000);pdf=(np.exp(-(x-locations[j])**2/(2*variances[j]))+np.exp(-(x+locations[j])**2/(2*variances[j])))/(2*np.sqrt(2*np.pi*variances[j]));ax.plot(x,pdf,color='black',lw=1,label='Population')
        outside={}
        for action,color in [('native','#4477aa'),('swap','#ee7733')]:
            generated=arr[action+'_generated'];finite=bool(np.isfinite(generated).all())
            if finite:
                latent=generated.astype(np.float64)@rotation;values=latent[:,j];counts,_=np.histogram(values,edges);ax.stairs(counts/len(values)/np.diff(edges),edges,color=color,label=action);outside[action]=float(np.mean(np.abs(values)>limit));examples.append(dict(cell=r['cell'],action=action,axis=j,gap_mass=float(np.mean(np.abs(values)<locations[j]/2)),outside_plot_fraction=outside[action]))
            else:ax.text(.04,.85,'Nonfinite '+action,transform=ax.transAxes);outside[action]=None
        ax.set(title=f'{labels[fam]}, axis {j}\noutside native/swap: {outside["native"]:.1%} / '+('nonfinite' if outside['swap'] is None else f'{outside["swap"]:.1%}'),xlabel='Latent coordinate',ylabel='Density');ax.legend(fontsize=7);ax.grid(alpha=.15)
fig.suptitle('Product16 / Adam .0003: every corner, all 4096 samples per histogram');finish(fig,'product-cases');save('product-case-readouts.csv',examples)
fig,axes=plt.subplots(4,2,figsize=(9,13),sharex=True,sharey=True)
for axs,fam in zip(axes,families):
    r=pair(tasks[0],fam,'muon',.02);arr=arrays[r['cell']]
    for ax,action in zip(axs,['native','swap']):
        target=arr['target'][::4];ax.scatter(*target.T,s=2,alpha=.25,color='black');generated=arr[action+'_generated'];finite=bool(np.isfinite(generated).all());outside=None
        if finite:outside=float(np.mean(np.any(np.abs(generated)>4,axis=1)));ax.scatter(*generated[::4].T,s=2,alpha=.45,color='#4477aa' if action=='native' else '#ee7733')
        ax.set(title=f'{labels[fam]} / {action}\n'+('nonfinite' if outside is None else f'outside frame {outside:.2%}'),xlim=(-4,4),ylim=(-4,4),aspect='equal');ax.grid(alpha=.15)
fig.suptitle('Ring8 / Muon .02: all corners, same 1024 display indices; scores use 4096');finish(fig,'ring-cases')
verification=dict(passed=True,checkpoints=24,evaluations=48,exact_native_array_anchors=24,exact_round_trip_probes=24,unchanged_tensor_buffer_checks=24,finite_sampling_scores=len(sw_errors),max_numpy_sw2_error=max(sw_errors),max_native_field_replay_error=max(native_field_errors),max_native_sw2_replay_error=max(native_sw_errors),field_improves_by_task={ds:sum(r['field_ratio'] is not None and r['field_ratio']<1 for r in pairs if r['dataset']==ds) for ds in tasks},scope='Paired frozen checkpoint contrasts; no training-method acceptance or inferential replication.')
(out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n');(out/'source-bindings.json').write_text(json.dumps(dict(runtime_commit='290359d596cfac5214e229367397f6fbe9b49f32',cohort_sha256=sha(cohort_file),analysis_sha256=sha(Path(__file__)),ordinary_csv_sha256=sha(Path(a.ordinary)),inputs=input_hashes,checkpoint_bindings=bindings),indent=2)+'\n');print(json.dumps(verification))
