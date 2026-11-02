"""Independently check every registered frozen shape/scale result."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('receipts',nargs=2);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
study=Path(__file__).resolve().parents[4]/'docs/autoresearch/professor-20261103'
parents=[]
for f in [study/'cycle3-results/endpoints.csv',study/'cycle6-results/endpoints.csv',study.parents[1]/'research-muon-pd-2026-11-02/wave2/analysis/endpoints.csv']:
    parents+=list(csv.DictReader(f.open()))
receipts=[json.loads(Path(f).read_text()) for f in a.receipts]
assert all(d['complete'] and len(d['rows'])==27 and len(d['bindings'])==3 for d in receipts)
assert len({d['analysis_source_sha256'] for d in receipts})==1
records=[];max_energy=max_pair=0.;verified_anchors=0
fig,axes=plt.subplots(2,3,figsize=(12,7))
for axs,d in zip(axes,sorted(receipts,key=lambda x:x['dataset'])):
    for ax,binding in zip(axs,d['bindings']):
        rows=[r for r in d['rows'] if r['cell']==binding['cell']]
        assert len(rows)==9 and len(binding['before'])==6
        parent=next(r for r in parents if r['id']==binding['cell'] and r['checkpoint_sha256']==binding['checkpoint_sha256'])
        anchor=next(r for r in rows if r['branch']=='anchor')
        assert abs(anchor['validation_field_mse']-float(parent['replayed_field_mse']))<2e-6
        assert abs(anchor['test_sw2']-float(parent['numpy_test_sw2']))<2e-6
        assert anchor['generated_replay_max_error']<2e-5
        verified_anchors+=1
        for r in rows:
            for name,before in binding['before'].items():
                s=np.array(before['singular_values']);new=np.array(r['matrices'][name]['singular_values']);q=np.linalg.norm(s)
                assert np.isfinite(new).all() and np.all(new>=0)
                if r['branch']=='anchor':expected=s
                else:
                    k=r['strength'];blend=(1-k)*s+k*q/np.sqrt(len(s));target=q*blend/np.linalg.norm(blend)
                    expected=target if r['branch']=='shape' else target[0]/s[0]*s
                assert np.allclose(new,expected,rtol=2e-6,atol=2e-6*max(s[0],1.))
                if r['branch'] in ['anchor','shape']:
                    err=abs(np.linalg.norm(new)/q-1);max_energy=max(max_energy,err);assert err<2e-6
                if r['branch']=='shape':
                    scale=next(x for x in rows if x['branch']=='scale' and x['strength']==r['strength'])
                    err=abs(new[0]-scale['matrices'][name]['sigma_max'])/max(s[0],1.);max_pair=max(max_pair,err);assert err<2e-6
            records.append(dict(dataset=d['dataset'],cell=r['cell'],family=r['family'],step=r['step'],branch=r['branch'],strength=r['strength'],field_mse=r['validation_field_mse'],test_sw2=r['test_sw2'],field_relative_change=r['validation_field_mse']/anchor['validation_field_mse']-1,test_sw2_relative_change=r['test_sw2']/anchor['test_sw2']-1 if r['test_sw2'] is not None else None,finite_generated_fraction=r['finite_generated_fraction'],sampling_metric_finite=r['sampling_metric_finite'],prediction_displacement_mse=r['field_prediction_displacement_mse']))
        for branch,style in [('shape','-o'),('scale','--s')]:
            rr=sorted([r for r in rows if r['branch'] in ['anchor',branch]],key=lambda x:x['strength'])
            ax.plot([r['strength'] for r in rr],[r['validation_field_mse']/anchor['validation_field_mse'] for r in rr],style,label=branch)
        ax.axhline(1,color='gray',ls=':');ax.set(title=f'{d["dataset"]}\n{anchor["family"]}, {anchor["step"]} updates',xlabel='Flattening strength / paired contraction',ylabel='Field MSE / untouched MSE',yscale='log');ax.legend();ax.grid(alpha=.2)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'field-sensitivity.{ext}',dpi=170,bbox_inches='tight')
with (out/'endpoints.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
verification=dict(passed=True,cases=len(records),anchors=verified_anchors,max_relative_shape_energy_error=max_energy,max_relative_shape_scale_operator_norm_error=max_pair,source_sha256=receipts[0]['analysis_source_sha256'],nonfinite_sampling_cases=sum(not r['sampling_metric_finite'] for r in records))
(out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification))
for r in records:
    if r['branch']=='anchor' or r['strength']==.02:print(json.dumps(r))
