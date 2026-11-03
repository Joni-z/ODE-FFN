"""Hash-bound four-corner analysis; final validation drives decisions."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--ordinary',required=True);p.add_argument('--states',nargs=2,required=True);a=p.parse_args()
out=Path(a.results);root=Path(__file__).resolve().parents[4];cohort=json.loads(Path(__file__).with_name('factorial-parents.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for file,digest in cohort['sources'].items():assert sha(root/file)==digest
tables={file:list(csv.DictReader((root/file).open())) for file in cohort['sources']}
for r in cohort['rows']:
    matches=[x for x in tables[r['source_csv']] if x['id']==r['id'] and x['checkpoint_sha256']==r['checkpoint_sha256']];assert len(matches)==1 and all(r[k]==v for k,v in matches[0].items())
native=list(csv.DictReader((out/'endpoints.csv').open()));ordinary=list(csv.DictReader(Path(a.ordinary).open()));assert len(native)==10
assert json.loads((out/'verification.json').read_text())['endpoint_steps']==[20000]
states=[json.loads(Path(f).read_text()) for f in a.states];assert all(s['passed'] and not s['partial'] and len(s['anchors'])==2 and not s['pending'] for s in states)
assert {r['cell'] for s in states for r in s['anchors']}=={r['id'] for r in native if r['role']=='anchor'}
corners=cohort['rows']+[dict(r,source_csv=str(out/'endpoints.csv')) for r in native if r['role']=='candidate'];assert len(corners)==24
families=['vector_pd_k2','pd_factorized_linear','pd_mlp_affine','pd_factorized_affine'];labels=['MLP / sigmoid','Linear / sigmoid','MLP / affine','Linear / affine'];tasks=['gmm_ring8_2d','rotated_product_bimodal_16d'];settings=[('adamw',.0003),('muon',.006),('muon',.02)]
def get(ds,fam,opt,lr):
    rr=[r for r in corners if r['dataset']==ds and r['family']==fam and r['optimizer']==opt and float(r['projection_lr'])==lr];assert len(rr)==1;return rr[0]
def field(r):return float(r['replayed_field_mse'])
def sw2(r):return float(r['numpy_test_sw2'])
def save(file,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (out/file).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
comparisons=[];table=[]
for ds in tasks:
    best=min(field(r) for r in ordinary if r['dataset']==ds and r['family'].startswith('swiglu'))
    for opt,lr in settings:
        ms,ls,ma,la=[get(ds,f,opt,lr) for f in families]
        assert la['parent_id']==ls['id'] and la['parent_checkpoint_sha256']==ls['checkpoint_sha256'] and la['mlp_affine_parent_id']==ma['id'] and la['mlp_affine_parent_sha256']==ma['checkpoint_sha256']
        comparisons.append(dict(dataset=ds,optimizer=opt,lr=lr,linearization_sigmoid_log_ratio=math.log(field(ls)/field(ms)),linearization_affine_log_ratio=math.log(field(la)/field(ma)),log_mse_interaction=math.log(field(la)/field(ma))-math.log(field(ls)/field(ms)),new_vs_linear_sigmoid_gain=1-field(la)/field(ls),new_vs_mlp_affine_gain=1-field(la)/field(ma),new_vs_linear_sigmoid_sw2_change=sw2(la)/sw2(ls)-1,new_vs_mlp_affine_sw2_change=sw2(la)/sw2(ma)-1,strongest_ordinary_field=best,continuation_eligible=field(la)<=.95*field(ls) and field(la)<=.95*field(ma) and field(la)<best))
        for r,label in zip([ms,ls,ma,la],labels):table.append(dict(dataset=ds,optimizer=opt,lr=lr,corner=label,cell=r['id'],validation_mse=field(r),test_sw2=sw2(r),strongest_ordinary_field=best,field_over_ordinary=field(r)/best,checkpoint_sha256=r['checkpoint_sha256'],source_csv=r['source_csv']))
save('factorial-endpoints.csv',table);save('factorial-interactions.csv',comparisons)
operator_rows=[];parent_curve_sources={}
for n,family in [(1,'pd_mlp_affine'),(2,'pd_factorized_linear')]:
    path=root/f'docs/autoresearch/professor-20261103/cycle{n}-results/curves.csv';parent_curve_sources[str(path.relative_to(root))]=sha(path)
    for row in csv.DictReader(path.open()):
        if row['family']==family and row['step']=='20000':operator_rows.append(dict(row,source_csv=str(path.relative_to(root))))
for row in csv.DictReader((out/'curves.csv').open()):
    if row['family']=='pd_factorized_affine' and row['step']=='20000':operator_rows.append(dict(row,source_csv=str(out/'curves.csv')))
assert len(operator_rows)==18
save('operator-endpoints.csv',operator_rows)

fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,ds in zip(axes,tasks):
    z=np.array([[next(r['field_over_ordinary'] for r in table if r['dataset']==ds and r['corner']==label and r['optimizer']==opt and r['lr']==lr) for opt,lr in settings] for label in labels]);im=ax.imshow(z,aspect='auto',vmin=min(1,float(z.min())),vmax=float(z.max()),cmap='YlOrRd')
    for i in range(4):
        for j,(opt,lr) in enumerate(settings):r=get(ds,families[i],opt,lr);ax.text(j,i,f'{field(r):.5f}\n{z[i,j]:.2f}x ordinary',ha='center',va='center',fontsize=8)
    ax.set(xticks=range(3),xticklabels=['Adam .0003','Muon .006','Muon .02'],yticks=range(4),yticklabels=labels,title=ds);fig.colorbar(im,ax=ax,label='Field MSE / strongest ordinary')
fig.suptitle('All four corners at every fixed20k setting; lower is better');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'factorial.{ext}',dpi=160,bbox_inches='tight')
plt.close(fig)
curves=list(csv.DictReader((out/'curves.csv').open()));fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,ds in zip(axes,tasks):
    for row in [r for r in native if r['dataset']==ds]:
        cc=[c for c in curves if c['cell']==row['id'] and int(c['step'])>=1000];label=f"{row['family']} / {row['optimizer']} {row['projection_lr']}";ax.plot([int(c['step']) for c in cc],[float(c['field_mse']) for c in cc],label=label)
    ax.set(title=ds,xlabel='Native updates',ylabel='Validation oracle MSE',yscale='log');ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'learning.{ext}',dpi=160,bbox_inches='tight')
plt.close(fig)
(out/'factorial-bindings.json').write_text(json.dumps(dict(parent_cohort_sha256=sha(Path(__file__).with_name('factorial-parents.json')),ordinary_csv_sha256=sha(Path(a.ordinary)),new_endpoint_csv_sha256=sha(out/'endpoints.csv'),analysis_source_sha256=sha(Path(__file__)),parent_curve_sources=parent_curve_sources,state_receipts=states),indent=2)+'\n')
print(json.dumps(dict(corners=len(table),settings=len(comparisons),comparisons=comparisons)))
