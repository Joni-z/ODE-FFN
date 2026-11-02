"""Render the completed second wave beside all audited first-wave controls."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('receipts',nargs='+');p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
receipts=[json.loads(Path(f).read_text()) for f in a.receipts]
rows=[r for d in receipts for r in d['endpoints']]
assert len(rows)==30 and len({r['id'] for r in rows})==30
root=Path(__file__).resolve().parents[3]
grid=json.loads((Path(__file__).parent/'grid.json').read_text())
assert {r['id'] for r in rows}=={r['id'] for r in grid['cells']}
for r in rows:
    expected=next(c for c in grid['cells'] if c['id']==r['id'])
    assert all(r[k]==v for k,v in expected.items())
def save(name,entries):
    keys=list(dict.fromkeys(k for r in entries for k in r))
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(entries)
save('endpoints.csv',rows)
for name in ('curves','updates'):save(name+'.csv',[r for d in receipts for r in d[name]])
old=list(csv.DictReader((root/'docs/research-muon-pd-2026-11-02/wave2/analysis/endpoints.csv').open()))
combined=[]
for wave,items in [(1,old),(2,rows)]:
    for r in items:
        if r['dataset']=='gaussian_aniso_2d':continue
        combined.append(dict(wave=wave,dataset=r['dataset'],family=r['family'],optimizer=r['optimizer'],lr=float(r['projection_lr']),decay=float(r.get('projection_decay',0)),field_mse=float(r['replayed_field_mse']),test_sw2=float(r['numpy_test_sw2'])))
save('all-comparisons.csv',combined)
fig,axs=plt.subplots(1,2,figsize=(11,5))
colors={'scalar_k2':'#4477aa','swiglu128':'#228833','swiglu192':'#66aa55','vector_dp_k2':'#ee7733','vector_pd_k2':'#cc3311','vector_sandwich_k2':'#aa3377'}
for ax,ds in zip(axs,['gmm_ring8_2d','rotated_product_bimodal_16d']):
    for family,color in colors.items():
        for wave,marker in [(1,'o'),(2,'^')]:
            rr=[r for r in combined if r['dataset']==ds and r['family']==family and r['wave']==wave]
            ax.scatter([r['field_mse'] for r in rr],[r['test_sw2'] for r in rr],c=color,marker=marker,label=family if (wave==1 and family!='vector_sandwich_k2') or (wave==2 and family=='vector_sandwich_k2') else None,alpha=.8)
    ax.set(xlabel='Validation field MSE',ylabel='Test sliced W2',title=ds)
    ax.grid(alpha=.2)
axs[1].legend(fontsize=8);fig.suptitle('All registered settings: circles = wave1; triangles = wave2; lower is better')
fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
checks=dict(endpoints=len(rows),maximum_field_replay_error=max(r['field_abs_error'] for r in rows),maximum_sw2_replay_error=max(r['sw2_abs_error'] for r in rows),maximum_sample_replay_error=max(r['generated_replay_max_abs_error'] for r in rows))
(out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
for ds in ['gmm_ring8_2d','rotated_product_bimodal_16d']:
    print(ds)
    for r in sorted([r for r in combined if r['dataset']==ds],key=lambda x:x['field_mse']):print(r)
