"""Publish completed registered trials, full controls, and validation-only decisions."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('receipts',nargs='+');p.add_argument('--out',required=True);p.add_argument('--ledger',required=True);p.add_argument('--runtime-commit',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[3]
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
receipts=[json.loads(Path(f).read_text()) for f in a.receipts]
rows=[r for d in receipts for r in d['endpoints']]
grid=json.loads(Path(__file__).with_name('grid.json').read_text())
assert len(rows)==len(grid['cells']) and {r['id'] for r in rows}=={r['id'] for r in grid['cells']}
for r in rows:
    cell=next(c for c in grid['cells'] if c['id']==r['id']);assert all(r[k]==v for k,v in cell.items())

def save(name,records):
    keys=list(dict.fromkeys(k for row in records for k in row))
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)

old=[]
for file in ['docs/research-muon-pd-2026-11-02/wave2/analysis/endpoints.csv','docs/research-muon-pd-2026-11-02/wave2/results/endpoints.csv']:
    old+=list(csv.DictReader((root/file).open()))
for r in old:
    r['replayed_field_mse']=float(r['replayed_field_mse']);r['numpy_test_sw2']=float(r['numpy_test_sw2']);r['projection_lr']=float(r['projection_lr'])
decisions=[]
for r in rows:
    if r['role']=='anchor':
        assert r['anchor_field_abs_error']<2e-6 and r['anchor_sample_max_abs_error']<2e-5
        decision='anchor';parent=None;improvement=None;sampling_change=None
    else:
        parent=next(x for x in old if x['dataset']==r['dataset'] and x['family']=='vector_pd_k2' and x['optimizer']==r['optimizer'] and x['projection_lr']==r['projection_lr'])
        improvement=1-r['replayed_field_mse']/parent['replayed_field_mse']
        sampling_change=r['numpy_test_sw2']/parent['numpy_test_sw2']-1
        decision='keep' if improvement>=.05 else 'discard'
    ordinary=min(x['replayed_field_mse'] for x in old if x['dataset']==r['dataset'] and x['family'].startswith('swiglu'))
    decisions.append(dict(cell=r['id'],dataset=r['dataset'],family=r['family'],optimizer=r['optimizer'],lr=r['projection_lr'],status=decision,
        field_mse=r['replayed_field_mse'],test_sw2=r['numpy_test_sw2'],parent=parent['id'] if parent else None,
        field_relative_improvement=improvement,test_sw2_relative_change=sampling_change,better_field_than_strongest_ordinary=r['replayed_field_mse']<ordinary))
save('endpoints.csv',rows);save('decisions.csv',decisions)
for name in ['curves','updates']:save(name+'.csv',[r for d in receipts for r in d[name]])
verification=dict(endpoints=len(rows),max_field_replay_error=max(r['field_abs_error'] for r in rows),max_sw2_error=max(r['sw2_abs_error'] for r in rows),max_generated_error=max(r['generated_replay_max_abs_error'] for r in rows),max_anchor_field_error=max(r.get('anchor_field_abs_error',0) for r in rows),max_anchor_generated_error=max(r.get('anchor_sample_max_abs_error',0) for r in rows))
(out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
ledger=Path(a.ledger);ledger.parent.mkdir(parents=True,exist_ok=True)
assert not ledger.exists(),'Do not overwrite the append-only cycle ledger'
with ledger.open('x') as f:
    f.write('commit\tvalidation_mse\ttest_sw2\tstatus\tdescription\n')
    for row in decisions:f.write(f"{a.runtime_commit}\t{row['field_mse']:.10g}\t{row['test_sw2']:.10g}\t{row['status']}\t{row['cell']}\n")
for row in decisions:print(json.dumps(row))
print(json.dumps(verification))
