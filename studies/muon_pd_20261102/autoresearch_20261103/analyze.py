"""Publish completed registered trials, full controls, and validation-only decisions."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('receipts',nargs='+');p.add_argument('--out',required=True);p.add_argument('--ledger',required=True);p.add_argument('--runtime-commit',required=True);p.add_argument('--grid');p.add_argument('--cycle1-parents');p.add_argument('--cycle2-parents');p.add_argument('--cycle3-parents');p.add_argument('--cycle4-parents');p.add_argument('--state-receipts',nargs='+');p.add_argument('--append-ledger',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[3]
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
receipts=[json.loads(Path(f).read_text()) for f in a.receipts]
rows=[r for d in receipts for r in d['endpoints']]
grid=json.loads((Path(a.grid) if a.grid else Path(__file__).with_name('grid.json')).read_text())
assert len(rows)==len(grid['cells']) and {r['id'] for r in rows}=={r['id'] for r in grid['cells']}
for r in rows:
    cell=next(c for c in grid['cells'] if c['id']==r['id']);assert all(r[k]==v for k,v in cell.items())

def save(name,records):
    keys=list(dict.fromkeys(k for row in records for k in row))
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)

old=[]
cycle1=list(csv.DictReader(Path(a.cycle1_parents).open())) if a.cycle1_parents else []
cycle2=list(csv.DictReader(Path(a.cycle2_parents).open())) if a.cycle2_parents else []
cycle3=list(csv.DictReader(Path(a.cycle3_parents).open())) if a.cycle3_parents else []
cycle4=list(csv.DictReader(Path(a.cycle4_parents).open())) if a.cycle4_parents else []
for file in ['docs/research-muon-pd-2026-11-02/wave2/analysis/endpoints.csv','docs/research-muon-pd-2026-11-02/wave2/results/endpoints.csv']:
    old+=list(csv.DictReader((root/file).open()))
for r in old:
    r['replayed_field_mse']=float(r['replayed_field_mse']);r['numpy_test_sw2']=float(r['numpy_test_sw2']);r['projection_lr']=float(r['projection_lr'])
state_anchors={}
if a.state_receipts:
    states=[json.loads(Path(f).read_text()) for f in a.state_receipts]
    assert all(s['passed'] for s in states)
    state_anchors={r['cell']:r for s in states for r in s['anchors']}
decisions=[]
for r in rows:
    if r.get('role')=='anchor' and r.get('parent_id'):
        witness=state_anchors[r['id']]
        assert witness['exact_numerical_state'] and witness['exact_samples'] and witness['parent_sha256']==r['parent_checkpoint_sha256']
        original=next(x for x in old+cycle1+cycle2+cycle3+cycle4 if x['id']==r['parent_id'] and x['checkpoint_sha256']==r['parent_checkpoint_sha256'])
        r['anchor_field_abs_error']=abs(r['replayed_field_mse']-float(original['replayed_field_mse']))
        r['anchor_sample_max_abs_error']=0.
    if r.get('role')=='reference':
        decision='reference';parent=None;improvement=None;sampling_change=None
    elif r.get('role')=='anchor':
        assert r['anchor_field_abs_error']<2e-6 and r['anchor_sample_max_abs_error']<2e-5
        decision='anchor';parent=None;improvement=None;sampling_change=None
    else:
        if r.get('relaxation') in ['shape','scale']:
            reference=old+cycle1+cycle2+cycle3+cycle4;parent_family=r['family']
        elif r['family']=='pd_factorized_affine':
            reference,parent_family=cycle2,'pd_factorized_linear'
        elif r['family'] in ['pd_radial_k1','pd_scaled_raw_k1']:
            reference,parent_family=cycle3,'pd_direct_raw_k1'
        elif r['family']=='pd_factorized_linear':reference,parent_family=cycle1,'pd_linear_sigmoid'
        elif r['family']=='pd_direct_raw_k1':reference,parent_family=cycle2,'pd_direct_raw'
        elif r['family']=='pd_direct_abs_k1':reference,parent_family=cycle3,'pd_direct_raw_k1'
        elif r['family']=='pd_direct_bounded_k1':reference,parent_family=cycle2,'pd_direct_bounded'
        elif r['family'] in ['pd_constant_k1','pd_direct_shift_k1']:reference,parent_family=cycle3,'pd_direct_raw_k1'
        elif r['family']=='scalar_k1':reference,parent_family=old,'scalar_k2'
        else:reference,parent_family=old,'vector_pd_k2'
        parent=next(x for x in reference if x['dataset']==r['dataset'] and x['family']==parent_family and x['optimizer']==r['optimizer'] and float(x['projection_lr'])==r['projection_lr'])
        if r.get('relaxation') in ['shape','scale']:assert parent['id']==r['parent_id'] and parent['checkpoint_sha256']==r['parent_checkpoint_sha256']
        if r['family'] in ['pd_radial_k1','pd_scaled_raw_k1','pd_factorized_affine']:assert parent['id']==r['parent_id'] and parent['checkpoint_sha256']==r['parent_checkpoint_sha256']
        improvement=1-r['replayed_field_mse']/float(parent['replayed_field_mse'])
        sampling_change=r['numpy_test_sw2']/float(parent['numpy_test_sw2'])-1
        decision='keep' if improvement>=.05 else 'discard'
    ordinary=min(float(x['replayed_field_mse']) for x in old+cycle3+cycle4+rows if x['dataset']==r['dataset'] and (x['family'].startswith(('swiglu','bilinear')) or x['family']=='pd_constant_k1'))
    decisions.append(dict(cell=r['id'],dataset=r['dataset'],family=r['family'],optimizer=r['optimizer'],lr=r['projection_lr'],status=decision,
        field_mse=r['replayed_field_mse'],test_sw2=r['numpy_test_sw2'],parent=parent['id'] if parent else None,
        field_relative_improvement=improvement,test_sw2_relative_change=sampling_change,better_field_than_strongest_ordinary=r['replayed_field_mse']<ordinary))
    if r['family'] in ['pd_constant_k1','pd_direct_bounded_k1','pd_direct_shift_k1']:
        raw=next(x for x in cycle3 if x['dataset']==r['dataset'] and x['family']=='pd_direct_raw_k1' and x['optimizer']==r['optimizer'] and float(x['projection_lr'])==r['projection_lr'])
        decisions[-1].update(raw_k1_field_relative_improvement=1-r['replayed_field_mse']/float(raw['replayed_field_mse']),raw_k1_test_sw2_relative_change=r['numpy_test_sw2']/float(raw['numpy_test_sw2'])-1)
save('endpoints.csv',rows);save('decisions.csv',decisions)
for name in ['curves','updates']:save(name+'.csv',[r for d in receipts for r in d[name]])
verification=dict(endpoints=len(rows),max_field_replay_error=max(r['field_abs_error'] for r in rows),max_sw2_error=max(r['sw2_abs_error'] for r in rows),max_generated_error=max(r['generated_replay_max_abs_error'] for r in rows),max_anchor_field_error=max(r.get('anchor_field_abs_error',0) for r in rows),max_anchor_generated_error=max(r.get('anchor_sample_max_abs_error',0) for r in rows))
verification['endpoint_steps']=sorted({receipt.get('endpoint_step',20000) for receipt in receipts})
(out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
ledger=Path(a.ledger);ledger.parent.mkdir(parents=True,exist_ok=True)
exists=ledger.exists()
assert not exists or a.append_ledger,'Existing ledger requires explicit append; never overwrite'
if exists:
    previous={line.split('\t')[-1] for line in ledger.read_text().splitlines()[1:]}
    assert not previous.intersection(row['cell'] for row in decisions),'Duplicate trial ledger entries'
with ledger.open('a' if exists else 'x') as f:
    if not exists:f.write('commit\tvalidation_mse\ttest_sw2\tstatus\tdescription\n')
    for row in decisions:f.write(f"{a.runtime_commit}\t{row['field_mse']:.10g}\t{row['test_sw2']:.10g}\t{row['status']}\t{row['cell']}\n")
for row in decisions:print(json.dumps(row))
print(json.dumps(verification))
