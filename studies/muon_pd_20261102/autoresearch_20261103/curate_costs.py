"""Attach recorded native wall times to independently replayed endpoints."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--raw-root',required=True);p.add_argument('--out',required=True);p.add_argument('--cycles',nargs='+',type=int,default=[1,2,3,4,5]);a=p.parse_args()
study=Path(__file__).resolve().parents[3]/'docs/autoresearch/professor-20261103';rows=[];expected=0
for cycle in a.cycles:
    replay={r['id']:r for r in csv.DictReader((study/f'cycle{cycle}-results/endpoints.csv').open())}
    expected+=len(replay)
    seen=set()
    for host in ['145','lenovo']:
        raw=json.loads((Path(a.raw_root)/f'nov03-runtime-{host}-cycle{cycle}.json').read_text())
        assert raw['replayed'] is False
        for row in raw['recorded_endpoints']:
            parent=replay[row['id']];assert parent['host']==host and row['id'] not in seen
            assert abs(row['field_mse']-float(parent['replayed_field_mse']))<2e-6
            assert abs(row['test_sw2']-float(parent['numpy_test_sw2']))<2e-6
            assert row['wall_seconds']>0;seen.add(row['id'])
            rows.append(dict(cycle=cycle,host=host,cell=row['id'],dataset=row['dataset'],family=row['family'],optimizer=row['optimizer'],lr=row['projection_lr'],native_training_wall_seconds=row['wall_seconds'],config_sha256=row['config_sha256']))
    assert seen==set(replay)
assert len(rows)==expected
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
with (out/'trajectories.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(rows=len(rows),sum_process_wall_seconds=sum(r['native_training_wall_seconds'] for r in rows))))
