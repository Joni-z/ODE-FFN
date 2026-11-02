"""Read-only recorded endpoints; replay validation is deliberately separate."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);p.add_argument('--save',required=True);a=p.parse_args()
root=Path(a.output_root);rows=[]
for folder in sorted((root/'professor_autoresearch_20261103').glob('*/*/seed000')):
    if not (folder/'ADAPTER_COMPLETE').exists():continue
    cfg=json.loads((folder/'config.json').read_text());m=json.loads((folder/'metrics.jsonl').read_text().splitlines()[-1]);t=json.loads((folder/'final_test_sampling.json').read_text())
    assert m['step']==20000
    row=dict(cfg['cell'],field_mse=m['validation_oracle_mse'],test_sw2=t['sample']['sliced_w2'],wall_seconds=m['wall_seconds'],config_sha256=hashlib.sha256((folder/'config.json').read_bytes()).hexdigest())
    rows.append(row)
result=dict(replayed=False,recorded_endpoints=rows)
file=Path(a.save);file.parent.mkdir(parents=True,exist_ok=True);file.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(complete=len(rows),saved=str(file))))
