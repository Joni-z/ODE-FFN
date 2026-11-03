"""Read stored native D/amplitude statistics without executing a model."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);p.add_argument('--save',required=True);a=p.parse_args()
root=Path(a.output_root)/'professor_autoresearch_cycle10_20261103';rows=[];bindings=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for folder in sorted(root.glob('*/*/seed000')):
    assert (folder/'ADAPTER_COMPLETE').exists()
    cfg=json.loads((folder/'config.json').read_text());file=folder/'metrics.jsonl'
    bindings.append(dict(cell=cfg['cell']['id'],metrics_sha256=sha(file),config_sha256=sha(folder/'config.json')))
    for line in file.read_text().splitlines():
        m=json.loads(line)
        for layer in m['ode_layers']:
            v=dict(cell=cfg['cell']['id'],dataset=cfg['cell']['dataset'],family=cfg['cell']['family'],step=m['step'],layer=layer['layer'])
            v.update({k:x for k,x in layer['activation'].items() if isinstance(x,(float,int,str))})
            rows.append(v)
assert len(bindings)==6
result=dict(passed=True,bindings=bindings,rows=rows,scope='Stored native scalar activation statistics; no rerun or quality-based selection.')
path=Path(a.save);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(cells=len(bindings),layer_records=len(rows),saved=str(path))))
