"""Audit completed clock-study artifacts and write compact comparison tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.train import sha256


def analyze(study: Path, output: Path) -> dict:
    status=json.loads((study/'status.json').read_text())
    assert status['state']=='COMPLETE' and status['completed']==6
    provenance=json.loads((study/'provenance.json').read_text())
    for name,digest in provenance['source_sha256'].items():
        assert sha256(ROOT/name.replace('\\','/'))==digest,name
    assert sha256(ROOT/'docs/research-2026-10-24-lenovo/protocol.md')==provenance['protocol_sha256']
    cfg=json.loads((study/'config.json').read_text())
    rows=[]
    arrays_checked=0
    checkpoints_checked=0
    common={}
    for dataset in cfg['datasets']:
        initial={}
        for kind in cfg['models']:
            folder=study/'runs'/cfg['experiment_id']/dataset/kind/'seed000'
            assert (folder/'COMPLETE').read_text().strip()=='5000'
            metrics=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            assert [r['step'] for r in metrics]==list(range(0,5001,500))
            assert sha256(folder/'config.json')==sha256(study/'config.json')
            for step in range(0,5001,1000):
                cp=torch.load(folder/f'checkpoints/step_{step:07d}.pt',map_location='cpu',weights_only=False)
                assert cp['step']==step
                assert cp['identity']==dict(dataset=dataset,model=kind,seed=0)
                assert cp['config_sha256']==sha256(study/'config.json')
                assert cp['dataset_manifest_sha256']==provenance['dataset_manifest_sha256']
                if step==0:
                    initial[kind]=cp['model']
                checkpoints_checked+=1
            diagnostic=study/'diagnostics'/dataset/kind
            summary=json.loads((diagnostic/'summary.json').read_text())
            assert summary['checkpoint_sha256']==sha256(folder/'checkpoints/step_0005000.pt')
            recomputed=[]
            for name,digest in summary['arrays_sha256'].items():
                path=diagnostic/name
                assert sha256(path)==digest,name
                arrays_checked+=1
                with np.load(path,allow_pickle=False) as a:
                    assert all(np.isfinite(a[key]).all() for key in a.files)
                    if name.startswith('native-'):
                        key=(dataset,name)
                        if key in common:
                            np.testing.assert_array_equal(a['x'],common[key])
                        else:
                            common[key]=a['x'].copy()
                        recomputed.append((np.mean((a['prediction']-a['oracle'])**2),
                            np.mean(np.sum((a['jacobian']-a['oracle_jacobian'])**2,axis=(1,2)))))
            field,jac=np.mean(recomputed,axis=0)
            native=summary['aggregate']['native']
            np.testing.assert_allclose([field,jac],[native['velocity_mse'],native['jacobian_mse']],rtol=1e-12)
            final=metrics[-1]
            rows.append(dict(dataset=dataset,model=kind,step=5000,
                validation_oracle_mse=final['validation_oracle_mse'],
                validation_sw2=final['sample']['sliced_w2'],
                common_field_mse=float(field),common_jacobian_error=float(jac),
                training_wall_seconds=final['wall_seconds']))
        for key,value in initial['ode_original'].items():
            torch.testing.assert_close(value,initial['ode_gate_linear'][key],atol=0,rtol=0)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit=dict(state='PASS',training_cells=6,checkpoints_checked=checkpoints_checked,
        diagnostic_arrays_checked=arrays_checked,common_states_match=True,
        native_control_initial_parameters_identical=True,
        source_and_protocol_match=True,status=status,
        decision=json.loads((study/'decision.json').read_text()),
        benchmark=json.loads((study/'benchmark.json').read_text()),rows=rows)
    (output/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.study,args.output)
    print(json.dumps({key:result[key] for key in ('state','checkpoints_checked','diagnostic_arrays_checked','rows')},indent=2))
