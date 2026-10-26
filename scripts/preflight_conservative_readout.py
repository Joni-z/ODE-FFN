"""Native-host actual-trainer and checkpoint checks, independent of task outcomes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_conservative_readout import ARMS,build,train_cell,base
from synthetic.conservative_readout import compute_budget


def run(output,data):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    hashes=json.loads((ROOT/'source-sha256.json').read_text())
    for name,value in hashes.items():
        assert base.sha256(ROOT/name)==value
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests',
        '-p','test_conservative_readout.py','-v'],cwd=ROOT,capture_output=True,text=True,timeout=120)
    (output/'unit-tests.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError('native unit tests failed')
    dataset='rotated_product_bimodal_16d'
    config=dict(seed=0,steps=3,datasets=[dataset],data_roots={dataset:str(data)})
    config_path=output/'study.json'
    base.write(config_path,config)
    base.prepare_banks(config,dataset,output)
    rows=[]
    x=torch.linspace(-1.,1.,512*16).reshape(512,16)
    t=torch.full((512,1),.4)
    for arm,setting in ARMS.items():
        model=build(16,setting['family'],setting['readout'])
        costs=compute_budget(model)
        with torch.no_grad():
            for _ in range(3):model(x,t)
            start=time.perf_counter()
            for _ in range(10):model(x,t)
        inference_seconds=(time.perf_counter()-start)/10
        train_cell(config_path,output,dataset,arm,steps=3)
        folder=output/'runs'/dataset/arm
        record=json.loads((folder/'result.json').read_text())
        state=torch.load(folder/'step_0000003.pt',map_location='cpu',weights_only=False)
        model=build(16,setting['family'],setting['readout'])
        base.restore_model(model,state)
        replay=base.field_evaluation(model,output/'banks'/dataset)
        for key,value in replay.items():
            assert value==record[key],(arm,key)
        if setting['readout']=='gradient':
            initial=torch.load(folder/'step_0000000.pt',map_location='cpu',weights_only=False)['model']
            direct=torch.load(output/'runs'/dataset/(setting['family']+'_direct')/'step_0000000.pt',map_location='cpu',weights_only=False)['model']
            assert initial.keys()==direct.keys()
            for key in initial:assert torch.equal(initial[key],direct[key]),key
        rows.append(dict(arm=arm,costs=costs,isolated_batch512_inference_seconds=inference_seconds,
            stream_digest=record['input_stream_digest'],checkpoint_reload='exact'))
        print(json.dumps(rows[-1]),flush=True)
    assert len({r['stream_digest'] for r in rows})==1
    for name,value in hashes.items():
        assert base.sha256(ROOT/name)==value
    base.write(output/'PASS.json',dict(status='PASS',unit_methods=6,cells=rows,source_sha256=hashes,
        scope='three-update actual trainer and isolated preflight timing; no formal task outcomes'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    a=p.parse_args()
    run(a.output.resolve(),a.data.resolve())
