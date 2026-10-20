"""Tiny seed0 CPU pipeline test, using temporary outputs only."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from synthetic.prepare import prepare
from synthetic.train import run,sha256
from synthetic.models import VelocityNet
from scripts.run_stage_waves import diagnose
root=Path.cwd()
with tempfile.TemporaryDirectory() as temp:
    p=Path(temp)
    dc=json.loads((root/'configs/synthetic/data_v1.json').read_text())
    dc.update(datasets=['gaussian_aniso_2d'],train_size=64,validation_size=256,test_size=64,probe_size=64)
    (p/'data.json').write_text(json.dumps(dc));prepare(p/'data.json',p/'data')
    cfg=json.loads((root/'configs/synthetic/stage_20261020/gate_k2_time.json').read_text())
    cfg.update(experiment_id='pipeline_fixture',data_root=str(p/'data'),datasets=['gaussian_aniso_2d'],steps=2,batch_size=8,
               eval_interval=1,sample_interval=1,checkpoint_interval=1,sample_count=16,sliced_directions=8)
    cfg['model_options'].update(width=8,depth=1,gate_hidden=8,ffn_hidden=16)
    path=p/'train.json';path.write_text(json.dumps(cfg))
    run(SimpleNamespace(config=str(path),dataset='gaussian_aniso_2d',model='ode_time',seed=0,device='cpu',
                        output_root=str(p/'runs'),steps=None,dry_run=False,resume=None))
    folder=p/'runs/pipeline_fixture/gaussian_aniso_2d/ode_time/seed000'
    entry={'label':'fixture','model':'ode_time','orders':2,'sha256':sha256(path),
           'parameters':VelocityNet(2,'ode_time',**cfg['model_options']).parameter_count(),'forward_macs':0}
    diagnose(folder,cfg,entry,'gaussian_aniso_2d',2,p/'result.json','cpu',root)
    result=json.loads((p/'result.json').read_text())
    assert len(result['time_bins'])==10 and len(result['sampling'])==6
    assert all(layer['gate_correction_norm']['max']==0 for layer in result['layers'])
    print('Temporary end-to-end checkpoint, provenance, NFE and time-gate diagnostics: PASS')
