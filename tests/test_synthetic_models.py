"""Checks original Taylor semantics, differentiable cap, and time-series logging."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from synthetic.models import OriginalODELayer, OriginalODEFFN, VelocityNet
from synthetic.diagnostics import collect
from synthetic.prepare import prepare
from synthetic.train import run


class ModelTests(unittest.TestCase):
    def test_professor_taylor_formula(self):
        torch.manual_seed(3)
        x=torch.randn(9,8)
        layer=OriginalODELayer(8,12,orders=2)
        s=layer.step_size(x)
        p=layer.proj.weight
        expected=x+s*(x@p.T)+0.5*s.square()*(x@p.T@p.T)
        torch.testing.assert_close(layer(x),expected,atol=1e-6,rtol=1e-6)
        self.assertTrue((s>=0.2).all() and (s<=1.0).all())
        ffn=OriginalODEFFN(8,12,orders=2)
        torch.testing.assert_close(ffn(x),ffn.ode2(ffn.act(ffn.ode1(x))))

    def test_cap_and_gradients(self):
        torch.manual_seed(4)
        x=torch.randn(6,8)
        layer=OriginalODELayer(8,8,orders=2,spectral_cap=0.5)
        with torch.no_grad():
            layer.proj.weight.mul_(10)
        sigma=torch.linalg.matrix_norm(layer.operator(),ord=2)
        self.assertLessEqual(sigma.item(),0.500001)
        loss=layer(x).square().mean()
        loss.backward()
        self.assertIsNotNone(layer.proj.weight.grad)
        self.assertTrue(torch.isfinite(layer.proj.weight.grad).all())
        self.assertGreater(layer.proj.weight.grad.abs().sum().item(),0)

    def test_equal_interface_and_actual_hidden_probe(self):
        x=torch.randn(16,2)
        t=torch.rand(16,1)
        counts={}
        for kind in ('ffn','ode_original','ode_spectral_cap'):
            net=VelocityNet(2,kind,width=16,depth=2,gate_hidden=16,ffn_hidden=32)
            pred=net(x,t)
            self.assertEqual(pred.shape,x.shape)
            self.assertTrue(torch.all(pred==0))
            records=collect(net,x,t)
            self.assertEqual(len(records),0 if kind=='ffn' else 4)
            if records:
                self.assertIn('term2_rms_over_input',records[0]['activation'])
                if kind=='ode_spectral_cap':
                    self.assertLessEqual(records[0]['effective']['sigma_max'],1.00001)
            counts[kind]=net.parameter_count()
        self.assertLess(abs(counts['ffn']-counts['ode_original'])/counts['ffn'],0.08)

    def test_resume_matches_uninterrupted_and_keeps_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg_data=json.loads(Path('configs/synthetic/data_v1.json').read_text())
            cfg_data.update(datasets=['gaussian_aniso_2d'],train_size=64,
                            validation_size=64,test_size=64,probe_size=64)
            data_config=root/'data_config.json'
            data_config.write_text(json.dumps(cfg_data))
            dataset=root/'data'
            prepare(data_config,dataset)
            cfg=json.loads(Path('configs/synthetic/train_v1.json').read_text())
            cfg.update(data_root=str(dataset),datasets=['gaussian_aniso_2d'],models=['ode_original'],
                       seeds=[11],steps=3,batch_size=8,eval_interval=1,
                       sample_interval=2,checkpoint_interval=2,
                       sample_count=16,sliced_directions=8,heun_steps=2)
            cfg['model_options'].update(width=8,depth=1,gate_hidden=8,ffn_hidden=16)
            config_path=root/'train_config.json'
            config_path.write_text(json.dumps(cfg))
            def options(output,steps=None,resume=None):
                return SimpleNamespace(config=str(config_path),dataset='gaussian_aniso_2d',
                    model='ode_original',seed=11,device='cpu',output_root=str(output),
                    steps=steps,dry_run=False,resume=resume)
            full=root/'full'
            resumed=root/'resumed'
            run(options(full))
            run(options(resumed,steps=2))
            folder=resumed/cfg['experiment_id']/'gaussian_aniso_2d'/'ode_original'/'seed011'
            snapshot=folder/'checkpoints'/'step_0000002.pt'
            run(options(resumed,steps=3,resume=str(snapshot)))
            full_folder=full/cfg['experiment_id']/'gaussian_aniso_2d'/'ode_original'/'seed011'
            self.assertTrue((folder/'checkpoints'/'step_0000000.pt').exists())
            self.assertTrue((folder/'checkpoints'/'step_0000003.pt').exists())
            a=torch.load(full_folder/'checkpoints'/'step_0000003.pt',weights_only=False,map_location='cpu')
            b=torch.load(folder/'checkpoints'/'step_0000003.pt',weights_only=False,map_location='cpu')
            for key in a['model']:
                torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
            steps=[json.loads(line)['step'] for line in (folder/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(steps,[0,1,2,3])
            with self.assertRaises(FileExistsError):
                run(options(resumed))


if __name__=='__main__':
    unittest.main()
