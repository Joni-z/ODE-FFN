"""Actual causal-control checks for gate context and budget checkpointing."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from synthetic.models import VelocityNet, OriginalODELayer
from synthetic.diagnostics import collect
from synthetic.train import run
from synthetic.prepare import prepare
from scripts.diagnose_synthetic import polynomial, layer_diagnostics


class StageTests(unittest.TestCase):
    def test_legacy_and_paired_initialization(self):
        root=Path(__file__).resolve().parents[1]
        fixture=root/'docs/research-2026-10-20/models-before-waves.py'
        spec=importlib.util.spec_from_file_location('old',fixture)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        for k in (1,2):
            for kind in ('ffn','swiglu','ode_original','ode_spectral_cap'):
                torch.manual_seed(0);a=old.VelocityNet(2,kind,orders=k)
                torch.manual_seed(0);b=VelocityNet(2,kind,orders=k)
                for name,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[name],atol=0,rtol=0)
                # Nonzero head checks the whole network rather than masking a changed FFN.
                with torch.no_grad():a.output.weight.fill_(.1);b.output.weight.fill_(.1)
                x=torch.randn(9,2);t=torch.rand(9,1)
                torch.testing.assert_close(a(x,t),b(x,t),atol=0,rtol=0)
        states=[]
        for kind,k in [('ode_original',1),('ode_time',1),('ode_original',2),('ode_time',2),('ode_original',4),('ode_original',8)]:
            torch.manual_seed(0);states.append(VelocityNet(2,kind,orders=k).state_dict())
        for st in states[1:]:
            for name,value in states[0].items():torch.testing.assert_close(value,st[name],atol=0,rtol=0)

    def test_time_gate_jacobian_and_context_logging(self):
        torch.manual_seed(0)
        layer=OriginalODELayer(8,8,orders=2,gate_source='time').double()
        h=torch.randn(8,dtype=torch.float64);c=torch.randn(8,dtype=torch.float64)
        jac=torch.autograd.functional.jacobian(lambda x:layer(x,c),h)
        s=layer.step_size(h,c).reshape(1)
        expected=polynomial(layer.operator(),s,2)[0]
        torch.testing.assert_close(jac,expected,atol=1e-12,rtol=1e-12)
        net=VelocityNet(2,'ode_time',width=8,depth=1,gate_hidden=8,ffn_hidden=16)
        with torch.no_grad():net.output.weight.normal_()
        x=torch.randn(16,2);t=torch.rand(16,1)
        recorded=collect(net,x,t)
        context=net.time(t)
        for rec,(_,mod) in zip(recorded,net.ode_layers()):
            expected=mod.step_size(torch.zeros_like(context),context).mean().item()
            self.assertAlmostEqual(rec['activation']['s_mean'],expected,places=6)
        ds=layer_diagnostics(net,x,t)
        self.assertTrue(all(r['gate_correction_norm']['max']==0 for r in ds))
        net(x,t).square().mean().backward()
        for _,mod in net.ode_layers():
            self.assertGreater(mod.t_func[0].weight.grad.abs().sum().item(),0)
        with self.assertRaises(ValueError):layer(h)

    def test_milestones_do_not_change_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            dc=json.loads(Path('configs/synthetic/data_v1.json').read_text())
            dc.update(datasets=['gaussian_aniso_2d'],train_size=64,validation_size=64,test_size=64,probe_size=64)
            (p/'data.json').write_text(json.dumps(dc));prepare(p/'data.json',p/'data')
            cfg=json.loads(Path('configs/synthetic/train_swiglu_v1.json').read_text())
            cfg.update(data_root=str(p/'data'),datasets=['gaussian_aniso_2d'],models=['ode_time'],steps=4,
                       batch_size=8,eval_interval=4,sample_interval=4,checkpoint_interval=4,sample_count=16,sliced_directions=8,heun_steps=2)
            cfg['model_options'].update(width=8,depth=1,gate_hidden=8,ffn_hidden=16)
            checkpoints=[]
            for name,extra in [('control',[]),('milestones',[2,3])]:
                cc={**cfg,'experiment_id':name,'milestone_steps':extra}
                path=p/(name+'.json');path.write_text(json.dumps(cc))
                run(SimpleNamespace(config=str(path),dataset='gaussian_aniso_2d',model='ode_time',seed=0,
                                    device='cpu',output_root=str(p/'runs'),steps=None,dry_run=False,resume=None))
                folder=p/'runs'/name/'gaussian_aniso_2d'/'ode_time'/'seed000'
                for step in extra:self.assertTrue((folder/'checkpoints'/f'step_{step:07d}.pt').exists())
                checkpoints.append(torch.load(folder/'checkpoints/step_0000004.pt',weights_only=False))
            for key,value in checkpoints[0]['model'].items():
                torch.testing.assert_close(value,checkpoints[1]['model'][key],atol=0,rtol=0)
            torch.testing.assert_close(checkpoints[0]['data_rng'],checkpoints[1]['data_rng'],atol=0,rtol=0)

if __name__=='__main__':unittest.main()
