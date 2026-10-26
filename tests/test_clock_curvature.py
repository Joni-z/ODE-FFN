"""Scientific controls for the exact centered K2 intervention."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from synthetic.models import OriginalODELayer, VelocityNet


class ClockCurvatureTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_removed_term_and_derivative_match_exact_identity(self) -> None:
        native = OriginalODELayer(8, 8).double()
        linear = OriginalODELayer(8, 8, clock_curvature=0).double()
        linear.load_state_dict(native.state_dict())
        with torch.no_grad():
            native.t_func[-1].weight.mul_(30)
            linear.load_state_dict(native.state_dict())
        x = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
        s, p = native.step_size(x), native.proj.weight
        residual = .5*(s-.6).square()*F.linear(F.linear(x, p), p)
        difference = native(x)-linear(x)
        torch.testing.assert_close(difference, residual, atol=1e-14, rtol=1e-12)
        actual = torch.autograd.grad(difference.sum(), x, retain_graph=True)[0]
        expected = torch.autograd.grad(residual.sum(), x)[0]
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)

    def test_equal_reference_value_and_gate_tangent(self) -> None:
        native = OriginalODELayer(5, 5).double()
        linear = OriginalODELayer(5, 5, clock_curvature=0).double()
        linear.load_state_dict(native.state_dict())
        x = torch.randn(4, 5, dtype=torch.float64)
        s = torch.full((4, 1), .6, dtype=torch.float64, requires_grad=True)
        native.step_size = lambda *_: s
        linear.step_size = lambda *_: s
        torch.testing.assert_close(native(x), linear(x), atol=0, rtol=0)
        gn = torch.autograd.grad(native(x).sum(), s)[0]
        gl = torch.autograd.grad(linear(x).sum(), s)[0]
        torch.testing.assert_close(gn, gl, atol=0, rtol=0)

    def test_parameter_initialization_and_batch_stream_remain_paired(self) -> None:
        models = []
        for kind in ('ode_original', 'ode_gate_linear'):
            torch.manual_seed(0)
            models.append(VelocityNet(2, kind, width=8, depth=2, gate_hidden=8))
        a, b = models
        self.assertEqual(a.parameter_count(), b.parameter_count())
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
        for _, layer in b.ode_layers():
            self.assertEqual(layer.clock_curvature, 0)

    def test_invalid_order_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'orders=2'):
            OriginalODELayer(4, 4, orders=1, clock_curvature=0)

    def test_checkpoint_diagnostic_runs_on_known_data_and_restores_model(self) -> None:
        from scripts.check_clock_curvature import check
        from synthetic.prepare import prepare
        from synthetic.train import sha256
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data_cfg=json.loads(Path('configs/synthetic/data_v1.json').read_text())
            data_cfg.update(datasets=['gmm_ring8_2d'],train_size=64,
                validation_size=32,test_size=32,probe_size=16)
            source=folder/'data-config.json'
            source.write_text(json.dumps(data_cfg))
            data=prepare(source,folder/'data')
            options=dict(width=8,depth=1,gate_hidden=8)
            cfg=folder/'config.json'
            cfg.write_text(json.dumps(dict(model_options=options)))
            model=VelocityNet(2,'ode_original',**options)
            with torch.no_grad():
                model.output.weight.normal_()
                for _, layer in model.ode_layers():
                    layer.t_func[-1].weight.mul_(20)
            cp=folder/'model.pt'
            torch.save(dict(model=model.state_dict(),step=1,
                identity=dict(dataset='gmm_ring8_2d',model='ode_original',seed=0),
                config_sha256=sha256(cfg),dataset_manifest_sha256=sha256(data/'manifest.json')),cp)
            result=check(cp,cfg,data,folder/'diagnostic',points=16)
            self.assertEqual(len(result['rows']),20)
            self.assertGreater(result['aggregate']['all_removed']['field_change_squared'],0)
            self.assertTrue((folder/'diagnostic/summary.json').exists())


if __name__ == '__main__':
    unittest.main()
