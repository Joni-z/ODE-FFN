"""Gate ablation contracts: pairing, derivatives, serialization and data isolation."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from synthetic.models import VelocityNet, OriginalODELayer
from synthetic.train import fixed_eval_data, run
from synthetic.prepare import prepare
from scripts.diagnose_synthetic import layer_diagnostics, polynomial
from scripts.run_gate_study import select_checkpoint


class GateStudyTests(unittest.TestCase):
    def test_constant_duration_is_absorbed_by_p(self) -> None:
        layer = OriginalODELayer(8, 8, orders=4, gate_source="scalar").double()
        fixed = OriginalODELayer(8, 8, orders=4, gate_source="fixed").double()
        with torch.no_grad():
            layer.duration_logit.fill_(8)
            s = layer.step_size(torch.zeros(1, 8, dtype=torch.float64))[0, 0]
            fixed.proj.weight.copy_(layer.proj.weight * s / .6)
        h = torch.randn(13, 8, dtype=torch.float64)
        torch.testing.assert_close(layer(h), fixed(h), atol=1e-12, rtol=1e-12)

    def test_selection_ignores_test_and_breaks_ties_early(self) -> None:
        rows = [{"step": 1000, "sample": {"sliced_w2": .1}, "test_sw2": 100},
                {"step": 2000, "sample": {"sliced_w2": .1}, "test_sw2": 0},
                {"step": 0, "sample": {"sliced_w2": 0}}]
        self.assertEqual(select_checkpoint(rows, [1000, 2000]), 1000)
        with self.assertRaises(ValueError):
            select_checkpoint(rows, [1000, 3000])

    def test_paired_common_weights_and_no_dormant_gate(self) -> None:
        torch.manual_seed(0)
        original = VelocityNet(2, "ode_original")
        for kind in ("ode_fixed", "ode_scalar"):
            torch.manual_seed(0)
            model = VelocityNet(2, kind)
            for name, value in model.state_dict().items():
                if "duration_logit" not in name:
                    torch.testing.assert_close(value, original.state_dict()[name], rtol=0, atol=0)
            self.assertFalse(any("t_func" in n for n, _ in model.named_parameters()))
            clone = VelocityNet(2, kind)
            clone.load_state_dict(model.state_dict(), strict=True)
            with torch.no_grad():
                model.output.weight.normal_()
                clone.load_state_dict(model.state_dict())
            x, t = torch.randn(9, 2), torch.rand(9, 1)
            torch.testing.assert_close(model(x, t), clone(x, t), atol=0, rtol=0)
            model(x, t).square().mean().backward()
            if kind == "ode_scalar":
                for _, layer in model.ode_layers():
                    self.assertGreater(layer.duration_logit.grad.abs().item(), 0)

    def test_scalar_fixed_equivalence_and_zero_gate_jacobian(self) -> None:
        x = torch.randn(8, dtype=torch.float64)
        for source in ("fixed", "scalar"):
            layer = OriginalODELayer(8, 8, gate_source=source).double()
            jac = torch.autograd.functional.jacobian(layer, x)
            expected = polynomial(layer.operator(), x.new_tensor([0.6]), 2)[0]
            torch.testing.assert_close(jac, expected, atol=1e-12, rtol=1e-12)
            model = VelocityNet(2, f"ode_{source}", width=8, depth=1, gate_hidden=8)
            records = layer_diagnostics(model, torch.randn(16, 2), torch.rand(16, 1))
            self.assertTrue(all(r["gate_correction_norm"]["max"] == 0 for r in records))
        torch.manual_seed(0)
        fixed = VelocityNet(2, "ode_fixed")
        torch.manual_seed(0)
        scalar = VelocityNet(2, "ode_scalar")
        with torch.no_grad():
            fixed.output.weight.fill_(0.1)
            scalar.output.weight.fill_(0.1)
        xx, tt = torch.randn(16, 2), torch.rand(16, 1)
        torch.testing.assert_close(fixed(xx, tt), scalar(xx, tt), atol=0, rtol=0)

    def test_validation_sampling_never_opens_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rng = np.random.default_rng(0)
            val = rng.normal(size=(32, 2)).astype(np.float32)
            np.savez(folder / "validation.npz", x1=val)
            np.savez(folder / "probe.npz", xt=val, t=np.ones((32, 1), np.float32) * .5)
            fixed = fixed_eval_data(folder, None, 2, 0, "cpu", 8, 4, "validation")
            torch.testing.assert_close(fixed["sample_target"], torch.from_numpy(val[:8]))
            # Test file is deliberately absent: validation must succeed without it.
            with self.assertRaises(FileNotFoundError):
                fixed_eval_data(folder, None, 2, 0, "cpu", 8, 4)

    def test_new_variants_train_and_resume_without_test_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_cfg = json.loads(Path("configs/synthetic/data_v1.json").read_text())
            data_cfg.update(datasets=["gaussian_aniso_2d"], train_size=64,
                            validation_size=64, test_size=64, probe_size=64)
            (root / "data.json").write_text(json.dumps(data_cfg))
            prepare(root / "data.json", root / "data")
            (root / "data/gaussian_aniso_2d/test.npz").unlink()
            for kind in ("ode_fixed", "ode_scalar"):
                cfg = json.loads(Path(f"configs/synthetic/gates_20261021/{kind}.json").read_text())
                cfg.update(data_root=str(root / "data"), datasets=["gaussian_aniso_2d"],
                           steps=3, batch_size=8, eval_interval=1, sample_interval=1,
                           checkpoint_interval=1, sample_count=16,
                           sliced_directions=8, heun_steps=2)
                cfg["model_options"].update(width=8, depth=1, gate_hidden=8)
                config = root / f"{kind}.json"
                config.write_text(json.dumps(cfg))
                args = SimpleNamespace(config=str(config), dataset="gaussian_aniso_2d",
                                       model=kind, seed=0, device="cpu", output_root=str(root / "runs"),
                                       steps=2, dry_run=False, resume=None)
                run(args)
                folder = root / "runs" / cfg["experiment_id"] / args.dataset / kind / "seed000"
                args.steps = 3
                args.resume = str(folder / "checkpoints/step_0000002.pt")
                run(args)
                self.assertEqual((folder / "COMPLETE").read_text().strip(), "3")
                state = torch.load(folder / "checkpoints/step_0000003.pt", weights_only=False)
                self.assertEqual(state["identity"]["seed"], 0)
                self.assertTrue(all(torch.isfinite(v).all() for v in state["model"].values()))


if __name__ == "__main__":
    unittest.main()
