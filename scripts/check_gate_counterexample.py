"""Verify the exact K=2 gated-layer counterexample with autograd."""
import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthetic.models import OriginalODELayer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layer = OriginalODELayer(2, 2, orders=2).double()
    with torch.no_grad():
        layer.proj.weight.copy_(torch.eye(2, dtype=torch.float64))
        layer.t_func[0].weight.copy_(torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64))
        layer.t_func[0].bias.zero_()
        layer.t_func[2].weight.copy_(torch.tensor([[1., -1.]], dtype=torch.float64))
        layer.t_func[2].bias.zero_()
    rows = []
    for radius in (1, 10, 100, 1000, 10000):
        h = torch.tensor([0., float(radius)], dtype=torch.float64)
        jacobian = torch.autograd.functional.jacobian(layer, h)
        expected = torch.tensor([[1.78, 0.], [.032*radius, 1.78]], dtype=torch.float64)
        torch.testing.assert_close(jacobian, expected, rtol=1e-12, atol=1e-12)
        rows.append(dict(input_radius=radius, step=layer.step_size(h).item(),
                         P_singular_values=[1., 1.], jacobian=jacobian.tolist(),
                         jacobian_eigenvalues=torch.linalg.eigvals(jacobian).real.tolist(),
                         jacobian_singular_values=torch.linalg.svdvals(jacobian).tolist()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        description="Analytic construction, not trained model evidence",
        K=2, tau=10., scale=.8, shift=.2, rows=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
