"""Professor-guided candidates; the archived FM trainer/evaluator stay fixed."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as base
import torch
from torch import nn
from synthetic.models import VectorODELayer


class ConditionalSpectrumLayer(VectorODELayer):
    @torch.no_grad()
    def activation_stats(self, x, context=None):
        stats = super().activation_stats(x, context)
        d = self.step_size(x, context).reshape(-1, x.shape[-1])[:2]
        p = self.operator()
        # Column scaling is P D for column-vector conventions.
        stats['conditional_generator_spectra'] = [base.spectral(p*row[None, :]) for row in d]
        return stats


def build_model(dim, family):
    if family in ('pd_linear_sigmoid', 'pd_mlp_affine'):
        model = base.build_model(dim, 'vector_pd_k2')
        for _, layer in model.ode_layers():
            if family == 'pd_linear_sigmoid':
                first, _, last = layer.t_func
                with torch.random.fork_rng(devices=[]):
                    affine = nn.Linear(64, 64)
                with torch.no_grad():
                    probability = torch.sigmoid(first.bias)
                    slope = probability*(1+first.bias*(1-probability))
                    affine.weight.copy_((last.weight*slope[None, :])@first.weight)
                    affine.bias.copy_(last(torch.nn.functional.silu(first.bias)))
                layer.t_func = nn.Sequential(nn.Identity(), nn.Identity(), affine)
            else:
                layer.gate_transform = 'affine'
            layer.__class__ = ConditionalSpectrumLayer
        return model
    model = base.build_model(dim, family)
    if family == 'vector_pd_k2':
        for _, layer in model.ode_layers():
            layer.__class__ = ConditionalSpectrumLayer
    return model


if __name__ == '__main__':
    base.execute(base.argument_parser().parse_args(),
                 grid_path=Path(__file__).with_name('grid.json'),
                 experiment_id='professor_autoresearch_20261103', model_builder=build_model)
