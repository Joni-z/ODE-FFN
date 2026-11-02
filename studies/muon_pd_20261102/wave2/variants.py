"""Local counterfactuals around the pinned Muon and archived ODE layer."""
from pathlib import Path
import sys

PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARENT))
import run as base
from synthetic.models import VectorODELayer
from vendor.muon import zeropower_via_newtonschulz5
import torch
from torch.nn import functional as F


class MatchedMomentum(torch.optim.Optimizer):
    """Official Muon momentum recurrence, raw direction at its shadow Muon norm.

    The momentum/NS expressions follow vendor/muon.py at its pinned commit.
    Only the final matrix direction is replaced; this is an ablation, not a
    general-purpose optimizer proposal. Norm matching is within a current
    gradient/momentum state, not between independently evolving training runs.
    """
    def __init__(self, params, lr, momentum=.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))
        self.shadow_norms = {}

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if not state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                momentum = state['momentum_buffer']
                beta = group['momentum']
                momentum.lerp_(p.grad, 1-beta)
                raw = p.grad.lerp_(momentum, beta)
                shadow = zeropower_via_newtonschulz5(raw, steps=5)
                shadow *= max(1, shadow.size(-2)/shadow.size(-1))**.5
                target_norm = torch.linalg.vector_norm(shadow.float())
                raw_norm = torch.linalg.vector_norm(raw)
                factor = torch.where(raw_norm > 0, target_norm/raw_norm.clamp_min(1e-30),
                                     torch.zeros_like(raw_norm))
                p.add_(raw*factor, alpha=-group['lr'])
                self.shadow_norms[p] = target_norm


class SandwichLayer(VectorODELayer):
    """Same gate and P; A(h)=sqrt(D(h)) P sqrt(D(h)), frozen per solve."""
    def forward(self, x, context=None, *, return_duration=False):
        duration = self.step_size(x, context)
        root = duration.sqrt()
        term, total = x, x
        for order in range(1, self.orders+1):
            term = F.linear(term*root, self.operator())*root/order
            total = total+term
        return (total, duration) if return_duration else total

    @torch.no_grad()
    def activation_stats(self, x, context=None):
        # Reuse controller statistics, replacing every placement-dependent term.
        stats = super().activation_stats(x, context)
        root = self.step_size(x, context).sqrt()
        rms = lambda z: float(z.square().mean().sqrt())
        denominator = max(rms(x), 1e-12)
        term = x
        for order in range(1, self.orders+1):
            term = F.linear(term*root, self.operator())*root/order
            stats[f'term{order}_rms_over_input'] = rms(term)/denominator
        stats['output_rms_over_input'] = rms(self(x, context))/denominator
        stats['rate_placement'] = 'balanced_sandwich'
        # The archived centered-curvature diagnostic is specifically for DP.
        stats.pop('centered_curvature_rms_over_input', None)
        stats.pop('clock_curvature', None)
        return stats


def build_model(dim, family):
    if family != 'vector_sandwich_k2':
        return base.build_model(dim, family)
    model = base.build_model(dim, 'vector_dp_k2')
    with torch.random.fork_rng(devices=[]):
        for block in model.blocks:
            for name in ('ode1', 'ode2'):
                old = getattr(block.ffn, name)
                layer = SandwichLayer(64, 32, reference_gate_hidden=64, orders=2)
                layer.load_state_dict(old.state_dict(), strict=True)
                layer.rate_placement = 'balanced_sandwich'
                setattr(block.ffn, name, layer)
    return model


def optimizer(model, cell, log_path=None):
    if cell['optimizer'] != 'matched_momentum':
        return base.ProjectionOptimizer(model, cell['optimizer'], cell['projection_lr'],
                                        log_path, weight_decay=cell.get('projection_decay', 0.))
    result = base.ProjectionOptimizer(model, 'muon', cell['projection_lr'], log_path)
    result.projections = MatchedMomentum([p for _, p in result.selected], cell['projection_lr'])
    result.name = 'matched_momentum'
    return result
