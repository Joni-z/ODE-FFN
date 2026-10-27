"""Float64 nonlinear telescoping/cancellation and affine normal-gain fixtures."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def verify(core, native):
    from defect_core import advance, trace, suffix, normal_suffix_gains, account, energy, forced_readout
    rng = np.random.default_rng(0);basis = core.make_basis();source = torch.tensor(rng.standard_normal((7, 128)), dtype=torch.float64)
    maximum = 0.;normal_max = 0.;negative_seen = False
    for variance in (0., .04):
        def oracle(x, t):return torch.from_numpy(core.oracle(x.numpy(), t.numpy(), basis, variance)['oracle']).to(x.dtype)
        direction = torch.tensor(rng.standard_normal((1, 128)), dtype=torch.float64)*.01
        def model(x, t):return oracle(x, t)+torch.cos(2*torch.pi*t)*direction.to(x.dtype)+.003*torch.sin(x)
        path = trace(model, source)
        bridges = torch.stack([suffix(oracle, path[i], i) for i in range(33)]).numpy()
        local = np.stack([(path[i+1]-advance(oracle, path[i], i)[0]).numpy() for i in range(32)])
        summary, rows, signed = account(bridges, local, basis, normal_suffix_gains(variance))
        maximum = max(maximum, summary['telescoping_max'])
        normal_max = max(normal_max, summary['normal_propagation_max'], summary['normal_total_propagation_max'])
        assert normal_max<=1e-11
        negative_seen |= any(r['signed_endpoint_mse_contribution'] < -1e-12 for r in rows)
        np.testing.assert_array_equal(trace(model, source.float())[-1].numpy(), native.legacy.generated_samples(model, source.float(), 32).numpy())
    assert negative_seen
    q = np.eye(6)[:, :2];x = torch.tensor(rng.standard_normal((5, 6)), dtype=torch.float64)
    def expanding(value, t):
        first = -.2*value[:, :2]+.1*torch.sin(value[:, :2])
        return torch.cat((first, .4*value[:, 2:]), 1)
    def shifted(value, t):return expanding(value, t)+.02*torch.cos(t)
    path = trace(shifted, x);bridges = torch.stack([suffix(expanding, path[i], i) for i in range(33)]).numpy()
    local = np.stack([(path[i+1]-advance(expanding, path[i], i)[0]).numpy() for i in range(32)])
    factor = 1+.4/32+.5*(.4/32)**2;gains = factor**np.arange(32, -1, -1)
    summary, rows, _ = account(bridges, local, q, gains)
    assert max(r['finite_rms_gain'] for r in rows if r['scope']=='normal')>1.3
    assert summary['normal_propagation_max']<=1e-11
    unit = np.zeros((1, 6));unit[0, 2] = 1
    assert energy(unit, q)['normal']==.25
    state = dict(model={
        'raw.output.weight':torch.tensor(rng.standard_normal((6, 3))*.1),
        'raw.output.bias':torch.tensor(rng.standard_normal(6)*.01),
        'raw.norm.weight':torch.tensor(rng.uniform(.8, 1.2, 3)),
        'raw.norm.bias':torch.tensor(rng.standard_normal(3)*.02)})
    def raw_model(value, t):
        h = torch.sin(value[:, :3]+t);h = h-h.mean(-1, keepdim=True)
        h = h/torch.sqrt(h.square().mean(-1, keepdim=True)+1e-5)
        raw = (h*state['model']['raw.norm.weight']+state['model']['raw.norm.bias'])@state['model']['raw.output.weight'].T+state['model']['raw.output.bias']
        return (raw-value)/(1-t).clamp_min(.05)
    def reference(value, t):
        a = 1-t;gain = t*.04/(a*a+t*t*.04);beta = (gain-1)/(1-t).clamp_min(.05)
        return torch.cat((-.1*torch.sin(value[:, :2]), beta*value[:, 2:]), 1)
    path = trace(raw_model, x)
    bridges = torch.stack([suffix(reference, path[i], i) for i in range(33)]).numpy()
    forced, _, _ = forced_readout(bridges, x.numpy(), state, q, normal_suffix_gains(.04))
    assert forced['forced_bridge_prediction_max']<=1e-11
    return dict(status='PASS', mixture_vector_telescoping_max=maximum,
        mixture_normal_affine_max=normal_max, signed_cancellation_verified=negative_seen,
        expanding_suffix_verified=True, normal_dimension_normalization=True,
        native_float32_step_arithmetic_bitwise=True,
        forced_readout_bridge_max=forced['forced_bridge_prediction_max'])
