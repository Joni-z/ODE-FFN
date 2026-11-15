"""Exact low-dimensional derivatives around the unchanged native Heun sampler."""
import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('density_symmetry', HERE.parent/'sharing_symmetry_20261116/common.py')
symmetry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(symmetry)
torch = symmetry.torch
base = symmetry.base


class MixtureOracle(torch.nn.Module):
    """Analytic derivatives of the pinned native mixture laws, independently checked."""
    def __init__(self, law):
        super().__init__()
        self.dim = law.dim
        self.product = hasattr(law, 'locations')
        self.register_buffer('rotation', torch.as_tensor(getattr(law, 'rotation', None), dtype=torch.double)
                             if hasattr(law, 'rotation') else torch.eye(law.dim, dtype=torch.double))
        if self.product:
            self.register_buffer('locations', torch.as_tensor(law.locations.copy(), dtype=torch.double))
            self.register_buffer('variances', torch.as_tensor(law.variances.copy(), dtype=torch.double))
        else:
            raw = getattr(law, 'base', law)
            self.register_buffer('means', torch.as_tensor(raw.means.copy(), dtype=torch.double))
            self.register_buffer('variances', torch.as_tensor(raw.variances.copy(), dtype=torch.double))
            self.register_buffer('weights', torch.as_tensor(raw.weights.copy(), dtype=torch.double))

    def evaluate(self, x, t):
        assert x.ndim == 2 and x.shape[1] == self.dim and t.shape == (len(x), 1)
        z = x@self.rotation
        if self.product:
            c = (1-t).square()+t.square()*self.variances
            a = (t*self.variances-(1-t))/c
            u = t*self.locations*z/c
            sign = torch.tanh(u)
            velocity = a*z+(1-t)*self.locations*sign/c
            score = (-z+t*self.locations*sign)/c
            divergence = (a+t*(1-t)*self.locations.square()*(1-sign.square())/c.square()).sum(1)
            logp = (-.5*torch.log(2*math.pi*c)-(z.square()+t.square()*self.locations.square())/(2*c)
                    +torch.logaddexp(u, -u)-math.log(2)).sum(1)
        else:
            tt = t[:, :, None]
            c = (1-tt).square()+tt.square()*self.variances
            residual = z[:, None]-tt*self.means
            logits = torch.log(self.weights)-.5*(torch.log(2*math.pi*c)+residual.square()/c).sum(2)
            logp = torch.logsumexp(logits, 1)
            responsibility = torch.softmax(logits, 1)
            component_score = -residual/c
            a = (tt*self.variances-(1-tt))/c
            component_velocity = self.means+a*residual
            velocity = (responsibility[:, :, None]*component_velocity).sum(1)
            score = (responsibility[:, :, None]*component_score).sum(1)
            divergence = (responsibility*(a.sum(2)+((component_velocity-velocity[:, None])*component_score).sum(2))).sum(1)
        return dict(velocity=velocity@self.rotation.T, score=score@self.rotation.T,
                    divergence=divergence, log_density=logp)


def exact_jacobian(model, x, t):
    """One reverse pass per output coordinate; models are sample-separable."""
    with torch.enable_grad():
        state = x.detach().requires_grad_(True)
        value = model(state, t.detach())
        rows = [torch.autograd.grad(value[:, j].sum(), state, retain_graph=j < x.shape[1]-1)[0]
                for j in range(x.shape[1])]
    return value.detach(), torch.stack(rows, 1).detach()


def normal_log_density(x):
    return -.5*(x.square().sum(1)+x.shape[1]*math.log(2*math.pi))


class DensityTracker(torch.nn.Module):
    """Observer at the native sampler's real stages; returns its unaltered velocity."""
    def __init__(self, model, oracle, source, steps, progress=None):
        super().__init__()
        assert steps > 0 and steps % 8 == 0
        self.model, self.oracle, self.steps = model, oracle, steps
        self.callback, self.calls = progress, 0
        self.first = None
        self.integral_div = torch.zeros(len(source), dtype=source.dtype, device=source.device)
        self.integral_source = torch.zeros_like(self.integral_div)
        self.integral_local_logdet = torch.zeros_like(self.integral_div)
        self.time_parts = torch.zeros((len(source), 8, 3), dtype=source.dtype, device=source.device)
        self.identity = torch.eye(source.shape[1], dtype=source.dtype, device=source.device)
        self.nonpositive_steps = torch.zeros((), dtype=torch.long, device=source.device)
        self.jacobian_value_error = torch.zeros((), dtype=source.dtype, device=source.device)
        self.max_field_jacobian = torch.zeros_like(self.jacobian_value_error)

    def forward(self, x, t):
        index, stage = divmod(self.calls, 2)
        expected = (index+stage)/self.steps
        assert index < self.steps and bool(torch.all(torch.abs(t-expected) < 1e-13))
        # This is the value used by the original sampler, with its original context.
        with torch.no_grad():
            value = self.model(x, t)
        derivative_value, jacobian = exact_jacobian(self.model, x, t)
        with torch.no_grad():
            self.jacobian_value_error = torch.maximum(self.jacobian_value_error, (derivative_value-value).abs().max())
            self.max_field_jacobian = torch.maximum(self.max_field_jacobian, jacobian.square().sum((1, 2)).sqrt().max())
            true = self.oracle.evaluate(x, t)
            error = value-true['velocity']
            div = jacobian.diagonal(dim1=1, dim2=2).sum(1)
            div_error = div-true['divergence']
            alignment = (error*true['score']).sum(1)
            source = div_error+alignment
            parts = torch.stack((div_error, alignment, error.square().mean(1)), 1)
            self.time_parts[:, index*8//self.steps] += .5/self.steps*parts
            if stage == 0:
                self.first = (jacobian, div, source)
            else:
                first_jac, first_div, first_source = self.first
                dt = 1/self.steps
                step_jac = self.identity+.5*dt*(first_jac+jacobian+dt*(jacobian@first_jac))
                sign, logdet = torch.linalg.slogdet(step_jac)
                self.nonpositive_steps += (sign <= 0).sum()
                self.integral_local_logdet += logdet
                self.integral_div += .5*dt*(first_div+div)
                self.integral_source += .5*dt*(first_source+source)
                self.first = None
            self.calls += 1
            if self.callback is not None and stage == 1 and ((index+1) % 32 == 0 or index+1 == self.steps):
                self.callback(index+1, self.steps)
        return value


def sample_density(model, oracle, source, steps, progress=None):
    observer = DensityTracker(model, oracle, source, steps, progress)
    generated = base.train.generated_samples(observer, source, steps)
    assert observer.calls == 2*steps and observer.first is None
    with torch.no_grad():
        logp = oracle.evaluate(generated, torch.ones_like(source[:, :1]))['log_density']
        start = normal_log_density(source)
        source_ratio = -observer.integral_source
        divergence_ratio = start-observer.integral_div-logp
        discrete_local_ratio = start-observer.integral_local_logdet-logp
        result = dict(generated=generated, log_ratio_source=source_ratio,
                      log_ratio_divergence=divergence_ratio,
                      log_ratio_discrete_local=discrete_local_ratio,
                      chain_residual=source_ratio-divergence_ratio,
                      time_parts=observer.time_parts, integral_divergence=observer.integral_div,
                      integral_local_logdet=observer.integral_local_logdet,
                      endpoint_log_target=logp,
                      nonpositive_steps=observer.nonpositive_steps,
                      jacobian_value_error=observer.jacobian_value_error,
                      max_field_jacobian=observer.max_field_jacobian)
    assert all(bool(torch.isfinite(x).all()) for x in result.values())
    assert torch.allclose(observer.time_parts[:, :, :2].sum((1, 2)), -source_ratio, atol=1e-10, rtol=1e-10)
    return result
