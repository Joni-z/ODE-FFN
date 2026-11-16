"""Oracle-path error sensitivity around the unchanged native Heun integrator."""
import numpy as np
from oracle_check import MixtureOracle, torch, base

LAMBDAS = (.125, .25, .5, 1.)
RESOLUTIONS = (128, 256, 512, 1024)


class OracleResponse(MixtureOracle):
    def forward(self, x, t):
        return self.evaluate(x, t)['velocity']

    def jvp(self, x, t, direction):
        """Exact native-law J_v(x,t) direction, including t=0 and t=1."""
        z, dz = x@self.rotation, direction@self.rotation
        if self.product:
            c = (1-t).square()+t.square()*self.variances
            a = (t*self.variances-(1-t))/c
            sign = torch.tanh(t*self.locations*z/c)
            derivative = a+t*(1-t)*self.locations.square()*(1-sign.square())/c.square()
            result = derivative*dz
        else:
            tt = t[:, :, None]
            c = (1-tt).square()+tt.square()*self.variances
            residual = z[:, None]-tt*self.means
            logits = torch.log(self.weights)-.5*(torch.log(2*torch.pi*c)+residual.square()/c).sum(2)
            probability = torch.softmax(logits, 1)
            score = -residual/c
            average_score = (probability[:, :, None]*score).sum(1)
            a = (tt*self.variances-(1-tt))/c
            component_v = self.means+a*residual
            velocity = (probability[:, :, None]*component_v).sum(1)
            dlogp = ((score-average_score[:, None])*dz[:, None]).sum(2)
            result = (probability[:, :, None]*a).sum(1)*dz
            result += (probability[:, :, None]*(component_v-velocity[:, None])*dlogp[:, :, None]).sum(1)
        return result@self.rotation.T


class BlendedField(torch.nn.Module):
    def __init__(self, model, oracle, coefficient: float):
        super().__init__()
        self.model, self.oracle, self.coefficient = model, oracle, coefficient

    def forward(self, x, t):
        if self.coefficient == 1:
            return self.model(x, t)
        if self.coefficient == 0:
            return self.oracle(x, t)
        true = self.oracle(x, t)
        return true+self.coefficient*(self.model(x, t)-true)


class SensitivityField(torch.nn.Module):
    def __init__(self, model, oracle):
        super().__init__()
        self.model, self.oracle = model, oracle

    def forward(self, state, t):
        x, z = state.chunk(2, dim=1)
        true = self.oracle(x, t)
        tangent = self.oracle.jvp(x, t, z)+self.model(x, t)-true
        return torch.cat((true, tangent), dim=1)


class RecordedField(torch.nn.Module):
    """Record real accepted states; return the wrapped field without alteration."""
    def __init__(self, field, steps: int):
        super().__init__()
        assert steps > 0 and steps % 8 == 0
        self.field, self.steps, self.calls = field, steps, 0
        self.snapshots = []

    def forward(self, x, t):
        index, stage = divmod(self.calls, 2)
        assert index < self.steps
        assert bool(torch.all(torch.abs(t-(index+stage)/self.steps) < 1e-13))
        if stage == 0 and index % (self.steps//8) == 0:
            self.snapshots.append(x.detach().clone())
        self.calls += 1
        return self.field(x, t)


def integrate(field, source, steps: int):
    observer = RecordedField(field, steps)
    endpoint = base.train.generated_samples(observer, source, steps)
    assert observer.calls == 2*steps and len(observer.snapshots) == 8
    observer.snapshots.append(endpoint.clone())
    path = torch.stack(observer.snapshots)
    assert torch.isfinite(path).all()
    return endpoint, path


def observe(model, oracle, source, steps: int) -> dict:
    state = torch.cat((source, torch.zeros_like(source)), dim=1)
    augmented, path = integrate(SensitivityField(model, oracle), state, steps)
    dim = source.shape[1]
    oracle_endpoint, tangent = augmented[:, :dim], augmented[:, dim:]
    endpoints, paths = [], []
    for coefficient in LAMBDAS:
        endpoint, curve = integrate(BlendedField(model, oracle, coefficient), source, steps)
        endpoints.append(endpoint)
        paths.append(curve)
    return dict(steps=np.asarray(steps), lambdas=np.asarray(LAMBDAS),
        oracle_endpoint=oracle_endpoint.numpy(), tangent=tangent.numpy(),
        oracle_path=path[:, :, :dim].numpy(), tangent_path=path[:, :, dim:].numpy(),
        endpoints=torch.stack(endpoints).numpy(), paths=torch.stack(paths).numpy())


def response_summary(data: dict) -> dict:
    difference = data['endpoints']-data['oracle_endpoint'][None]
    remainder = difference-data['lambdas'][:, None, None]*data['tangent'][None]
    signal = np.mean(difference**2, axis=(1, 2))
    residual = np.mean(remainder**2, axis=(1, 2))
    return dict(displacement_mse=signal.tolist(), remainder_mse=residual.tolist(),
        explained_fraction=(1-residual/np.maximum(signal, np.finfo(float).tiny)).tolist())


def qualify(coarse: dict, fine: dict) -> dict:
    endpoint_ok = np.allclose(coarse['endpoints'], fine['endpoints'], atol=1e-4, rtol=1e-3)
    oracle_ok = np.allclose(coarse['oracle_endpoint'], fine['oracle_endpoint'], atol=1e-4, rtol=1e-3)
    tangent_ok = np.allclose(coarse['tangent'], fine['tangent'], atol=1e-4, rtol=1e-3)
    a, b = response_summary(coarse), response_summary(fine)
    fraction_change = abs(a['explained_fraction'][-1]-b['explained_fraction'][-1])
    return dict(passed=bool(endpoint_ok and oracle_ok and tangent_ok and fraction_change <= .01),
        coarse_steps=int(coarse['steps']), fine_steps=int(fine['steps']),
        endpoints_allclose=bool(endpoint_ok), oracle_allclose=bool(oracle_ok), tangent_allclose=bool(tangent_ok),
        endpoint_max_difference=float(np.max(np.abs(coarse['endpoints']-fine['endpoints']))),
        tangent_max_difference=float(np.max(np.abs(coarse['tangent']-fine['tangent']))),
        full_scale_fraction_change=float(fraction_change))
