"""Independent deterministic checks before learned transport observations."""
import argparse
import numpy as np
from common import base, torch, dump
from learned_transport import SensitivityField, BlendedField, integrate, target_derivative, squared_sw2


class Affine(torch.nn.Module):
    def __init__(self, matrix, bias):
        super().__init__()
        self.matrix, self.bias = matrix, bias

    def forward(self, x, t):
        return x@self.matrix.T+self.bias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 3, 3))*.1
    c, d = rng.normal(size=(2, 3))*.1
    source = rng.normal(size=(32, 3))
    native, crossed = (Affine(torch.tensor(m), torch.tensor(v)) for m,v in [(a,c),(b,d)])
    state = torch.cat((torch.tensor(source), torch.zeros_like(torch.tensor(source))), 1)
    field = SensitivityField(native, crossed)
    actual, _ = integrate(base.train.generated_samples, field, state)
    x, u = source.copy(), np.zeros_like(source)
    h = 1/128
    for _ in range(128):
        v = x@a.T+c
        dv = u@a.T+x@(b-a).T+d-c
        p, du = x+h*v, u+h*dv
        w = p@a.T+c
        dw = du@a.T+p@(b-a).T+d-c
        x, u = x+h/2*(v+w), u+h/2*(dv+dw)
    reference = np.concatenate((x,u), 1)
    assert np.all(np.abs(actual.numpy()-reference) <= 1e-11*(1+np.abs(reference)))
    assert field.primal_qualified
    replay = base.train.generated_samples(native, torch.tensor(source), 128)
    assert torch.equal(actual[:, :3], replay)
    finite = []
    for epsilon in (1e-4, 1e-5):
        plus = base.train.generated_samples(BlendedField(native,crossed,epsilon),torch.tensor(source),128)
        minus = base.train.generated_samples(BlendedField(native,crossed,-epsilon),torch.tensor(source),128)
        error = float(torch.max(torch.abs((plus-minus)/(2*epsilon)-actual[:,3:])))
        assert error < 1e-8
        finite.append(dict(epsilon=epsilon,error=error))
    # Rotation covariance of this independent affine recurrence.
    rotation, _ = np.linalg.qr(rng.normal(size=(3,3)))
    r0 = Affine(torch.tensor(rotation.T@a@rotation), torch.tensor(c@rotation))
    r1 = Affine(torch.tensor(rotation.T@b@rotation), torch.tensor(d@rotation))
    rotated = torch.tensor(source@rotation)
    rot, _ = integrate(base.train.generated_samples,SensitivityField(r0,r1),
                       torch.cat((rotated,torch.zeros_like(rotated)),1))
    assert np.allclose(rot[:,:3].numpy(),x@rotation,atol=1e-11,rtol=1e-11)
    assert np.allclose(rot[:,3:].numpy(),u@rotation,atol=1e-11,rtol=1e-11)
    directions = rng.normal(size=(13,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    target = rng.normal(size=(32,3))
    metric = target_derivative(x,u,target,directions)
    xt = torch.tensor(x,requires_grad=True)
    residual = torch.sort(xt@torch.tensor(directions).T,dim=0).values-torch.sort(torch.tensor(target@directions.T),dim=0).values
    gradient, = torch.autograd.grad(residual.square().mean(),xt)
    ad = float((gradient*torch.tensor(u)).sum())
    assert abs(ad-metric['loss_derivative']) < 1e-12 and metric['qualified']
    shift = np.array([.2,-.1,.3])
    translation = target_derivative(x,np.repeat(shift[None],len(x),0),x+shift,directions)
    assert abs(translation['loss_derivative']+2*translation['loss']) < 1e-12
    tied = target_derivative(np.zeros((2,1)),np.array([[1.],[-1.]]),
                              np.array([[-1.],[1.]]),np.ones((1,1)))
    assert tied['projection_ties'] == 1 and tied['loss_derivative'] == -2.
    zero = target_derivative(x,u,x,directions)
    assert zero['loss'] == 0 and zero['loss_derivative'] == 0
    assert abs(zero['sw2_derivative']-np.sqrt(np.mean((u@directions.T)**2))) < 1e-12
    assert squared_sw2(x,target,directions) == metric['loss']
    result = dict(passed=True,seed=0,native_primal_exact=True,
        independent_heun_max_error=float(np.max(np.abs(actual.numpy()-reference))),
        endpoint_finite_differences=finite,metric_autograd_error=abs(ad-metric['loss_derivative']),
        tie_right_derivative=tied,zero_distance=zero,source_sha256=base.digest(__file__))
    dump(args.output,result)
    print('TARGET_TRANSPORT_MATH_PASS',flush=True)


if __name__ == '__main__':main()
