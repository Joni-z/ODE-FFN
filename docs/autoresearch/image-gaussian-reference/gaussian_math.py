"""Train-only Gaussian reference primitives; this is not an ODE-FFN candidate."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import torch

DIMENSION = 3 * 32 * 32
SCALAR_BUDGET = 185744
RANK = (SCALAR_BUDGET - DIMENSION - 1) // (DIMENSION + 1)
assert RANK == 59


def stored_scalars(parameters):
    return sum(parameters[k].size for k in ('mean', 'basis', 'eigenvalues')) + 1


def validate(parameters):
    mean, basis, eigen = (parameters[k] for k in ('mean', 'basis', 'eigenvalues'))
    tail = float(parameters['tail_variance'])
    assert mean.dtype == basis.dtype == eigen.dtype == np.float64
    assert mean.ndim == eigen.ndim == 1 and basis.shape == (len(mean), len(eigen))
    assert len(eigen) < len(mean) and np.isfinite(tail) and tail > 0
    assert all(np.isfinite(v).all() for v in (mean, basis, eigen))
    if len(eigen):
        np.testing.assert_allclose(basis.T @ basis, np.eye(len(eigen)), rtol=0, atol=1e-10)
        assert np.all(np.diff(eigen) <= 1e-12)
        assert np.all(eigen >= tail - 1e-12)
    return parameters


def from_covariance(mean, covariance, rank):
    mean = np.array(mean, dtype=np.float64, copy=True)
    covariance = np.asarray(covariance, dtype=np.float64)
    d = len(mean)
    assert covariance.shape == (d, d) and 0 <= rank < d
    np.testing.assert_allclose(covariance, covariance.T, rtol=0, atol=1e-12)
    if rank:
        eigen, basis = scipy.linalg.eigh(covariance, subset_by_index=(d-rank, d-1),
                                       driver='evr', check_finite=True)
        eigen, basis = eigen[::-1].copy(), basis[:, ::-1].copy()
    else:
        eigen, basis = np.empty(0, np.float64), np.empty((d, 0), np.float64)
    tail = float((np.trace(covariance) - eigen.sum()) / (d-rank))
    return validate(dict(mean=mean, basis=basis, eigenvalues=eigen, tail_variance=tail))


def normalize_pixels(pixels):
    """Use the native Torch float32 target normalization before promotion."""
    x = torch.from_numpy(np.ascontiguousarray(pixels)).float() / 127.5 - 1.
    return x.numpy().reshape(len(x), -1).astype(np.float64)


def stream_statistics(images, indices, batch_size=512, progress=None):
    """Two-pass MLE moments; no validation pixels or fitted neural weights."""
    n = len(indices)
    d = int(np.prod(images.shape[1:]))
    mean = np.zeros(d, np.float64)
    for start in range(0, n, batch_size):
        x = normalize_pixels(images[indices[start:start+batch_size]])
        mean += x.sum(axis=0)
        if progress:
            progress('mean', min(start+batch_size, n), n)
    mean /= n
    covariance = np.zeros((d, d), np.float64)
    for start in range(0, n, batch_size):
        x = normalize_pixels(images[indices[start:start+batch_size]]) - mean
        covariance += x.T @ x
        if progress:
            progress('covariance', min(start+batch_size, n), n)
    covariance /= n
    covariance = (covariance + covariance.T) * .5
    assert np.isfinite(covariance).all()
    return mean, covariance


def covariance_action(parameters, rows):
    u = parameters['basis']
    spread = parameters['eigenvalues'] - parameters['tail_variance']
    return rows * parameters['tail_variance'] + ((rows @ u) * spread) @ u.T


def independent_mean(parameters, images, times):
    """Woodbury solve for the actual stored covariance, without spectral gains."""
    mean, u, eigen = (parameters[k] for k in ('mean', 'basis', 'eigenvalues'))
    tail = parameters['tail_variance']
    shape = images.shape
    x = np.asarray(images, dtype=np.float64).reshape(len(images), -1)
    times = np.asarray(times, dtype=np.float64)
    assert times.shape == (len(x),)
    answer = np.empty_like(x)
    gram, spread = u.T @ u, eigen-tail
    for t in np.unique(times):
        selected = times == t
        residual = x[selected] - t*mean
        base = t*t*tail + (1-t)**2
        assert base > 0
        if len(eigen):
            system = np.eye(len(eigen)) + (t*t/base) * (spread[:, None]*gram)
            rhs = ((residual @ u)*spread*(t*t/base)).T
            correction = np.linalg.solve(system, rhs).T @ u.T
            inverse_action = (residual-correction)/base
        else:
            inverse_action = residual/base
        answer[selected] = mean + t*covariance_action(parameters, inverse_action)
    return answer.reshape(shape)


class GaussianPredictor(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        validate(parameters)
        for key in ('mean', 'basis', 'eigenvalues'):
            self.register_buffer(key, torch.tensor(parameters[key], dtype=torch.float64))
        self.register_buffer('tail_variance', torch.tensor(parameters['tail_variance'], dtype=torch.float64))

    def predict64(self, images, times):
        shape = images.shape
        x, t = images.reshape(len(images), -1).double(), times.double()[:, None]
        assert x.shape[1] == len(self.mean)
        residual = x-t*self.mean
        base = t*self.tail_variance/(t.square()*self.tail_variance+(1-t).square())
        answer = self.mean+base*residual
        if len(self.eigenvalues):
            gains = t*self.eigenvalues/(t.square()*self.eigenvalues+(1-t).square())
            answer = answer+((residual @ self.basis)*(gains-base)) @ self.basis.T
        return answer.reshape(shape)

    def forward(self, images, times, labels=None):
        return self.predict64(images, times).float()


def save_parameters(path, parameters):
    path = Path(path)
    assert not path.exists()
    validate(parameters)
    np.savez_compressed(path, **parameters)


def load_parameters(path):
    with np.load(path, allow_pickle=False) as data:
        p = {k: data[k].copy() for k in ('mean', 'basis', 'eigenvalues')}
        p['tail_variance'] = float(data['tail_variance'])
    return validate(p)


def fixtures():
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    records = []
    grid = np.array([0., .1, .3, .5, .7, .9, 1.])
    for d, rank in ((7, 0), (9, 2), (12, 4)):
        basis = np.linalg.qr(rng.standard_normal((d, rank)), mode='reduced')[0]
        p = validate(dict(mean=rng.normal(size=d), basis=basis,
                          eigenvalues=np.linspace(1.7, .8, rank), tail_variance=.3))
        sigma = covariance_action(p, np.eye(d))
        x = rng.normal(size=(len(grid), d))
        model = GaussianPredictor(p)
        got = model.predict64(torch.from_numpy(x), torch.from_numpy(grid)).detach().numpy()
        dense = np.array([p['mean']+t*sigma @ np.linalg.solve(t*t*sigma+(1-t)**2*np.eye(d), z-t*p['mean'])
                          for z, t in zip(x, grid)])
        other = independent_mean(p, x, grid)
        np.testing.assert_allclose(got, dense, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(other, dense, rtol=1e-12, atol=1e-12)
        state_error, time_error = 0., 0.
        for z, t in zip(x, grid):
            jt = torch.autograd.functional.jacobian(
                lambda value: model.predict64(value[None], torch.tensor([t], dtype=torch.float64))[0],
                torch.tensor(z, dtype=torch.float64)).numpy()
            v = t*t*sigma+(1-t)**2*np.eye(d)
            a = t*sigma @ np.linalg.inv(v)
            derivative = sigma @ np.linalg.inv(v)-t*sigma @ np.linalg.solve(v, (2*t*sigma-2*(1-t)*np.eye(d)) @ np.linalg.inv(v))
            dt = torch.autograd.functional.jacobian(
                lambda value: model.predict64(torch.tensor(z[None]), value[None])[0],
                torch.tensor(t, dtype=torch.float64)).numpy()
            expected_dt = derivative @ (z-t*p['mean'])-a @ p['mean']
            np.testing.assert_allclose(jt, a, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(dt, expected_dt, rtol=1e-12, atol=1e-12)
            assert np.linalg.eigvalsh((jt+jt.T)/2).min() >= -1e-12
            state_error = max(state_error, float(np.max(np.abs(jt-a))))
            time_error = max(time_error, float(np.max(np.abs(dt-expected_dt))))
        records.append(dict(dimension=d, rank=rank, times=len(grid), dense_error=float(np.max(np.abs(got-dense))),
                            independent_error=float(np.max(np.abs(other-dense))), state_J_error=state_error, time_error=time_error))
    x = rng.standard_normal((83, 11))
    mean = x.mean(0)
    covariance = (x-mean).T @ (x-mean)/len(x)
    p = from_covariance(mean, covariance, 3)
    eigen = np.linalg.eigvalsh(covariance)[::-1]
    np.testing.assert_allclose(p['eigenvalues'], eigen[:3], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(p['tail_variance'], eigen[3:].mean(), rtol=1e-12, atol=1e-12)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'prior.npz'
        save_parameters(path, p)
        restored = load_parameters(path)
        a, b = GaussianPredictor(p), GaussianPredictor(restored)
        for key in a.state_dict():
            assert torch.equal(a.state_dict()[key], b.state_dict()[key])
        probe = torch.from_numpy(rng.normal(size=(7, 11))).float()
        assert torch.equal(a(probe, torch.from_numpy(grid).float()), b(probe, torch.from_numpy(grid).float()))
    count = DIMENSION+(DIMENSION+1)*RANK+1
    assert count == 184380 and count <= SCALAR_BUDGET < count+DIMENSION+1
    gain_cases = []
    for variance in (.1, 1., 4.):
        t = 1/np.sqrt(1+variance)
        gain = t*variance/(t*t*variance+(1-t)**2)
        np.testing.assert_allclose(gain, (1+np.sqrt(1+variance))/2, rtol=1e-12, atol=1e-12)
        assert gain > 1
        gain_cases.append(dict(variance=variance, time=float(t), gain=float(gain)))
    return dict(status='PASS',seed=0,cases=records,MLE_eigen_and_tail=True,physical_parameter_resume_bitwise=True,
                dimension=DIMENSION,rank=RANK,stored_scalars=count,stored_float64_bytes=count*8,
                next_rank_exceeds_budget=True,noncontractive_exact_denoisers=gain_cases,real_training_statistics_fitted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    value = dict(fixtures(), seconds=time.monotonic()-started)
    if args.output:
        assert not args.output.exists()
        args.output.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps(value), flush=True)
