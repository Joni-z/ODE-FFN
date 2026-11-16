"""Exact controls, oracle JVPs and real native-checkpoint sensitivity witnesses."""
import argparse
from pathlib import Path
import numpy as np
from geometry import read, dump, sha
from frozen_probe import restore, tensor_hashes, assets, torch, exact_jacobian, base
from trajectory import OracleResponse, BlendedField, SensitivityField, integrate, observe, LAMBDAS


class GaussianOracle(torch.nn.Module):
    def coefficient(self, t):
        return (2*t-1)/((1-t).square()+t.square())

    def forward(self, x, t):
        return self.coefficient(t)*x

    def jvp(self, x, t, direction):
        return self.coefficient(t)*direction


class GaussianModel(torch.nn.Module):
    def __init__(self, oracle, matrix, bias):
        super().__init__()
        self.oracle = oracle
        self.register_buffer('matrix', torch.tensor(matrix, dtype=torch.double))
        self.register_buffer('bias', torch.tensor(bias, dtype=torch.double))

    def forward(self, x, t):
        return self.oracle(x, t)+torch.nn.functional.linear(x, self.matrix)+self.bias


def exact_controls() -> dict:
    source = torch.linspace(-1., 1., 16, dtype=torch.double).reshape(8, 2)
    oracle = GaussianOracle()
    i0 = np.sqrt(2)*np.arcsinh(1.)
    rows = []
    settings = [('bias', np.zeros((2, 2)), np.array([.2, -.1])),
                ('coefficient', .3*np.eye(2), np.zeros(2)),
                ('rotation', np.array([[0., -.4], [.4, 0.]]), np.zeros(2))]
    for name, matrix, bias in settings:
        model = GaussianModel(oracle, matrix, bias)
        errors = []
        for steps in [128, 256]:
            result = observe(model, oracle, source, steps)
            if name == 'bias':
                expected = source.numpy()[None]+np.asarray(LAMBDAS)[:, None, None]*bias[None, None]*i0
                tangent = np.broadcast_to(bias*i0, source.shape)
            elif name == 'coefficient':
                expected = source.numpy()[None]*np.exp(.3*np.asarray(LAMBDAS))[:, None, None]
                tangent = .3*source.numpy()
            else:
                angle = .4*np.asarray(LAMBDAS)
                rotations = np.stack([np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) for a in angle])
                expected = np.einsum('nd,ked->kne', source.numpy(), rotations)
                tangent = source.numpy()@matrix.T
                assert np.max(np.abs(rotations@rotations.transpose(0, 2, 1)-np.eye(2))) < 1e-14
            endpoint_error = float(np.max(np.abs(result['endpoints']-expected)))
            tangent_error = float(np.max(np.abs(result['tangent']-tangent)))
            assert max(endpoint_error, tangent_error) < 1e-4
            errors.append(dict(steps=steps, endpoint_max_error=endpoint_error, tangent_max_error=tangent_error))
            raw = base.train.generated_samples(oracle, source, steps).numpy()
            assert np.array_equal(raw, result['oracle_endpoint'])
        ratio = errors[0]['endpoint_max_error']/errors[1]['endpoint_max_error']
        # Order two is asymptotic, not an exact finite-mesh fourfold identity.
        assert ratio > 3., (name, errors, ratio)
        rows.append(dict(control=name, errors=errors, endpoint_error_ratio=ratio,
                         oracle_recording_exact=True))
    return dict(passed=True, controls=rows, bias_integral=i0,
        rotation_preserves_exact_isotropic_law=True,
        interpretation='Endpoint displacement is coupling dependent, not a law discrepancy')


def oracle_checks(law, arrays: dict) -> dict:
    oracle = OracleResponse(law)
    rows = []
    direction = torch.sin(torch.arange(1, law.dim+1, dtype=torch.double))
    direction /= direction.norm()
    for time in [0., .1, .3, .5, .7, .9, 1.]:
        point = (1-time)*arrays['source_0'][:7].astype(float)+time*arrays['target_0'][:7].astype(float)
        point = np.concatenate([point, np.zeros((1, law.dim))])
        x = torch.from_numpy(point)
        t = x.new_full((len(x), 1), time)
        v, jac = exact_jacobian(oracle, x, t)
        jvp = oracle.jvp(x, t, direction.expand_as(x))
        expected = jac@direction
        ad_error = float((jvp-expected).abs().max())
        native_error = float(np.max(np.abs(v.numpy()-law.velocity(point, time))))
        eps = 1e-6
        finite = (oracle(x+eps*direction, t)-oracle(x-eps*direction, t))/(2*eps)
        finite_error = float((finite-jvp).abs().max())
        assert ad_error < 2e-10 and native_error < 2e-10
        assert torch.allclose(finite, jvp, atol=1e-5, rtol=2e-5)
        rows.append(dict(time=time, analytic_jvp_vs_autograd=ad_error,
                         native_velocity_max_error=native_error, finite_difference_jvp_max_error=finite_error))
    return dict(passed=True, rows=rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ['study', 'data-root', 'output']:
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    study, out, data_root = map(Path, [args.study, args.output, args.data_root])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    dump(out/'exact-controls.json', exact_controls())
    anchors = read(study/'anchors.json')
    rows, checked_laws = [], {}
    for row in anchors['references']+anchors['new_cases']:
        model, law, cell, evaluation, identity = restore(study, anchors, row, data_root)
        arrays, _, binding = assets(anchors['assets'], cell['dataset'], 'selection', 'cpu')
        if cell['dataset'] not in checked_laws:
            checked_laws[cell['dataset']] = oracle_checks(law, arrays)
        source = torch.from_numpy(np.concatenate([arrays[f'source_{b}'][:1] for b in range(8)]))
        saved = np.concatenate([np.load(evaluation/f'20000-b{b}.npy')[:1] for b in range(8)])
        fp32 = base.train.generated_samples(model, source, 128)
        native_error = float(np.max(np.abs(fp32.numpy()-saved)))
        assert np.allclose(fp32.numpy(), saved, atol=1e-4, rtol=1e-4)
        model.double()
        source = source.double()
        snapshot = tensor_hashes(model)
        oracle = OracleResponse(law)
        endpoint, _ = integrate(BlendedField(model, oracle, 1.), source, 128)
        direct = base.train.generated_samples(model, source, 128)
        assert torch.equal(endpoint, direct)
        precision_error = float((endpoint-fp32.double()).abs().max())
        assert torch.allclose(endpoint, fp32.double(), atol=1e-3, rtol=1e-3)
        initial = torch.cat((source, torch.zeros_like(source)), dim=1)
        augmented, _ = integrate(SensitivityField(model, oracle), initial, 32)
        true = base.train.generated_samples(oracle, source, 32)
        assert torch.equal(augmented[:, :law.dim], true)
        eps = 1e-5
        plus = base.train.generated_samples(BlendedField(model, oracle, eps), source, 32)
        minus = base.train.generated_samples(BlendedField(model, oracle, -eps), source, 32)
        fd = (plus-minus)/(2*eps)
        tangent = augmented[:, law.dim:]
        error = float((fd-tangent).abs().max())
        assert torch.allclose(fd, tangent, atol=2e-5, rtol=1e-4), (row['id'], error)
        assert snapshot == tensor_hashes(model)
        rows.append(dict(id=row['id'], native_fp32_sample_error=native_error,
            fp64_vs_fp32_sample_error=precision_error, lambda1_native_endpoint_exact=True,
            augmented_oracle_endpoint_exact=True, native_discrete_tangent_fd_error=error,
            unchanged_parameters_and_buffers=True, assets_sha256=binding['sha256'], **identity))
        print(f'{len(rows)}/12 native sensitivity checks', flush=True)
    dump(out/'PASS.json', dict(passed=True, cases=12, laws=checked_laws, rows=rows,
        exact_controls_sha256=sha(out/'exact-controls.json'),
        anchors_sha256=sha(study/'anchors.json'),
        source_sha256={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('trajectory.py')]}))


if __name__ == '__main__':
    main()
