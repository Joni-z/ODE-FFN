"""Check target identities against unchanged Torch/native NumPy mixture oracles."""
import argparse
from pathlib import Path
import sys
import numpy as np
from geometry import TIMES, read, sha, dump, posterior, target

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'coefficient_20261116'))
from gpu_resolution import base, torch
sys.path.insert(0, str(HERE.parent/'density_transport_20261116'))
from density_ops import MixtureOracle, exact_jacobian
from evaluate_screen import assets


class Velocity(torch.nn.Module):
    def __init__(self, law):
        super().__init__()
        self.oracle = MixtureOracle(law)

    def forward(self, x, t):
        return self.oracle.evaluate(x, t)['velocity']


def counterexamples() -> dict:
    # Exact autonomous linear flow, followed by a decoder: hidden SPD is insufficient.
    full_negative = -np.exp(.5)
    h = torch.tensor(-1., dtype=torch.double, requires_grad=True)
    clock = .2+.8*torch.sigmoid(20*(h+1))
    response = h*torch.exp(.5*clock)
    derivative = float(torch.autograd.grad(response, h)[0])
    assert derivative < 0 and full_negative < 0
    # A Gaussian Bayes denoiser can expand: PSD does not mean nonexpansive.
    var, sigma0, t = np.array([1., 4.]), 1.7, .7
    jm = t*var/(((1-t)*sigma0)**2+t*t*var)
    assert jm.max() > 1
    # Isotropic Gaussian FM: adding b leaves J unchanged and shifts terminal mean.
    from scipy.integrate import solve_ivp
    a = lambda u: (2*u-1)/((1-u)**2+u*u)
    mu = solve_ivp(lambda u, m: a(u)*m+.2, (0, 1), [0.], rtol=1e-11, atol=1e-13).y[0, -1]
    assert mu > .1
    # Adding omega Kz preserves the same Gaussian covariance path, while adding curl.
    k = np.array([[0., -1.], [1., 0.]])
    s = (1-t)**2+t*t
    j = a(t)*np.eye(2)+.4*k
    covariance_derivative_error = float(np.max(np.abs(j*(s)+(j*s).T-(4*t-2)*np.eye(2))))
    assert covariance_derivative_error < 1e-14
    return dict(hidden_spd_plus_decoder_full_derivative=full_negative,
        bounded_positive_P_input_clock_full_derivative=derivative,
        gaussian_nonunit_noise=sigma0, expanding_bayes_jacobian=jm.tolist(),
        same_jacobian_bias_terminal_mean=float(mu),
        rotating_field_same_marginal_covariance_derivative_error=covariance_derivative_error,
        interpretation='Bayes Jacobian geometry is neither a hidden-P property nor a sampling certificate')


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ['study', 'data-root', 'output']:
        p.add_argument('--'+key, required=True)
    args = p.parse_args()
    study, out = Path(args.study), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    anchor = read(study/'anchors.json')
    rows = []
    for dataset in ['gmm_ring8_2d', 'rotated_product_bimodal_16d']:
        ref = next(r for r in anchor['references'] if r['dataset'] == dataset)
        cfg = read(Path(ref['folder'])/'config.json')
        cache = base.verify_data(args.data_root, cfg['cell'])
        manifest = read(cache/'manifest.json')
        law = base.train.load_distribution(manifest['datasets'][dataset])
        array, _, binding = assets(anchor['assets'], dataset, 'selection', 'cpu')
        oracle = Velocity(law)
        for t in TIMES:
            x = (1-t)*array['source_0'][:24].astype(float)+t*array['target_0'][:24].astype(float)
            x = np.concatenate([x, np.zeros((1, law.dim))])
            xt = torch.tensor(x, dtype=torch.double)
            tt = xt.new_full((len(x), 1), t)
            v, jv = exact_jacobian(oracle, xt, tt)
            analytic = target(law, x, t)
            native = law.velocity(x, t)
            vm = np.max(np.abs(analytic['velocity']-native))
            jm = np.max(np.abs(analytic['velocity_jacobian']-jv.numpy()))
            identity_error = np.max(np.abs(x+(1-t)*v.numpy()-analytic['mean']))
            direction = np.sin(np.arange(law.dim)+1.)
            direction /= np.linalg.norm(direction)
            eps = 1e-5
            plus = posterior(law, x+eps*direction, t)[0]
            minus = posterior(law, x-eps*direction, t)[0]
            finite = (plus-minus)/(2*eps)
            fd = np.max(np.abs(finite-np.einsum('nij,j->ni', analytic['image_jacobian'], direction)))
            psd_min = float(np.linalg.eigvalsh(analytic['image_jacobian']).min())
            assert max(vm, jm, identity_error) < 2e-10 and fd < 2e-6 and psd_min > -1e-10
            row = dict(dataset=dataset, time=t, n=len(x), posterior_vs_native_velocity_max=float(vm),
                covariance_vs_torch_jacobian_max=float(jm), conditional_mean_identity_max=float(identity_error),
                finite_difference_image_jvp_max=float(fd), image_jacobian_min_eigenvalue=psd_min,
                assets_sha256=binding['sha256'])
            rows.append(row)
    # Independent direct Gaussian algebra with nonunit source and anisotropic prior.
    sigma0, t = 1.7, .7
    var = np.array([1., 4.])
    c = ((1-t)*sigma0)**2+t*t*var
    direct_jv = (t*var-(1-t)*sigma0**2)/c
    converted = (t*var/c-1)/(1-t)
    assert np.max(np.abs(direct_jv-converted)) < 1e-14
    from types import SimpleNamespace
    gaussian = SimpleNamespace(dim=2, means=np.zeros((1, 2)), variances=var[None], weights=np.ones(1))
    point = np.array([[.2, -.3], [1., 2.]])
    calculated = target(gaussian, point, t, sigma0)
    nonunit_posterior_error = float(np.max(np.abs(calculated['mean']-t*var*point/c)))
    nonunit_jacobian_error = float(np.max(np.abs(calculated['velocity_jacobian']-np.diag(direct_jv))))
    assert max(nonunit_posterior_error, nonunit_jacobian_error) < 1e-14
    dump(out/'PASS.json', dict(passed=True, rows=rows, counterexamples=counterexamples(),
        nonunit_noise_velocity_jacobian_identity_max=float(np.max(np.abs(direct_jv-converted))),
        nonunit_posterior_max_error=nonunit_posterior_error,
        nonunit_full_jacobian_max_error=nonunit_jacobian_error,
        source_sha256={f.name: sha(f) for f in [Path(__file__), HERE/'geometry.py']},
        anchors_sha256=sha(study/'anchors.json')))
    print('PASS target identities:', len(rows), 'native task/time cells', flush=True)


if __name__ == '__main__':
    main()
