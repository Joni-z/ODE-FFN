"""Deterministic controls for the sign and scale of midpoint modified bias."""
import argparse
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from geometry import dump, sha


def q(y: np.ndarray) -> np.ndarray:
    return y/np.sqrt(1+1.2*y*y)


def components(y: np.ndarray, p: np.ndarray) -> tuple:
    d = np.power(1+1.2*y*y, -.5)
    first, second = d**3, -3.6*y*d**5
    f = q(y)@p.T
    curvature = -(second*f*f)@p.T/24
    chain = -(first*((first*f)@p.T))@p.T/6
    return f, first, curvature, chain


def midpoint(y0: np.ndarray, p: np.ndarray, steps: int, nonlinear: bool = True) -> np.ndarray:
    y, dt = y0.copy(), 1/steps
    response = q if nonlinear else lambda value:value
    for _ in range(steps):
        y += dt*(response(y+.5*dt*(response(y)@p.T))@p.T)
    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    y0 = .6*np.sin(np.arange(1, 21).reshape(5, 4))
    p = np.array([[-.3, .2, .1, 0.], [-.1, .1, .2, -.1],
                  [.1, -.2, -.25, .15], [.05, .1, -.1, .2]])
    d = np.power(1+1.2*y0*y0, -.5)
    eps = 1e-5
    first_fd = (q(y0+eps)-q(y0-eps))/(2*eps)
    second_fd = (np.power(1+1.2*(y0+eps)**2, -1.5)-np.power(1+1.2*(y0-eps)**2, -1.5))/(2*eps)
    first_error = float(np.max(np.abs(first_fd-d**3)))
    second_error = float(np.max(np.abs(second_fd+3.6*y0*d**5)))
    assert first_error < 1e-8 and second_error < 1e-8
    local = []
    for step in [1/8, 1/16, 1/32, 1/64]:
        f, _, curvature, chain = components(y0, p)
        coarse = y0+step*(q(y0+.5*step*f)@p.T)
        result = solve_ivp(lambda t, value:(q(value.reshape(y0.shape))@p.T).ravel(),
                           (0, step), y0.ravel(), method='DOP853', rtol=1e-13, atol=1e-14)
        assert result.success
        exact = result.y[:, -1].reshape(y0.shape)
        predicted = step**3*(curvature+chain)
        local.append(dict(step=step, relative_bias_residual=float(np.linalg.norm(coarse-exact-predicted)/np.linalg.norm(predicted))))
    assert all(a['relative_bias_residual'] > b['relative_bias_residual'] for a, b in zip(local, local[1:]))
    assert all(1.6 < a['relative_bias_residual']/b['relative_bias_residual'] < 2.4 for a, b in zip(local, local[1:]))
    # An exact nonlinear example independently checks the coefficient signs.
    scalar = []
    value = .4
    for step in [1/8, 1/16, 1/32, 1/64]:
        exact = value/(1-step*value)
        coarse = value+step*(value+.5*step*value**2)**2
        predicted = -.75*step**3*value**4
        scalar.append(dict(step=step, relative_bias_residual=abs((coarse-exact-predicted)/predicted)))
    assert all(a['relative_bias_residual'] > b['relative_bias_residual'] for a, b in zip(scalar, scalar[1:]))
    # For linear f=Py, g=-P^3 y/6 and z(1)=-exp(P)P^3 y0/6.
    exact_linear = y0@expm(p).T
    z_linear = -(y0@np.linalg.matrix_power(p, 3).T)@expm(p).T/6
    linear = []
    for steps in [4, 8, 16, 32, 64]:
        delta = midpoint(y0, p, steps, nonlinear=False)-exact_linear
        predicted = z_linear/steps**2
        linear.append(dict(steps=steps, relative_global_residual=float(np.linalg.norm(delta-predicted)/np.linalg.norm(delta))))
    assert all(a['relative_global_residual'] > b['relative_global_residual'] for a, b in zip(linear, linear[1:]))
    shape = y0.shape
    initial = np.stack([y0, np.zeros_like(y0), np.zeros_like(y0)]).ravel()
    def augmented(time: float, value: np.ndarray) -> np.ndarray:
        y, z_h, z_j = value.reshape(3, *shape)
        f, first, curvature, chain = components(y, p)
        return np.stack([f, (first*z_h)@p.T+curvature, (first*z_j)@p.T+chain]).ravel()
    reference = solve_ivp(augmented, (0, 1), initial, method='DOP853', rtol=1e-12, atol=1e-13)
    assert reference.success
    exact, z_h, z_j = reference.y[:, -1].reshape(3, *shape)
    nonlinear = []
    for steps in [4, 8, 16, 32, 64]:
        delta = midpoint(y0, p, steps)-exact
        prediction = (z_h+z_j)/steps**2
        nonlinear.append(dict(steps=steps, relative_global_residual=float(np.linalg.norm(delta-prediction)/np.linalg.norm(delta))))
    assert all(a['relative_global_residual'] > b['relative_global_residual'] for a, b in zip(nonlinear, nonlinear[1:]))
    # Pure linear rotation has a tangent leading bias, not leading normal work.
    k = np.array([[0., -.7], [.7, 0.]])
    start = y0[:, :2]
    end = start@expm(k).T
    z_skew = -(start@np.linalg.matrix_power(k, 3).T)@expm(k).T/6
    normal_work = float(np.max(np.abs(np.sum(end*z_skew, axis=1))))
    assert normal_work < 1e-14
    result = dict(passed=True, response_derivative_max_error=first_error,
        response_second_derivative_max_error=second_error, isru_local=local,
        exact_quadratic_local=scalar, exact_linear_global=linear,
        isru_variational_global=nonlinear, skew_leading_normal_work=normal_work,
        source_sha256=sha(Path(__file__)), scope='deterministic mathematical controls; no learned checkpoint observed')
    dump(Path(args.output), result)
    print('PASS midpoint bias controls: derivative errors', first_error, second_error,
          'finest ISRU global residual', nonlinear[-1]['relative_global_residual'], flush=True)


if __name__ == '__main__':
    main()
