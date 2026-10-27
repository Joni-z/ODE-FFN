"""Complete conditional field derivatives with mixture-responsibility terms."""
from __future__ import annotations

import numpy as np


def conditional(x: np.ndarray, clock: float, rows: np.ndarray, basis: np.ndarray,
                variance: float, head: str) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if not 0 <= clock < 1:
        raise ValueError('time must be below one')
    dim = x.shape[1];a = 1-clock;delta = max(a, .05)
    identity = np.eye(dim);projector = rows.T@rows
    np.testing.assert_allclose(rows@rows.T, np.eye(len(rows)), atol=1e-10, rtol=0)
    sigma = variance*identity+(.04-variance)*(basis@basis.T)
    covariance = a*a*np.eye(len(rows))+clock*clock*rows@sigma@rows.T
    inverse = np.linalg.inv(covariance)
    c = rows@basis;k = clock*sigma@rows.T@inverse;m = basis-clock*k@c
    angles = np.arange(8)*2*np.pi/8
    means = 3*np.column_stack((np.cos(angles), np.sin(angles)))
    u = x@rows.T
    residual = u[:, None, :]-clock*(means@c.T)[None]
    logits = -.5*np.einsum('bki,ij,bkj->bk', residual, inverse, residual)
    probabilities = np.exp(logits-logits.max(1, keepdims=True))
    probabilities /= probabilities.sum(1, keepdims=True)
    mean = probabilities@means
    centered = means[None]-mean[:, None]
    mixture_covariance = np.einsum('bk,bki,bkj->bij', probabilities, centered, centered)
    clean = u@k.T+mean@m.T
    clean_jacobian = k@rows+np.einsum('di,bij,je->bde', m, mixture_covariance, clock*c.T@inverse@rows)
    posterior_covariance = sigma-clock*clock*sigma@rows.T@inverse@rows@sigma
    posterior_covariance = posterior_covariance+np.einsum('di,bij,ej->bde', m, mixture_covariance, m)
    covariance_jacobian = clock/(a*a)*(posterior_covariance@projector)
    covariance_error = float(np.max(np.abs(clean_jacobian-covariance_jacobian)))
    assert covariance_error <= 1e-8
    if head == 'direct':
        factor = a*identity+clock*projector
        field = (clean@factor.T-x@projector)/delta
        jacobian = (factor@clean_jacobian-projector)/delta
    elif head in ('clean', 'shifted'):
        field = (clean-x)/delta
        jacobian = (clean_jacobian-identity)/delta
    else:
        raise ValueError(head)
    return dict(field=field, jacobian=jacobian, clean_jacobian=clean_jacobian,
                posterior_covariance=posterior_covariance, covariance_identity_max=covariance_error)


def diagnostics(model: np.ndarray, reference: np.ndarray, truth: np.ndarray,
                basis: np.ndarray) -> dict[str, np.ndarray]:
    result = {}
    for scope in ('full', 'tangent'):
        matrices = [np.asarray(v, dtype=np.float64) for v in (model, reference, truth)]
        if scope == 'tangent':
            matrices = [np.einsum('di,bde,ej->bij', basis, v, basis) for v in matrices]
        fitted, optimum, full = matrices
        info, error, total = optimum-full, fitted-optimum, fitted-full
        norm = lambda value: np.sum(value*value, axis=(1, 2))
        sym = lambda value: (value+value.transpose(0, 2, 1))/2
        skew = lambda value: (value-value.transpose(0, 2, 1))/2
        rows = dict(total=norm(total), information_displacement=norm(info), residual=norm(error),
            cross=2*np.sum(info*error, axis=(1, 2)), residual_symmetric=norm(sym(error)),
            residual_skew=norm(skew(error)), model_skew=norm(skew(fitted)),
            reference_skew=norm(skew(optimum)), full_oracle_skew=norm(skew(full)))
        np.testing.assert_allclose(rows['total'], rows['information_displacement']+rows['residual']+rows['cross'], atol=1e-8, rtol=1e-10)
        np.testing.assert_allclose(rows['residual'], rows['residual_symmetric']+rows['residual_skew'], atol=1e-8, rtol=1e-10)
        if scope == 'tangent':
            maximum = np.linalg.eigvalsh(sym(full))[:, -1]
            rows.update(full_oracle_max_symmetric_eigenvalue=maximum,
                        full_oracle_expanding=(maximum>1e-8).astype(float))
        result.update({scope+'_'+key: value for key, value in rows.items()})
    return result
