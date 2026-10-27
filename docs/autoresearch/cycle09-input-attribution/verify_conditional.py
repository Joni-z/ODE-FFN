"""Dense joint-Gaussian checks independent of the efficient conditional formula."""
from __future__ import annotations

import numpy as np


def dense(x, t, rows, basis, variance):
    from conditional_oracle import ring_means
    dim = len(basis)
    sigma = variance*np.eye(dim)+(.04-variance)*(basis@basis.T)
    means = ring_means()@basis.T
    result_y, result_v = [], []
    for value, clock in zip(x, t[:, 0]):
        a, delta = 1-clock, max(1-clock, .05)
        covariance = rows@(a*a*np.eye(dim)+clock*clock*sigma)@rows.T
        residual = rows@value-clock*(means@rows.T)
        solved = np.linalg.solve(covariance, residual.T).T
        logits = -.5*np.sum(residual*solved, axis=1)
        p = np.exp(logits-logits.max());p /= p.sum()
        conditional_y = means+clock*(solved@rows@sigma)
        conditional_v = a/delta*means+(a/delta)*(solved@rows@(clock*sigma-a*np.eye(dim)))
        result_y.append(p@conditional_y);result_v.append(p@conditional_v)
    return np.array(result_y), np.array(result_v)


def verify(core):
    from conditional_oracle import input_rows, conditional_means, best_field, components
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.standard_normal((10, 2)))[0]
    rows = np.linalg.qr(rng.standard_normal((10, 6)))[0].T
    t = np.repeat([0., .1, .3, .7, .97], 4)[:, None]
    x = rng.standard_normal((len(t), 10))
    maximum = 0.
    for variance in (0., .04):
        clean, noise = conditional_means(x, t, rows, basis, variance)
        dense_clean, dense_v = dense(x, t, rows, basis, variance)
        decoded = (1-t)/np.maximum(1-t, .05)*(clean-noise)
        maximum = max(maximum, float(np.max(np.abs(clean-dense_clean))),
                      float(np.max(np.abs(decoded-dense_v))))
        np.testing.assert_allclose(clean, dense_clean, atol=1e-8, rtol=0)
        np.testing.assert_allclose(decoded, dense_v, atol=1e-8, rtol=0)
    full_basis = core.make_basis()
    full_x = rng.standard_normal((20, 128))
    full_maximum = 0.
    for variance in (0., .04):
        truth = core.oracle(full_x, t, full_basis, variance)['oracle']
        for head in ('direct', 'clean', 'shifted'):
            value = best_field(full_x, t, np.eye(128), full_basis, variance, head)
            full_maximum = max(full_maximum, float(np.max(np.abs(value-truth))))
            np.testing.assert_allclose(value, truth, atol=1e-8, rtol=0)
    deficient = np.concatenate((rows[:3], rows[:3]))
    extracted, _ = input_rows(deficient)
    assert len(extracted) == 3
    np.testing.assert_allclose(extracted.T@extracted, rows[:3].T@rows[:3], atol=1e-12)
    empty, _ = conditional_means(x, t, np.empty((0, 10)), basis, 0.)
    assert np.count_nonzero(empty) == 0
    components(rng.standard_normal(x.shape), rng.standard_normal(x.shape), x, basis)
    return dict(status='PASS', dense_conditioning_max=maximum,
                full_observation_max=full_maximum, rank_deficient_control=True,
                finite_cross_term_retained=True)
