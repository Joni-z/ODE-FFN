"""Native-order Heun steps and exact signed discrete path accounting."""
from __future__ import annotations

import numpy as np
import torch

STEPS = 32
SCOPES = ('full', 'tangent', 'normal')


def advance(field, state: torch.Tensor, index: int, steps: int = STEPS):
    dt = 1.0/steps
    t = torch.full((len(state), 1), index*dt, device=state.device, dtype=state.dtype)
    first = field(state, t);predictor = state+dt*first
    second = field(predictor, torch.full_like(t, (index+1)*dt))
    following = state+0.5*dt*(first+second)
    return following, first, predictor, second


def suffix(field, state: torch.Tensor, start: int, steps: int = STEPS):
    state = state.clone()
    for index in range(start, steps):state = advance(field, state, index, steps)[0]
    return state


def trace(field, source: torch.Tensor, steps: int = STEPS):
    states = [source.clone()]
    for index in range(steps):states.append(advance(field, states[-1], index, steps)[0])
    return torch.stack(states)


def normal_suffix_gains(variance: float, steps: int = STEPS) -> np.ndarray:
    t = np.arange(steps+1, dtype=np.float64)/steps;a = 1-t
    denominator = a*a+t*t*variance
    gain = np.divide(t*variance, denominator, out=np.zeros_like(t), where=denominator>0)
    beta = (gain-1)/np.maximum(a, .05);h = 1/steps
    factors = 1+h/2*(beta[:-1]+beta[1:]+h*beta[:-1]*beta[1:])
    result = np.ones(steps+1)
    for index in reversed(range(steps)):result[index] = factors[index]*result[index+1]
    return result


def project(value: np.ndarray, basis: np.ndarray):
    value = np.asarray(value, dtype=np.float64);plane = value@basis
    return (value, plane, value-plane@basis.T)


def energy(value: np.ndarray, basis: np.ndarray) -> dict:
    parts = project(value, basis);dimensions = (basis.shape[0], basis.shape[1], basis.shape[0]-basis.shape[1])
    return {scope:float(np.sum(v*v)/(len(v)*dimension)) for scope, v, dimension in zip(SCOPES, parts, dimensions)}


def account(bridges: np.ndarray, local: np.ndarray, basis: np.ndarray, normal_gains: np.ndarray):
    bridges = bridges.astype(np.float64);delta = np.diff(bridges, axis=0);error = bridges[-1]-bridges[0]
    telescoping = float(np.max(np.abs(delta.sum(0)-error)))
    assert telescoping<=1e-10
    dimensions = (basis.shape[0], basis.shape[1], basis.shape[0]-basis.shape[1])
    rows = [];summary = {};per_source = []
    projected = list(zip(SCOPES, dimensions, project(error, basis), project(delta, basis), project(local, basis)))
    for scope, dimension, e, d, l in projected:
        signed = np.einsum('nd,snd->sn', e, d)/dimension
        mse = float(np.sum(e*e)/(len(e)*dimension))
        np.testing.assert_allclose(signed.sum(0), np.sum(e*e, axis=1)/dimension, atol=1e-10, rtol=1e-10)
        per_source.append(signed)
        summary[scope+'_paired_endpoint_mse'] = mse
        summary[scope+'_late_signed_share'] = float(signed[24:].sum(0).mean()/mse) if mse>1e-20 else None
        share = summary[scope+'_late_signed_share']
        summary[scope+'_late_aligned'] = share>=.5 if share is not None else None
        for index in range(len(delta)):
            local_energy = float(np.sum(l[index]**2)/(len(e)*dimension))
            transported = float(np.sum(d[index]**2)/(len(e)*dimension))
            rows.append(dict(scope=scope, step=index, time=index/STEPS,
                signed_endpoint_mse_contribution=float(signed[index].mean()),
                local_step_energy=local_energy, transported_step_energy=transported,
                finite_rms_gain=float(np.sqrt(transported/local_energy)) if local_energy>1e-20 else None))
    normal_delta = projected[2][3];normal_local = projected[2][4]
    predicted = normal_gains[1:, None, None]*normal_local
    normal_error = float(np.max(np.abs(normal_delta-predicted)))
    normal_total = float(np.max(np.abs(project(error, basis)[2]-predicted.sum(0))))
    assert normal_error<=1e-4 and normal_total<=1e-4
    summary.update(telescoping_max=telescoping, normal_propagation_max=normal_error,
                   normal_total_propagation_max=normal_total)
    return summary, rows, np.stack(per_source, axis=-1)


def forced_readout(bridges: np.ndarray, source: np.ndarray, state: dict,
                   basis: np.ndarray, gains: np.ndarray):
    weight = state['model']['raw.output.weight'].double().numpy()
    gamma = state['model']['raw.norm.weight'].double().numpy()
    beta = state['model']['raw.norm.bias'].double().numpy()
    bias = state['model']['raw.output.bias'].double().numpy()
    center = np.eye(len(gamma))-np.ones((len(gamma), len(gamma)))/len(gamma)
    columns = (weight*gamma[None])@center;c0 = weight@beta+bias
    _, singular, right = np.linalg.svd(np.vstack((columns.T, basis.T)), full_matrices=True)
    rank = int(np.sum(singular>1e-10*max(float(singular[0]), 1.)))
    null = right[rank:].T;projector = null@null.T
    np.testing.assert_allclose(projector@projector, projector, atol=1e-8, rtol=0)
    assert np.max(np.abs(columns.T@projector))<=1e-8
    assert np.max(np.abs(basis.T@projector))<=1e-8
    skip_gains = normal_suffix_gains(0.)
    factors = skip_gains[:-1]/skip_gains[1:]
    prefix = np.concatenate(([1.], np.cumprod(factors)))
    observed = bridges.astype(np.float64)@projector
    predicted = gains[:, None, None]*(prefix[:, None, None]*(source.astype(np.float64)@projector)[None]
        +(1-prefix[:, None, None])*(c0@projector)[None, None])
    discrepancy = float(np.max(np.abs(observed-predicted)));assert discrepancy<=1e-4
    error = observed[-1]-observed[0];delta = np.diff(observed, axis=0)
    normal_dimension = basis.shape[0]-basis.shape[1]
    signed = np.einsum('nd,snd->sn', error, delta)/normal_dimension
    energy = float(np.mean(np.sum(error**2, axis=1))/normal_dimension)
    normal_error = project(bridges[-1].astype(np.float64)-bridges[0], basis)[2]
    total_energy = float(np.mean(np.sum(normal_error**2, axis=1))/normal_dimension)
    fraction = energy/total_energy if total_energy>1e-20 else None
    if fraction is not None:assert fraction<=1+1e-8
    summary = dict(forced_normal_dimension=basis.shape[0]-rank,
        forced_bridge_prediction_max=discrepancy, forced_normal_endpoint_mse=energy,
        forced_fraction_of_normal_paired_error=fraction)
    return summary, signed, projector
