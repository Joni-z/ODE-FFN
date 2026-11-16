"""Native midpoint energy work and a piecewise continuous reference."""
import numpy as np
from scipy.integrate import solve_ivp
from check_hidden_energy import potential, response

LEVELS = (4, 8, 16, 32)
TOLERANCES = ((1e-10, 1e-11), (1e-12, 1e-13))


def midpoint(entrance: np.ndarray, weights: np.ndarray, steps: int) -> tuple:
    assert steps >= 4 and steps % 4 == 0 and len(weights) == 4
    state = entrance.copy()
    work = np.zeros(len(state))
    dt = 1/steps
    for i in range(steps):
        matrix = weights[4*i//steps]
        middle = state+.5*dt*(response(state)@matrix.T)
        q = response(middle)
        drift = q@matrix.T
        work += dt*np.einsum('bi,bi->b', q, drift)
        state = state+dt*drift
    return state, work


def reference(entrance: np.ndarray, weights: np.ndarray, rtol: float, atol: float) -> tuple:
    shape = entrance.shape
    n = shape[0]
    state = np.concatenate([entrance.ravel(), np.zeros(n)])
    evaluations = 0
    for stage, matrix in enumerate(weights):
        def drift(time: float, value: np.ndarray) -> np.ndarray:
            y = value[:-n].reshape(shape)
            q = response(y)
            update = q@matrix.T
            return np.concatenate([update.ravel(), np.einsum('bi,bi->b', q, update)])
        result = solve_ivp(drift, (stage/4, (stage+1)/4), state,
                           method='DOP853', rtol=rtol, atol=atol)
        assert result.success and np.isfinite(result.y).all(), result.message
        state = result.y[:, -1]
        evaluations += result.nfev
    return state[:-n].reshape(shape), state[-n:], evaluations


def observables(z: dict) -> dict:
    entrance, endpoints, work = z['entrance'], z['endpoints'], z['work']
    loose, reference_y = z['references']
    reference_work = z['reference_work'][-1]
    initial_energy, final_energy = potential(entrance), potential(reference_y)
    reference_energy = potential(z['references'])
    closure = final_energy-initial_energy-reference_work
    qualified = np.all(np.abs(loose-reference_y) <= 1e-9+1e-9*np.abs(reference_y), axis=1)
    qualified &= np.abs(z['reference_work'][0]-reference_work) <= 1e-8*(1+np.abs(reference_work))
    qualified &= np.abs(closure) <= 1e-8*(1+np.abs(initial_energy)+np.abs(final_energy)+np.abs(reference_work))
    difference = endpoints-reference_y[None]
    squared = np.sum(difference**2, axis=-1)
    q = response(reference_y)
    qnorm2 = np.sum(q*q, axis=1)
    linear_energy = np.einsum('knd,nd->kn', difference, q)
    normal2 = np.divide(linear_energy**2, qnorm2[None],
                        out=np.zeros_like(squared), where=qnorm2[None] != 0)
    assert np.all(normal2 <= squared+1e-11)
    tangent2 = np.maximum(squared-normal2, 0)
    energy = potential(endpoints)
    energy_gap = energy-final_energy[None]
    remainder = energy_gap-linear_energy
    rounding = 64*np.finfo(float).eps*(1+np.abs(energy)+np.abs(final_energy)[None])
    assert np.all(np.abs(remainder) <= .5*squared+rounding)
    feature_floor = np.linalg.norm(loose-reference_y, axis=1)+64*np.finfo(float).eps*(1+np.linalg.norm(reference_y, axis=1))
    energy_floor = np.abs(reference_energy[0]-reference_energy[1])+64*np.finfo(float).eps*(1+np.abs(reference_energy[0])+np.abs(reference_energy[1]))
    return dict(qualified=qualified, reference_closure=closure,
        reference_feature_floor=feature_floor, reference_energy_floor=energy_floor,
        initial_energy=initial_energy, energy=energy, energy_gap=energy_gap,
        symmetric_work=work, balance_defect=energy-initial_energy[None]-work,
        reference_work=reference_work, work_change=work-reference_work[None],
        feature_error_squared=squared, normal_error_squared=normal2, tangent_error_squared=tangent2,
        linear_energy_change=linear_energy, energy_linear_remainder=remainder,
        zero_reference_gradient=qnorm2 == 0)


def summarize(values: dict) -> dict:
    rows = []
    for i, step in enumerate(LEVELS):
        error = values['feature_error_squared'][i]
        normal = values['normal_error_squared'][i]
        gap = values['energy_gap'][i]
        rows.append(dict(steps=step, feature_error_rms=float(np.sqrt(np.mean(error))),
            feature_error_max_norm=float(np.sqrt(error.max())),
            normal_squared_error_share=float(normal.sum()/error.sum()) if error.sum() else None,
            energy_gap_rms=float(np.sqrt(np.mean(gap**2))),
            balance_defect_rms=float(np.sqrt(np.mean(values['balance_defect'][i]**2))),
            work_change_rms=float(np.sqrt(np.mean(values['work_change'][i]**2))),
            resolved_feature_points=int(np.sum(np.sqrt(error) > values['reference_feature_floor'])),
            energy_at_floor_points=int(np.sum(np.abs(gap) <= values['reference_energy_floor'])),
            directional_only_points=int(np.sum((np.sqrt(error) > values['reference_feature_floor'])
                & (np.abs(gap) <= values['reference_energy_floor'])))))
    return dict(points=len(values['qualified']), qualified_points=int(values['qualified'].sum()),
        zero_reference_gradient_points=int(values['zero_reference_gradient'].sum()),
        reference_closure_max=float(np.max(np.abs(values['reference_closure']))), levels=rows)
