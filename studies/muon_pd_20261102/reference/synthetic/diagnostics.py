"""Endpoint and time-series diagnostics for internal ODE matrices and actual activations."""
import numpy as np
import torch


def spectrum(weight):
    p=weight.detach().float().cpu().numpy().astype(np.float64)
    singular=np.linalg.svd(p,compute_uv=False)
    eigen=np.linalg.eigvals(p)
    total=float(np.sum(singular**2))
    return {'sigma_max':float(singular[0]),'sigma_median':float(np.median(singular)),
            'sigma_min':float(singular[-1]),
            'sigma_quantiles':[float(a) for a in np.quantile(singular,[0,.1,.25,.5,.75,.9,1])],
            'stable_rank':float(total/(singular[0]**2)),
            'eig_real_max':float(eigen.real.max()),
            'eig_real_min':float(eigen.real.min()),
            'spectral_radius':float(np.max(np.abs(eigen))),
            'nonnormality':float(np.linalg.norm(p.T@p-p@p.T)/(total+1e-12))}


@torch.no_grad()
def collect(model,x,t):
    """Probe actual hidden inputs to both solves; x,t remain fixed across checkpoints."""
    nonlinear=list(model.nonlinear_layers()) if hasattr(model,'nonlinear_layers') else []
    covariance=list(model.covariance_layers()) if hasattr(model,'covariance_layers') else []
    diagonal=list(model.diagonal_layers()) if hasattr(model,'diagonal_layers') else []
    forced=list(model.forced_gate_layers()) if hasattr(model,'forced_gate_layers') else []
    state_clock=list(model.state_clock_layers()) if hasattr(model,'state_clock_layers') else []
    lowrank=list(model.lowrank_layers()) if hasattr(model,'lowrank_layers') else []
    if not list(model.ode_layers()) and not nonlinear and not covariance and not diagonal and not forced and not state_clock and not lowrank:
        return []
    inputs={}
    hooks=[]
    for name,layer in model.ode_layers():
        hooks.append(layer.register_forward_pre_hook(lambda module,args,key=name:inputs.__setitem__(key,
            (args[0].detach(), args[1].detach() if len(args)>1 and args[1] is not None else None))))
    for name,layer in nonlinear+covariance+diagonal+forced+state_clock+lowrank:
        hooks.append(layer.register_forward_pre_hook(lambda module,args,key=name:
            inputs.__setitem__(key,args[0].detach())))
    try:
        model(x,t)
    finally:
        for hook in hooks:
            hook.remove()
    records=[]
    for name,layer in model.ode_layers():
        raw=spectrum(layer.proj.weight)
        effective=spectrum(layer.operator())
        records.append({'layer':name,'raw':raw,'effective':effective,
                        'gate_source':layer.gate_source,
                        'rate_placement':layer.rate_placement,
                        'solve_return':layer.solve_return,
                        'operator_mode':layer.operator_mode,
                        'activation':layer.activation_stats(*inputs[name])})
    for name,layer in nonlinear:
        stats=layer.activation_stats(inputs[name])
        for role,projection in [('proj1',layer.proj1),('proj2',layer.proj2)]:
            raw=spectrum(projection.weight)
            records.append(dict(layer=name+'.'+role,raw=raw,effective=raw,
                gate_source=layer.gate_source,operator_type='nonlinear_drift_factor',
                activation=stats))
    for name,layer in diagonal:
        eigen=layer.eigenvalues().detach().cpu()
        conditional=layer.duration(inputs[name])*layer.eigenvalues()
        records.append(dict(layer=name,operator_type='conditional_diagonal_flow',
            solver=layer.solver,clock=layer.clock,beta=layer.beta,
            eigenvalue_coordinates=layer.raw_eigenvalues.detach().cpu().tolist(),
            P_eigenvalues=eigen.tolist(),P_sigma_max=float(eigen.abs().max()),
            P_eigen_min=float(eigen.min()),P_eigen_max=float(eigen.max()),
            conditional_generator_eigen_min=float(conditional.min()),
            conditional_generator_eigen_max=float(conditional.max()),
            activation=layer.activation_stats(inputs[name]),
            interpretation='Spectrum of latent P=diag(lambda). The conditional diag(s)*P and complete velocity Jacobian are distinct.'))
    for name,layer in forced:
        eigen=layer.eigenvalues().detach().cpu()
        hidden=inputs[name]
        conditional=layer.duration(hidden)*layer.eigenvalues()
        records.append(dict(layer=name,operator_type='conditional_forced_gate',
            solver=layer.solver,clock=layer.clock,
            eigenvalue_coordinates=layer.raw_eigenvalues.detach().cpu().tolist(),
            P_eigenvalues=eigen.tolist(),P_sigma_max=float(eigen.abs().max()),
            conditional_rate_min=float(conditional.min()),conditional_rate_max=float(conditional.max()),
            forcing_abs_max=float(torch.nn.functional.silu(hidden).abs().max()),
            activation=layer.activation_stats(hidden),
            interpretation='P=diag(lambda); augmented generator has rates lambda*s and forcing SiLU(a). Bounded eigenvalues do not bound its shear norm or the full velocity Jacobian.'))
    for name,layer in state_clock:
        records.append(dict(layer=name,operator_type='state_clock_vector_field',solver=layer.solver,
            raw=spectrum(layer.proj.weight),activation=layer.activation_stats(inputs[name]),
            interpretation='Raw P is shared by internal stages. D(h)P and D(h_mid)P are conditional field generators; neither P spectrum nor a midpoint rule certifies the complete velocity Jacobian.'))
    for name,layer in lowrank:
        hidden=inputs[name]
        mean_duration=layer.step_size(hidden).mean(dim=0)
        records.append(dict(layer=name,operator_type='pruned_lowrank_K1',role=layer.role,
            raw=spectrum(layer.full_operator()),
            compressed_at_probe_mean_duration=spectrum(layer.compressed_operator(mean_duration)),
            activation=layer.activation_stats(hidden),
            interpretation='Raw P is the zero-padded square lifted operator. V*D*U uses probe-mean D, not every input or the complete field Jacobian; factor norms are gauge dependent.'))
    for name,layer in covariance:
        hidden=inputs[name]
        field=layer.field
        probe=field.field_jacobian(hidden[:32])
        symmetric=.5*(probe+probe.transpose(-2,-1))
        skew=.5*(probe-probe.transpose(-2,-1))
        eigen=torch.linalg.eigvalsh(symmetric)
        reference=field.field_jacobian(hidden.new_zeros(hidden.shape[-1]))
        records.append(dict(layer=name,operator_type='covariance_internal_field',
            mode=field.mode,heads=field.heads,atoms=field.atoms,
            gamma=float(field.gamma),encoder_singular_values=torch.linalg.svdvals(field.weight).cpu().tolist(),
            jacobian_at_zero=spectrum(reference),probe_count=len(probe),
            probe_symmetric_eigen_min=float(eigen.min()),probe_symmetric_eigen_max=float(eigen.max()),
            probe_skew_frobenius_mean=float(torch.linalg.matrix_norm(skew,ord='fro').mean()),
            activation=layer.activation_stats(hidden),
            interpretation='Actual internal f Jacobian; excludes D, gate derivatives, and the surrounding backbone.'))
        if field.mode!='untied':
            delta_energy=field.potential(hidden)-field.potential(layer(hidden))
            records[-1]['energy_change_mean']=float(delta_energy.mean())
            records[-1]['energy_increase_max']=float(delta_energy.max())
            records[-1]['sufficient_energy_descent_margin']=1-max(-float(field.gamma),0)*float(layer.step_size(hidden).max())/2
    return records
