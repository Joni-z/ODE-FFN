# Remaining structural question: a single hidden generator

The completed controls release exponential and midpoint coefficients. They
retain the same P across internal stages. Thus they do not identify whether
operator sharing itself helps. Do not describe the entire ODE contribution
as resolved while this remains untested.

Use the active numerical checks to distinguish an effect of the converged
map from its coarse discretization. Use symmetry interventions to locate
the error component, rather than requiring an advantage after correcting
that component. If the state-feedback signal remains worth explaining,
compare the current four-step midpoint with a stage-dependent generator:

```text
P_1 = P
P_j = P + Delta_j,  j=2,3,4
f_j(y) = P_j ISRU(y)
y_next = y + dt*f_j(y + (dt/2)*f_j(y)),  dt=1/4.
```

Initialize every Delta_j to zero without RNG draws. Both function classes
have identical initial outputs. Mathematical gradients to common weights
match; preflight must quantify any floating-point accumulation differences.
New Delta gradients must be nonzero under a nonzero readout witness.
The released model includes arbitrary stage matrices, while the shared model
is the nested case Delta=0. Both use eight dense P-type applications and
eight response evaluations per solve; matrix addition adds pointwise work.
There are3*64^2*6=73,728additional parameters in the small model, explicitly
reported. This is a same-work nested-class control, not a parameter-matched
claim. The qualified SwiGLU baseline is unchanged.

Shared P defines an autonomous hidden field and a one-parameter flow.
Stage-dependent matrices define a nonautonomous field. Both are ODEs;
neither should be called an ODE-free control. What is tested is whether the
single-generator constraint improves learning or parameter efficiency.

Keep the native training data, seed0,20k budget, response, optimizer and
outer sampler fixed. If shared wins held-out generation while released
matches/betters training fit, it supports a sharing/generalization bias.
If released matches/wins, a single autonomous generator has no demonstrated
benefit here. If only training fit worsens, diagnose optimization before
claiming a structural benefit. In all cases the ordinary controls and
actual quality/time frontier remain visible.

Status: registered next conditional control, not launched. First complete
the already active25numerical/symmetry tasks. Do not add another response
search, constrain the spectrum, or assume that hidden and physical time
have become mathematically identical.
