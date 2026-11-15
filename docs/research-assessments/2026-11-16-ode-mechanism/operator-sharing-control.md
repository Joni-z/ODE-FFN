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

## Execution amendment

The completed full-sample fixed-midpoint4->16 checks retain the Ring8
within-family gain (SW2 0.071->0.074, still below fixed-K8 0.088). Product16
changes0.052->0.055, close to fixed-K8 0.056. This is enough to ask which
part of the state-feedback parameterization matters. The C8interventions
will locate the affected error component; they are not a prerequisite for
defining the sharing control. Overlap the independent control with the
remaining numerical work instead of leaving training idle. This changes
execution order only; it does not add a response or optimizer search.

Two native20k CPU runs, seed0, batch512, AdamW0.0003, no weight decay and
original clipping1. Their controls are the existing shared state-flow,
Free ISRU and SwiGLU checkpoints. Generate new selection/confirmation
streams2026112000/2026112100 before training; they are evaluation-array
RNGs, not new training seeds. All eight model/task combinations receive
the same new arrays. Reuse no opened confirmation scores to select a
checkpoint. Evaluate15k/20k on selection and20k on confirmation.

Current environment/data stay unchanged. Keep two one-thread training
workers; existing inference has two separate one-thread workers on the
eight-core Mac mini. No145GPU is allocated while occupied. Finish all
existing25tasks; none is canceled or duplicated by this amendment.

## Fresh documented readiness

Run verbatim from `/Users/mr.z/ODE-FFN-upstream`:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python \
studies/muon_pd_20261102/coefficient_20261116/sharing_check.py \
--data-root /Users/mr.z/ode-workspace-prep/data/synthetic \
--assets /Users/mr.z/ode-workspace-prep/coefficient-20261116/sharing-assets \
--output /Users/mr.z/ode-workspace-prep/coefficient-20261116/sharing-doc-witness
```

Require identical initial outputs, common-gradient agreement at
atol1e-14/rtol1e-10 (log actual error), nonzero new gradients, explicit
formula/gradient checks, two native8->16 continuation witnesses and two
real evaluator witnesses. Freeze/publish source before formal launch.
The generic initial check measured common-gradient roundoff below5e-19.

Original global gradient clipping includes the additional matrix gradients;
its policy is unchanged, but effective clipping can differ. A training-fit
deficit must therefore be treated as an optimization outcome, not automatic
proof of a useful ODE constraint. Keep training curves and all stage matrices.
Do not assume hidden and physical time have become mathematically identical.
