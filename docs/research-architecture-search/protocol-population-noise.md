# Stage10: identify conditional-noise sensitivity under one population law

Registered before this stage's outcomes. This is a diagnostic using unchanged architectures, not a deployable oracle-dependent method or another candidate P constraint. Stage8/9 remain stopped. The finite-bank mismatch and the conditional-gradient derivation are recorded in empirical-target-readout.md and next-identification-question.md.

## Question and exact risk relation

Use fresh independent target samples Y and standard Gaussian source Z, with t uniform and X=(1-t)Z+tY. Compare pair labels U=Y-Z with the analytic conditional mean m=E[U|X,t] of that same population. For any fixed predictor,

```text
E ||v_theta-U||² = E ||v_theta-m||² + E tr Cov(U|X,t).
```

The objectives therefore have the same real-arithmetic population minimizer and expected pre-clipping gradient. The conditional gradient covariance is4*A^T*Sigma*A, with A=partial v/partial theta. Adam's moments and gradient clipping respond nonlinearly to noisy gradients, so their actual update paths need not have equal means. We retain both mechanisms unchanged and log clipping fractions; this comparison studies their response to conditional target noise, not an Adam convergence theorem.

The previous cached-bank teacher generally differs from this analytic teacher. Both new arms use the same fresh population/input law. A change relative to the older bank runs therefore is not attributed solely to conditional-noise removal. The old audited bank endpoints are contextual evidence; only pair-versus-mean **within this stage** is an exactly paired supervision comparison.

## Twelve cells, no architecture change

On Ring8 and rotated product16, cross three unchanged models with two label choices:

- `vector_pair`, `vector_mean`: original independent vectorK1 ODE composition, retained middle SiLU,gate hidden32/reference64.
- `swiglu_budget_pair`, `swiglu_budget_mean`: nominal hidden96/actual64.
- `swiglu_pair`, `swiglu_mean`: nominal hidden128/actual85.

Each trains5000updates,seed0,width64,depth3,batch512,AdamW3e-4,wd0,clip1. Sampling/evaluation panels and Heun32 settings stay the same and use validation only. No covariance field, range relaxation, sharing, spectral cap or new optimizer is added. Reference initial states and the nonzero-output forward function must remain unchanged. Initial tensors and the complete actual input stream match between each model's two supervision arms.

The population parameters are exactly the manifest's values. Target generation uses double precision before rounding to the established float32 training-input format; teacher calculations use double precision before returning float32 labels. These numerical conventions are tested against the existing NumPy oracles, including endpoint limits and rotated unequal-covariance cases. The model receives no component identities, rotations, locations or analytic parameters as additional inputs. The mean-label arm uses privileged analytic supervision and is only a fitting/noise control.

Use an explicitly seeded data generator independent of model initialization. Every step samples fresh Y, then Z and t in an identical order across all six arms of a dataset. Teacher evaluation consumes no random draws. Hash-chain the actual Y/Z/t bytes after every minibatch, record the fingerprint at checkpoints, and retain it across exact resumes. Independently regenerate the stream once per dataset during the native audit, and compare all checkpoints/arms. Default bank-mode behavior must remain bit-identical to the preceding frozen trainer.

## Measurements and interpretation fixed before outcomes

Primary: final random-time validation population-oracle MSE. Retain full curves, common-field/Jacobian errors, sampling SW2, all costs and per-interval clipping fractions. Label-dependent training loss is called train_target_mse; mean-label training loss is not reported as pair loss. Never compare the pair and mean raw training-loss levels as if they share an irreducible floor.

For each model/task record pair-error/mean-error. Compare the ODE gap against the better of both SwiGLU sizes separately under each supervision choice:

- If ODE mean supervision is within10% of the better mean-supervised reference on both nonlinear tasks, while pair-supervised ODE is at least25% worse on either, conditional-noise sensitivity becomes a concrete hypothesis for the observed gap. This does not prove expressivity equality or universal optimizer causation.
- If ODE remains at least25% worse than the better reference under mean supervision on both tasks, conditional noise alone has not closed the fixed-budget gap. Investigate internal parameter response/fitting before claiming a representational lower bound.
- Otherwise retain a mixed/task-dependent result. Do not choose a global mechanism by looking only at the most favorable task or checkpoint.

These are diagnostic interpretation thresholds, not significance claims. No architecture is accepted from an oracle-supervised win. Any future candidate still needs ordinary pair-target training, both practical references, the original task/control scope, frozen fresh holdout and actual cost evidence. No automatic extension, loss trick or architecture rescue is authorized by a failed diagnostic. Runtime/source are immutable and mirrored; two CPU workers and10s telemetry continue on Lenovo.
