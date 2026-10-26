# A nonlinear16D population with an exact oracle

This population is a controlled analytic stress test. It is not presented as a new named benchmark or an exact reproduction of another group's dataset. Its purpose is fixed before any model outcome: distinguish a target with16 independent state-dependent posterior directions from the preceding2D Ring8 embedded among14 Gaussian coordinates.

Let latent coordinates be independent:

```text
Y_i = M_i mu_i + epsilon_i,
P(M_i=+1)=P(M_i=-1)=1/2,
epsilon_i ~ N(0,sigma_i^2),
mu_i = geomspace(.75,2.5,16), sigma_i^2=.04.
```

Rotate the result by a fixed orthogonal Q generated with seed0, as in the preceding16D task. The observed target is X1=QY. Components and Q are not supplied to the network. There are2^16 Gaussian components, with positive within-component variance and no singular support. The sampler stores only target coordinates, never component labels. Independence makes the oracle economical without enumerating65536 components.

## Exact conditional velocity

Because the Gaussian source is rotationally invariant, calculations can be made in latent coordinates. Denote a latent interpolation coordinate by z, with external FM time t. Define

```text
c_i(t) = (1-t)^2 + t^2 sigma_i^2
a_i(t) = [t sigma_i^2 - (1-t)] / c_i(t)
q_i(z,t) = t mu_i z_i / c_i(t).
```

The two posterior sign likelihoods have log ratio2q_i. Thus E[M_i|z,t]=tanh(q_i). Conditioning each Gaussian component gives the marginal velocity

```text
v_i(z,t) = a_i(t) z_i
           + [(1-t)/c_i(t)] mu_i tanh(q_i).
```

Differentiate directly:

```text
dv_i/dz_i = a_i(t)
  + [t(1-t) mu_i^2 / c_i(t)^2] sech(q_i)^2,
dv_i/dz_j = 0 for i != j.
```

Observed velocity and Jacobian are Qv(Q^T x,t) and Q diag(dv_i/dz_i) Q^T. At t=0 the velocity is -x, since the population mean is zero; at t=1 it is x. These are finite, smooth endpoints. The density is also analytic: sum over coordinates of -.5log(2pi*c_i)-(z_i^2+t^2mu_i^2)/(2c_i)+logcosh(q_i), with a stable logaddexp evaluation.

All within-component variances are equal, so a_i(t)=a(t). The remaining diagonal correction is positive for every finite input at0<t<1. Its rank is16 mathematically, and its values can vary independently with the16 latent coordinates. The difference of Jacobians at zero and at a point with every latent coordinate nonzero generically has rank16. A small-dimensional implementation test compares against an explicitly enumerated GMM; a16D test verifies this rank statement.

Full mathematical rank is not the same as a large effective rank on typical samples. Close to unambiguous wells the sech-squared terms can be tiny. Population-only calibration must report their participation ratio and active directions at the registered times; do not claim that every generated trajectory needs16 equally large corrections. More importantly, none of these facts identifies hidden P with the target Jacobian or proves a width32 controller sufficient for a whole trained network.

## Fixed validation plan

Prepare100000 training,20000 validation,20000 test targets and2048 probes from independent streams using the existing versioned generator, seed0. Freeze the data manifest. Architecture selection continues to use validation only; model test evaluation follows a fixed architecture and endpoint. Existing exposed test banks cannot be reused as untouched confirmation for a new architecture.

Report per-coordinate oracle error, full Jacobian error, error along the known latent coordinates, sample SW2, well-membership errors per coordinate, marginal sign balance and correlations between latent signs. With4096 samples and65536 population modes, demanding observed occupancy of every joint mode would be invalid. The population contains all sign combinations, so marginal correctness alone is also insufficient: joint/sign dependence must be checked. These measurements are diagnostic, not additional training losses.


Population calibration on20000 validation points gives median participation ratios10.83,10.22,5.98,1.72,1.02 at t=.1,.3,.5,.7,.9. Median numbers of corrections above10% of the largest coordinate correction are15,15,8,2,1. Thus the task supplies several simultaneously ambiguous directions early/mid-path, while late-path ambiguity is sparse. Both properties must be retained when interpreting model errors; the task is not uniformly rank16 in a numerical/effective sense. Values are fixed population diagnostics in product16-population.json, not model selection outcomes.
