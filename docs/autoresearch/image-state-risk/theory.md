# What the reference measures

Let X be uniform on the original45,000normalized training images and let
Z=tX+(1-t)e with independent standard Gaussian e. We use the ideal-real
Gaussian observation law with the exact stored float32 clean values promoted
to float64. This does not assert that a quantized floating-point RNG has a
continuous density.

Bayes' rule gives a weight proportional to exp(-||z-t*x_i||²/[2(1-t)²]).
The ||z||² term is common to every atom, giving the logits in protocol.md.
The conditional mean is m=sum(w_i*x_i). For a fixed prediction d:

```text
E[||d-X||² | Z=z] = ||d-m||² + tr Cov(X | Z=z).
J_z m = t/(1-t)² * Cov(X | Z=z).
```

The first identity follows by expanding X=m+(X-m) and using zero conditional
mean for X-m. The second follows from differentiating normalized Gaussian
weights. The reference derivative is PSD, but there is no universal upper
bound1 or global contraction claim. Differentiation of a hidden normalized
token has no such guarantee from this argument.

For our five times,1-t>.05, so native capped velocity and the uncapped
formula agree at these probes. Conditional velocity-risk excess is
||d-m||²/(1-t)², with corresponding conditional variance divided by the same
factor. Equal-time averages here are declared diagnostic summaries, not the
native logit-normal integrated training objective.

These identities hold for the specified finite prior. A held-out validation
image was not an atom of that prior. Consequently, m need not minimize its
observed paired clean-image error, and smaller distance to m on arbitrary
generated states is not proof of population quality. The independent audit
uses direct squared-distance likelihoods; no fitted Gaussian surrogate or
new component state is introduced.
