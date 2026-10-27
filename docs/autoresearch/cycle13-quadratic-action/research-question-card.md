# What does the second P application add at fitted layer inputs?

**Question.** Is the professor's K2 increment substantially outside a scalar K1 direction at actual hidden states, after accounting for one fixed matrix reparameterization? This is a mechanism diagnostic, not a proposal to reduce order or rescue cycle11.

**Why now.** In cycle11, input-frame K1 nearly matched K2 on thin field fitting; its thick result was not uniformly worse. Cycle12 also separates a shared output restriction from P-specific claims. A new high-order method therefore needs evidence for useful extra action, not just a convergent Taylor series. Existing direct-effective-map teacher experiments compare constant matrices around SiLU; they do not measure this input-dependent quadratic action at the fitted K2 states.

**Hypothesis to inspect.** A noticeable raw quadratic term can mostly be absorbed by `A=P+.3*P^2` at the original duration centre c=.6. The exact remainder with the original clock is `.5*s*(s-c)*P^2*x`. Remaining action outside `A*x` cannot be removed by any scalar duration at that fixed A/input. Whether either quantity is small on all six solves is an empirical question, not assumed before the run.

**Cohort and method.** All original, input-frame and global-frame K2 checkpoints from both clean laws, steps 0/1000/2500/5000: 24 checkpoints, six solves each. Capture all 20,000 native field-bank inputs per checkpoint in the original 1024-row evaluation batches. Report all layer/checkpoint outcomes, scalar least-squares envelopes with and without the original duration bounds, plus one frozen whole-network substitution using A and the original live clocks/frames. Preserve all models, banks and prior seals.

**Support and falsification.** Report continuous relative RMS residuals and the prespecified 1% local/1% complete-field-displacement descriptive flags. A small error supports only that frozen-function approximation on this bank. A large bounded-clock residual rejects the approximation at fixed A and exposes which layer/time bin fails; it does not establish an architecture-wide limitation. No empirical outcome nominates an oracle clock, retrained K1, new spectral constraint, different centre or new architecture.

**Novelty and practical limits.** Projection onto a scalar direction is elementary least squares. Matrix-action/Krylov reasoning is classical. This study supplies project-specific fitted evidence, not a novel solver. The relaxed clock may use arbitrary functions of the input and its closed interval includes sigmoid endpoints; it is not the finite MLP used by a trained K1 model. Local errors need not propagate linearly through the network.

**Decision owner.** The researcher retains the choice of a future architecture. This diagnostic can reject a rationale; it cannot substitute a theoretical label for an actual win against strong ordinary controls.
