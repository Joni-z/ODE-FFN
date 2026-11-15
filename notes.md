# ODE mechanism notes

The response screen is complete. Its positive Ring8 arctan signal does not
identify an ODE mechanism. This task supersedes immediate response-only image
promotion as the recommended priority.

Read: next-architecture/recommendation.md; order-response/findings.md;
response-screen/analysis-report.md. Relevant known issue: input-conditioned
A(h) is held fixed within each solve; composition reevaluates A at a new input.
A conditional matrix exponential need not give a semigroup on hidden states.
Full vector-field error transport is a separate issue from hidden-step time.

Plan only: no response sweep, optimizer switch, or new formal training.

Derived: F_s(h)=exp(s A(h))h generally violates hidden-state composition at
order s*u*(J_f-A)f. For ISRU this is s*u*P(D^3-D)Pq(h). Exact state evolution
uses D^3 in its second-order coefficient. These are different equations;
increasing Taylor order fixes neither distinction nor task utility.

Minimal first test: fixed Taylor8 versus learned alpha2..8 on the same
Taylor terms, alpha initialized1. Same P work and initial function;42 extra
scalars per small model. Seed0, native20k recipe; two new trajectories only.
Optional second test: four midpoint substeps of y'=P ISRU(y), eight P calls.
Two further trajectories only if trained-state diagnosis supports it.

Physical bridge: oracle trajectories, variational transport of full-field
error, and predeclared time-window oracle repair interventions. Test the
linearization at small error scales before extrapolating to the learned model.
Persistent full details and decisions are in research-question-card.md.

Primary literature: flow-map composition/tangent conditions are prior art;
numerical-solver dependence is prior art. A recent strain/vorticity preprint
contains an invalid general explicit-Euler norm inequality; excluded as a
theorem foundation, with a skew-matrix counterexample. No novelty claim is
made from the standard identities or midpoint construction.
