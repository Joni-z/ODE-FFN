# Can input-dependent operator directions help without changing its spectrum?

**Question.** Does choosing a small orthogonal input frame around the unchanged professor K2 polynomial improve the complete thin/thick clean-prediction comparison, beyond a global-frame parameterization and strong ordinary controls?

**Evidence.** Cycle09 attributes95–98% of the small thin/clean gap to remaining fitted error after conditioning on the learned input. Cycle10 finds99.97%/99.81% symmetric late-time tangent derivative residual for the original ODE. Older Product16 clock-response bounds show a restrictive trained response frame, without proving P alone causes that restriction. These observations motivate testing response orientation; they do not predict a gain.

**One change.** Keep the original P, scalar clock, K2 arithmetic, two independent solves, middle SiLU and entire backbone. Replace each conditional operator by Q(x) P Q(x)^T using one identity-initialized pair of Householder reflectors. No spectral flattening, new loss, changed time sampler or additional solver stage.

**Controls.** A global input-independent frame has the same added tensors/inference work but can be absorbed into P, testing extra parameterization. An input-frame K1 gated residual has the same learned tensors but no quadratic term. Swi192 nearly matches parameters and Swi257 dense inference/training work. Reuse sealed original K2/Swi192 trajectories and train the other four families on exactly the two unchanged clean-head banks.

**Limits.** Orthogonal adaptation and input-conditioned reflections are established. The proposal is a conditional low-rank hypernetwork interpretation of the old polynomial, not a novel solver or a proven FM-specific improvement. Its local controller-response witness is not a full-network capacity theorem. Whole-map Jacobians include frame and clock derivatives, so preserving a conditional spectrum is not a global stability certificate.

**Support/falsification.** The fixed protocol requires a material field gain against both Swi controls on thin data, no thick regression and derivative/generation guards, plus a gain over both mechanism controls. Failure stops this one frame/initialization/order setting. A pass only schedules confirmation on the older Product/dependent-sign tasks; it cannot complete the successor objective.

**Readiness.** Conditional GO after mathematical, implementation, budget, initial-pairing and native-resume gates pass. No fitted candidate outcome has been inspected at registration.
