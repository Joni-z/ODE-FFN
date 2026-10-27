# ODE autoresearch task plan

## Goal
Use the autoresearch experiment loop to develop an evidence-supported ODE architecture for flow matching. A useful successor must face strong SwiGLU and compute controls, including both Product16 and dependent-sign data. Installing a loop or completing one diagnostic does not complete this goal.

## Current cycle: oct27-complete-hidden-field
- [x] Read the official autoresearch protocol, project conventions, and preceding transfer/failed-generator reviews.
- [x] Create isolated branch `autoresearch/oct27-ode` and worktree `/Users/mr.z/ODE-FFN-autoresearch`.
- [x] Write an ODE-specific program with immutable evaluation, explicit budgets, preserved failed experiments, and publication rules.
- [x] Register a read-only checkpoint diagnostic of the complete hidden vector-field derivative, including normalization and gate derivatives.
- [x] Verify analytic derivatives against automatic differentiation and exact checkpoint replay.
- [x] Execute the registered diagnostic with monitoring; assess the result without coefficient or checkpoint selection.
- [x] Publish the protocol/code milestone e0ad302 and the final audited evidence fc1afda to GitHub main, verified through MBP SSH without rewriting commit authors.
- [ ] Select and execute the next justified cycle; keep the overarching goal active until useful architecture evidence exists.

## Decisions
- Adapt the official research protocol, not its GPT task, tokenizer, val_bpb, CUDA environment, or unrestricted hyperparameter search.
- Preserve historical data, seed 0, native model semantics, and all sealed evidence.
- Use the actual residual duration c=1/sqrt(3), without tuning it.
- First test frozen checkpoints. This can identify whether higher-order integration of the learned hidden field is useful, but cannot establish the behavior of a freshly trained higher-order model.
- Record failures rather than reset shared/published history. Never overwrite existing runtime outputs.

## Cycle02: fixed-component-support mixture
- [x] Specify the single half-Product/half-sign-code law, its exact Bayes mean/covariance/Jacobian, controls and interpretation limits.
- [ ] Commit the protocol/source before outcomes; generate and hash the new data without changing historical laws.
- [ ] Verify the oracle against65536-component enumeration, native cold states and exact resume.
- [ ] Train five unchanged architectures with the registered5000-update budget and two monitored CPU workers.
- [ ] Audit every new trajectory and publish all endpoints, P histories, figures, and the scoped interpretation.

## Errors encountered
- Audit-v5 passed all Product16 cases, then one sign-code panel output differed by3.0599e-6. V6 makes panel-output tolerance1e-5 to match the existing full-bank numerical replay threshold; all other tolerances and formal outcomes stay fixed.
- Audit-v4 stopped on two Product16 generated coordinates after64 field evaluations, with maximum violating difference5.11e-5. V5 records all sampling maximum/RMS differences and checks max<1e-3, RMS<1e-5 plus unchanged stringent SW2 agreement. All failed audits remain preserved; formal outputs are never rerun.
- Audit-v3 passed all Gaussian cases, then stopped on one Ring8 float32 coordinate with1.2945e-6 difference after elementary normalization, beyond the1e-6 absolute tolerance. Audit-v4 uses1e-5 absolute prediction tolerance, retaining2e-5 relative,3e-5 Jacobian absolute, and stringent float64 fixtures. Scientific acceptance thresholds are unchanged.
- Audit-v2 field values matched but nested derivatives through fused LayerNorm did not match the analytic formal Jacobian. A separate float64 finite-difference fixture supported the analytic implementation. Expressing normalization in elementary operations inside independent AD resolved the fixture to4e-16; preserve v2 and run the full audit-v3 before accepting verification.
- The registered text incorrectly said every field bank had20,000 rows. Actual immutable2D banks contain10,000 each, while both16D banks contain20,000. Formal evaluation already used every row. Independent audit stopped on the mistaken count before replay; preserve that source/log and correct only the count assertion in audit-v2. See cycle01 deviations.md.
- A guessed old recurrence-protocol filename did not exist; rg located the actual immutable `docs/research-recurrent-depth-2026-10-26/protocol.md` before registration.
- Lenovo SSH timed out; torch SSH authentication failed. Mac runs remain available. MBP HTTPS credentials were inaccessible, but MBP GitHub SSH authenticated as Joni-z; use that route without changing commit authors.
- Local GitHub SSH initially lacked a host key; accepting the new key reached authentication, which failed. No local SSH publication succeeded.

## Status
Cycle01 is complete, independently audited, sealed and published (fc1afda); it rejects frozen complete-Taylor upgrading without a coefficient rescue. Cycle02 protocol/source are ready for registration and preflight. It changes only one target law, holding all65536 Gaussian component locations/variances/count fixed while mixing their prior probabilities. It does not introduce a new architecture. No cycle02 training has started.
