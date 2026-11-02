# Prospective cycle2: rank control and direct input diagonals

Status: design prepared while the final two cycle1 candidates per task finish.
No cycle2 dispatch until cycle1 replay/decisions are published and native checks pass.
The user explicitly requests various D constructions; these are not additional
placement or temperature sweeps.

Candidate1: factorized-affine bounded controller. Replace the32-hidden SiLU by
identity with the exact local-value/Jacobian initialization in cycle1.md. Its
parameter count and logit-Jacobian rank cap match the MLP parent. This distinguishes
full affine's potentially higher rank from its removal of nonlinear features.

Candidate2: direct D(h)=diag(h), with all controller parameters removed.
Candidate3: direct bounded D(h)=diag(.2+.8 sigmoid(h/10)), also no learned controller.
The direct pair has identical parameter count and dense MACs; their transforms
and input derivatives differ. They do not have equal cold functions or amplitude,
and a result cannot be attributed exclusively to positivity or boundedness.
Keep those limitations explicit instead of adding a fitted scaling trick.

All three retain PD,K2,two solves/SiLU, original P/backbone draws, data/RNG/loss,
seed0,batch512,20k updates and the fixed evaluation. Cross each with selected-P
AdamW.0003 and official Muon.006/.02 on Ring8/Lenovo and Product16/145:18 new arms.
Reuse all verified parents and ordinary controls, conditional on exact native
three-update checks. Direct gates intentionally reduce parameter count; compare
them with each other for the intervention and with the strong ordinary controls
for practical utility. No claim of equal capacity to the learned-gate parents.

At K2, direct input D supplies exactly
F(h)=h+P(h*h)+.5 P[h*P(h*h)], an explicit quadratic/cubic computation.
It is a frozen-generator polynomial, not a solver that recomputes D along an
internal trajectory. Negative D entries do not make this finite map undefined;
they remove the interpretation as positive per-channel elapsed time.

Continue recording actual conditional A spectra, full local Jacobians, P/update
spectra, gate signs/range and term magnitudes. Raw-gate zeros must not create
log/division errors in diagnostics. Check zero input, signed inputs, dense formulas,
full gate derivatives, physical optimizer state and actual CUDA trainer resumes.
No change to loss, activation between solves, data or output head.

The factorized-linear initialization and cycle1 full-affine initialization actually
compute the same affine logit at every h in exact arithmetic, not merely at zero.
They start from the same complete function with different factorization/rank
constraints and parameter-gradient geometry. A later difference is therefore
not an initial-function effect; it still cannot be attributed to rank alone.
