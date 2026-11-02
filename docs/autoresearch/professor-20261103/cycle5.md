# Cycle5 registration: sign/orientation versus ordinary bilinear features

Registered after all18 cycle4 outcomes and all12 frozen parity cases, before
cycle5 outcomes. No accepted ODE model exists. RawK1 has a small Ring8 field
lead with worse sampling; Product16 ordinary controls remain stronger.

## Evidence and hypothesis

At the same P,h, Pdiag(h) and Pdiag(abs(h)) have exactly identical singular
values. Yet their K1 increments are P(h*h), an even feature, and P(h*abs(h)),
an odd feature. Both population target fields are odd. In the frozen Product16
diagnostic, excess even energy accounts algebraically for74% of rawK1's paired-
error gap to Swi128. Ring8 rawK1 has lower odd error but higher even energy than
the shown Swi controls. This motivates a sign/orientation contrast rather than
another singular-value cap or arbitrary gate range.

This is not a proof that gate parity causes the full model's error. SiLU, biases
and time modulation remain; the new whole predictor is not forced odd. The abs
feature is C1 with a second-derivative change at zero. Learned states and P can
diverge, so trained operator spectra are not guaranteed to remain matched.

## Registered families and14 cells

1. pd_direct_abs_k1: use d(h)=abs(h) with rawK1's exact inherited tensors/RNG,
   K1 and no controller. AdamW.0003, Muon.006/.02 on each task:6 cells.
2. bilinear64, ordinary reference: construct the unchanged native SwiGLU64,
   retain every tensor/RNG draw and replace only SiLU(gate)*value by gate*value.
   AdamW.0003/.001 and Muon.006/.02 on each task:8 cells. This known bilinear
   family is described in Shazeer's GLU Variants Section2, not a novel method.
   The native biases are retained; do not call this an exact paper replication.

Product16 parameter/MAC counts are58,512/57,408 for absK1 and58,572/57,024 for
bilinear64, matching the rawK1/Swi64 resources. Keep completed rawK1, constant
ordinary FFN, scalarK1 and all Swi64/128/192 controls. The bilinear control must
not be omitted if it explains a positive quadratic-feature result.

Both tasks/hosts, caches, seed0, batch512,20k final checkpoint, optimizer groups,
clipping, auxiliary AdamW.0003, validation bank and primary Heun32 stay fixed.
No symmetry projection, augmentation, extra loss, regularizer, scheduler,
full-network Muon, package install or image transfer. The symmetry premise is
specific to these targets, not a general property of natural images or FM.

## Decisions and registered follow-up readout

Provisional keep for abs requires >=5% validation improvement over matching
rawK1 optimizer/rate; bilinear is a reference, not a candidate. Report every
sampling result separately. No accepted method from a weak-parent win or one
task; retain the strongest ordinary controls including the constant FFN.

After replay, select the lowest-validation final checkpoint within each of the
two new families per task for the same frozen paired-parity diagnostic. This
selection is validation-only, not based on sampling or even-energy scores.
Compare both odd error and even energy against the previous12-case bank.
Lowering even energy without a useful field/sampling tradeoff is insufficient;
no post-hoc parity penalty or projection rescue. A benefit shared by ordinary
bilinear features weakens an ODE-specific interpretation.

## Plan

- [x] Implement isolated trial5.py and fixed grid using the existing trainer.
- [ ] Check exact tensors/RNG, conditional singular-value identity, full Jacobians
      including zero coordinates, native optimizer serialization and actual8-to16 resume.
- [ ] Freeze/push, deploy new worktrees, pass fresh documented CUDA witness.
- [ ] Run all14 with two monitored workers per free GPU; replay all outcomes.
- [ ] Run the registered four-checkpoint parity readout, publish full controls,
      and decide the next direction from the complete field/sampling evidence.

CPU checks pass: exact inherited tensors/RNG, matching conditional spectra,
full derivatives at zero/signed inputs, all original optimizer-arm identities
and physical state continuation. Fresh CUDA/native trainer smokes remain required.
Formal grids dispatch the ordinary bilinear references before abs candidates.
