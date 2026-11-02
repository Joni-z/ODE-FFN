# Prospective cycle3: order interaction and a smaller ordinary control

Registered after all18 cycle2 endpoints completed and replayed. Native validation must precede dispatch.

Original scalar K1 changes only h+sPh versus the existing K2 polynomial,
retaining identical tensors, controller, backbone and data stream. This tests
whether the observed Muon benefit requires P^2. Use selected-matrix AdamW.0003/.001
and official Muon.006/.02, both tasks,20k updates,batch512,seed0:8 new arms.
All K2 references and both ordinary widths/rates remain in the comparison.

Native SwiGLU nominal64 supplies a parameter-matched ordinary reference for
cycle2's gate-free designs. Its exact Product16 counts are58,572 parameters and
57,024 dense forward MACs, against58,512/81,984 for direct PD,K2. Cross the same
four optimizer/rate settings on both tasks:8 controls. The existing nominal128
ordinary control already approximately matches the direct-K2 dense MAC budget.

The direct-raw K2 Ring8 lead motivates an additional cold direct-raw K1 block,
retaining58,512 parameters and changing only each solve's order to1. Cross AdamW.0003
and Muon.006/.02 on both tasks:6 arms. Compare with its exact same-rate K2 parent.
Its dense MAC budget is57,408, close to SwiGLU64's57,024, so both parameter and
compute comparisons can be made without calling equal steps equal compute.

Total22 new trajectories:8originalK1,6directK1,8ordinary controls. These are three
independent blocks, not a combined candidate or an extra trick.
Do not describe a K1 improvement as an ODE accuracy theorem, or a gain over the
smaller reference as a compute advantage. Complete numerical identities, actual
CUDA continuation, source freezing and monitored execution before any dispatch.
Use unchanged native SwiGLU, not a new FFN implementation.

## Registered triage

For each K1 candidate retain a diagnostic lead only at>=5% lower validation field error than the same-optimizer/rate K2 parent; always show sampling tradeoffs. SwiGLU64 is a reference, not a candidate to discard. For direct K2 also compare with both SwiGLU64(parameter) and SwiGLU128(MAC), retaining all rates. No test-based checkpoint or candidate selection.

The frozen precision check did not reveal a material field-MSE discrepancy for the raw lead, despite its large local hidden Jacobians. This supports continuing the controlled order/budget test; it does not certify hidden-flow stability. See cycle2-results/precision/.
