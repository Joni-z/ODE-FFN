# Prospective cycle3: order interaction and a smaller ordinary control

No dispatch selected yet; cycle2 must finish and be replayed first.

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

These are two independent blocks, not a combined candidate or an extra trick.
Do not describe a K1 improvement as an ODE accuracy theorem, or a gain over the
smaller reference as a compute advantage. Complete numerical identities, actual
CUDA continuation, source freezing and monitored execution before any dispatch.
Use unchanged native SwiGLU, not a new FFN implementation.
