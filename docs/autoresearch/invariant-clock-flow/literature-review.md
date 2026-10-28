# Prior art and the actual experimental question

Zhizhe Zhang · October 28, 2026. Bounded primary-source review, not a novelty
certificate. No reported result below is transferred to our FM tasks.

## ER-20261028-action-angle-01 — the construction already exists

[Daigavane et al., Action-Angle Networks, 2022v1](https://arxiv.org/html/2211.15338v1),
Section2, Eqs4–7. Source: full primary workshop paper, relevant methods read.
An MLP takes radial invariants and advances angles; learned coordinate
transforms encode/decode the physical state. The evolution avoids repeated
numerical integration. This is direct prior art for invariant-conditioned
rotations, not evidence about a JiT FFN. We add no learned coordinate transform.
Claim strength: supported construction/transfer boundary; no novelty claim.

## ER-20261028-integrable-reference-01 — exact flow is a restricted family

[He, Neural Vector Field Transformations, October2026v1](https://arxiv.org/html/2610.29039v1),
Section2.5 and Lemma9. Source: full primary preprint, relevant sections read.
It explicitly rotates coordinate pairs with rates depending on squared
amplitudes. A multivariate rate map is Hamiltonian only when it is a gradient.
Its conjugacy approach targets physical vector fields with fixed qualitative
invariants. Our hidden processor is not identifying such a physical system;
no data-derived frequency initializer or learned conjugacy is imported.
Claim strength: supported prior art and terminology boundary. It does not
predict an FM gain or make this layer a new mathematical flow family.

## ER-20261028-ngpt-scope-01 — normalized features do not prescribe this flow

[nGPT, 2024v1](https://arxiv.org/html/2410.01131v1), Sections2.2/2.6.
Source: full primary preprint, relevant architecture recipe read. The recipe
normalizes states and matrices, changes residual updates and parameter scales,
and removes weight decay/warmup. It is not a single-FFN-rotation experiment.
Its language-model convergence results cannot validate our local constraint.
Keep our surrounding normalization, optimizer and objective unchanged.
Claim strength: supported scope boundary; no transferred quantitative claim.

## Project evidence and distinctions

- [The actual FFN input](../posterior-state-eligibility/hidden-input-check.md)
  has controlled scale, but learned norm weights and time modulation do not
  place every token on one exact sphere. Preserving amplitudes is an inductive
  hypothesis, not a consequence of Gaussian denoising or a data-manifold theorem.
- [Cycle03](../cycle03-gated-shear/analysis.md) already makes unchanged SwiGLU
  an exact conditional shear. An ODE representation alone earns no benefit.
- [Cycle04](../cycle04-gate-transport/analysis.md) added nilpotent gate mixing;
  [cycle11](../cycle11-conditional-frame/analysis.md) changed input-dependent
  matrix directions. Both stopped settings remain stopped. This candidate
  instead constrains P to fixed coordinate pairs and gates it using invariants
  of the actual evolving state; no input frame or new gate-transport matrix.
- [The frozen exponential study](../cycle21-image-exponential/analysis-report.md)
  did not establish a solver-accuracy benefit. Accordingly, exact rotations
  must face both nonconserving Euler and conserving normalized-Euler controls.
- [Cold ordinary controls](../image-clock-training/analysis-report.md) require
  initialization and shifted-matrix coordinates to remain explicit. A function-
  matched ordinary SiLU and fixed0.6P will be included, without claiming equal
  initial trainable tangent spaces to a constrained frequency model.

## Decision

There is a coherent constraint to test, not a new action-angle algorithm:
amplitude-preserving, coupled nonlinear feature transport inside the original
two-solves/SiLU FFN template. A finite old-law screen is justified only after
full derivatives, actual cost, active routes and baseline/resume checks pass.
The efficient implementation is an ordinary gated rotation, and a useful
constraint need not be renamed as an exclusive ODE computation. No empirical
advantage is inferred from the identities or this literature.
