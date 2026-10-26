# Pruned augmented-state K1: design and prior-art review

2026-10-25. **Conditional GO for one bounded prototype, not training authorization or an accepted architecture.** The hypothesis is joint budget reallocation: retain learned conditional modulation, but provide more coordinates before SiLU using lower-rank projections and narrower controllers. The completed [clock collapse](analysis-clock-collapse/analysis-report.md) establishes useful frozen-model dependence, not that expansion will improve learning. Neither novelty nor an FM-specific advantage is established.

## Review the pruned formulation

Let d=64, expanded width r=128, projection rank q=21 and controller hidden width g=21:

```text
z = [h;0] + s1(h) * U1 V1 h      U1: r×q, V1: q×d
y = SiLU(z)                      gate1: d→g→r
out = y[:d] + s2(y) * U2 V2 y    U2: d×q, V2: q×r
                                 gate2: r→g→d
```

Multiplication by s is coordinatewise; both clocks retain the original sigmoid/temperature. There is no learned encoder, decoder, forcing bias or additional activation. This is exactly zero-padding, two original K1 maps and fixed output selection, after deleting the first map's permanently unused input columns and the second map's unobserved output rows. That pruning is specific to this K1 composition; it does not license higher-order solves. The earlier naive square128 proposal is not the candidate reviewed here.

From [the existing constructors](../../synthetic/models.py), two-map dense MACs are `2(d+r)(q+g)`; parameters add `2g+d+r` controller biases. Existing-model counts were independently instantiated; proposed counts are algebraic, not measured executions.

| FFN | Dense MACs | Parameters |
|---|---:|---:|
| Original full-P64, controller32 | 16,384 | 16,576 |
| Pruned r128,q21,g21 | 16,128 | 16,362 |
| Unaugmented r64,q21,g21 | 10,752 | 10,922 |
| Unaugmented r64,q32,g32 | 16,384 | 16,576 |
| Ordinary SiLU128 | 16,384 | 16,576 |
| SwiGLU128, actual85 features | 16,320 | 16,554 |

The candidate's unchanged Product16 backbone totals83,022 parameters/81,216 MACs, below SwiGLU's83,598/81,792. Half its dense computation still goes to each of projections/controllers. More pointwise operations, intermediate storage and matrix calls remain; equal dense MACs do not establish equal wall time.

## Prior art supplies limits, not a performance promise

**Augmented Neural ODEs** (Dupont et al., NeurIPS2019) already introduces zero-augmented states. Its topology argument concerns uniquely solvable continuous flows and their homeomorphisms. The professor's finite K1 map can already be noninjective: constant s=c and P=−I/c give the zero map. The middle SiLU is not globally invertible either. Therefore ANODE's obstruction is not our demonstrated problem, and its solver/NFE gains do not transfer to two fixed K1 evaluations. [Primary paper, Sections3–5 and Appendix](https://proceedings.neurips.cc/paper_files/paper/2019/file/21be9a4bd4f81549a9d1d241981cec3c-Paper.pdf)

**Initialization and Regularization of Factorized Neural Layers** (Khodak et al., ICLR2021) directly studies matrix-product layers, SVD initialization and their different optimization/regularization behavior. It supplies relevant prior art for factorization and an initialization warning, not evidence for this generative architecture. Current training has weight_decay=0; no new Frobenius penalty is warranted. [Primary paper, Sections2–3](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/03/main_Initialization-Regularization-Factorized-Neural-Layers_final.pdf)

## What the algebra does and does not buy

The candidate contains the narrow q21/g21 model by ignoring extra coordinates, but does not contain the arbitrary full-P parent by this construction. More state coordinates are not more independent value directions. For each rectangular map,

```text
J_F − fixed_skip = D U V + diag(UV input) J_s,
rank(J_F − fixed_skip) ≤ q+g = 42.
```

The middle SiLU and fixed skips prevent treating42 as a whole-FFN/output rank bound. No global approximation separation from SwiGLU follows.

For a frozen square lifted generator, nonzero eigenvalues of `D U V` equal those of `V D U`. With the pruned zero blocks, first-map eigenvalues ignore U1's extra rows; second-map eigenvalues ignore V2's extra columns. Those blocks can nevertheless change hidden activations and returned outputs. Eigenvalues alone therefore miss the proposed mechanism. Factor changes `(U,V)→(UQ,Q^-1V)` preserve the product but change factor norms and optimizer coordinates; monitor actual products/actions, not individual-factor spectra as physical invariants.

## Minimal contrasts and initialization

A useful allocation test needs the candidate, narrow q21/g21 control, **same-budget unaugmented q32/g32 control**, original parent, SiLU128 and existing SwiGLU96/128 frontier. This is a joint allocation comparison, not a factorial that separately identifies rank, controller width and augmentation. Beating the narrow control alone uses1.5× its dense compute. SwiGLU84 matches that narrow control's10,752 MACs if a claim about the narrow model is made; it is unnecessary to make the candidate's main comparison.

Use recorded random references, never teacher weights. A single declared balanced-SVD factorization can bind initialization: rank-q truncation has spectral residual `sigma[q+1]` and squared Frobenius residual equal to discarded singular-value energy. These are matrix bounds, not whole-network error guarantees. Pair candidate/narrow common tensors; make first extra features nonzero, while setting V2's extra columns and controller2's extra input columns to zero. Their initial functions then match. Extra V2 parameter responses can be nonzero under a nonzero receiving head, so parameter tangents are not paired; a native zero velocity head can temporarily mask this. Zeroing both the producer and receiver risks dead features. No exact initialization match to the full-rank parent is possible in general; no outcome-dependent scale selection is acceptable.

**Falsifiable decision:** failure against the same-budget factorized control weakens allocation; failure against SwiGLU rejects practical utility even if the parent improves. If both improve, that establishes a useful budget choice only, not an ODE/topology novelty claim. Prototype approval must not become a rank/width/temperature sweep or reopen stopped families. No implementation, initialization search or training was performed for this review.
