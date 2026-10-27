# A conditional frame changes directions while preserving the spectrum

At one layer input x, let g(x)=A_r SiLU(W_g x+b_g)+b_r, using the existing controller features, and u=(1,g). Define H(u)=I-2uu^T/(u^T u), H0=H(e1), Q(x)=H(u)H0. The denominator is at least1, so no epsilon or ridge parameter is introduced. Q is orthogonal and equals I at A_r=b_r=0. The unchanged clock s(x) is computed from the original unrotated input.

With M_s=I+sP+s²P²/2, the candidate is

```text
F(x) = Q(x) M_s Q(x)^T x
P_x = Q(x) P Q(x)^T.
```

P_x has exactly P's eigenvalues and singular values, even for a nonsymmetric/non-normal P. Its polynomial and exact conditional exponential are conjugated by the same Q. Thus this changes the directions of the frozen conditional linear ODE, not its spectrum. Q and s are selected once at x; re-evaluating them during an inner solve would define a different equation and is not selected.

For a perturbation dx, write Omega=(dQ)Q^T, which is skew-symmetric. Let A=Q M_s Q^T. The complete differential is

```text
dF = A dx + [Omega,A] x + Q(P+sP²)Q^T x ds.
```

Both dQ and ds depend on dx. Omitting them gives a conditional linear generator, not the full layer Jacobian. When A is symmetric, [Omega,A] is symmetric: rotating response axes can therefore change symmetric stretching, despite Omega itself being skew. This is relevant to cycle10 but does not establish a performance benefit or constrain the final decoded field to be symmetric.

At identity initialization, a frame perturbation dg gives Omega=2(dg e1^T-e1 dg^T). For a diagonal P with distinct polynomial values m_j and x=e1, the frame-coordinate responses are `2(m_1-m_j)e_j`, j>1. They span d-1 directions, while the scalar-clock response is parallel to e1. This is a concrete **local controller-response** witness with P and the other weights held fixed. Changes in P and the complete backbone can compensate; it is not a strict full-network function-class separation or a prediction of training gain.

Q-I has rank at most2, so `P_x-P=(Q-I)P Q^T+P(Q^T-I)` has rank at most4. The candidate is exactly an input-conditioned low-rank operator update. An ordinary gated network can implement the same algebra. A global Q can be absorbed into the freely learned P with no function-class enlargement. These facts require both the global-frame parameterization control and an ordinary K1/control frontier.

At zero frame output, the old K2 forward arithmetic and all inherited parameter/input derivatives are preserved, including with an active nonzero receiving head. Frame parameters supply additional tangents; complete kernels are not claimed to match. No RNG is consumed while adding zero parameters. K1 has different hidden features/output-head tangents from K2 and is explicitly not an initialization-kernel match.

The conditional polynomial norm is unchanged under conjugation, but dQ and ds can increase the whole-map derivative. No full-map Lipschitz, contraction, invertibility or long-depth stability guarantee follows. Learned P also differs between trained models, so preserving spectra across inputs within one frozen layer does not isolate a training gain solely to orientation.

## Primary literature and novelty boundary

[HOFT (2025), Section3](https://arxiv.org/html/2505.16531v2) uses Householder-based orthogonal factors to adapt weights; its scope is foundation-model fine-tuning. It supplies prior art for efficient orthogonal weight adaptation and paired-reflector identity initialization. It does not establish that a conditional conjugation improves flow matching.

[DeltaProduct, NeurIPS2025](https://papers.nips.cc/paper_files/paper/2025/hash/e1ea2520fbf9cd1600b287dde67e0a3c-Abstract-Conference.html) uses products of generalized Householder transitions for linear recurrent state tracking and language modeling. Input-conditioned reflection transitions are established; its automata/sequence evidence cannot be transferred to this FFN comparison.

[HouseGNN (2026 preprint), SectionIV](https://arxiv.org/html/2608.12514v1) conditions node-wise reflections on neighborhood messages. This is particularly close prior art for data-dependent reflection directions, but it updates graph features and does not conjugate the professor's conditional polynomial. Its norm-preservation argument is not a global derivative guarantee for our state-dependent frame.

[ODEtoODE, NeurIPS2020](https://research.google/pubs/an-ode-to-an-ode/) evolves Neural-ODE parameters on the orthogonal group, including isospectral flows. Orthogonal/isospectral parameter dynamics are therefore not new. It is retained as historical prior art, not a frontier performance recommendation. This bounded search does not certify exhaustive novelty.
