# Conditional state expansion under the original FFN budget

**Conditional GO for one bounded prototype, not formal training or an accepted architecture.** This reallocates projection/controller computation to a larger nonlinear state while retaining the original two conditional K1 solves. It changes rank, controller width and state width together; it is a controlled budget-allocation hypothesis, not a pure augmentation experiment. No topology, expressivity or FM-specific superiority theorem follows.

The completed finite collapse shows that the original clocks materially help its frozen function: removing their variation increases Ring8 field error about20-fold and Product16 about2.82-fold. That is not a retrained-static-model comparison. Conversely, existing SiLU128 improves Product16 but worsens Ring8 relative to the parent. Together these findings justify considering how to retain modulation while providing more nonlinear features; they do not predict success. The failed clock, sharing, diagonal-flow, state-clock and bias settings remain closed.

## One precise construction

Let d=64, r=128, q=g=21. Let E pad d coordinates with zeros and R select the first d coordinates. The proposed FFN is

```text
z = E h + D1(h) U1 V1 h       U1:r×q, V1:q×d
y = SiLU(z)                   controller1:d→g→r
out = R y + D2(y) U2 V2 y     U2:d×q, V2:q×r
                              controller2:r→g→d.
```

Both clocks retain `.2+.8 sigmoid(gate/10)`, independent controllers and the original middle activation. There is no learned encoder/decoder, forcing vector, new activation or loss. Constant clocks give a structured expanded SiLU FFN, `(R+cU2V2) SiLU((E+cU1V1)h)`, not an unrestricted dense SiLU128 layer.

The exact square-state interpretation is

```text
P1 = U1 [V1,0]             r×r
P2 = [U2;0] V2             r×r.
```

Pad the first input, apply two conditional K1 maps with middle SiLU, then select the first d outputs. The omitted first-map input columns multiply zero; the omitted second-map output rows are discarded. They are structurally unused in this K1 composition and should neither be allocated nor counted. That argument does not license another order: such columns/rows can participate in intermediate higher-order products of an unrestricted augmented model.

## Derivatives and spectral limits

Define `w1=U1V1h`, `w2=U2V2y`. Complete rectangular input derivatives are

```text
J1 = E + D1 U1 V1 + diag(w1) J_s1
J2 = R + D2 U2 V2 + diag(w2) J_s2
J_FFN = J2 diag(SiLU'(z)) J1.
```

The gate derivatives must remain live. Each controller input Jacobian has rank at most g, so each rectangular update satisfies `rank(Ji-skip_i)<=q+g=42`. The skips, middle activation and surrounding network prevent interpreting42 as the rank of the full FFN or complete velocity Jacobian. More state coordinates do not mean128 independent input/output directions.

For one map with fixed incoming input x, receiving cotangent e and controller s, write `k=Vx`, `w=Uk`. Then

```text
grad_U L = (s*e) k^T
grad_V L = U^T(s*e) x^T
controller cotangent = e*w.
```

The latter passes through the original sigmoid/MLP derivatives. These identities, combined with the skips and middle SiLU, give independent numerical checks of all factor/controller gradients.

For a **frozen** square generator G=DUV,

```text
G^k = DU (VDU)^(k-1) V, k>=1.
```

Its nonzero eigenvalues, including algebraic multiplicity, are those of VDU; the two matrices may have different additional zeros. In the pruned first map the reduced matrix is `V1 D1_top U1_top`, independent of U1's extra rows at fixed D. In the second it is `V2_left D2 U2`, independent of V2's extra columns. Those extra blocks can nevertheless change activations and returned outputs. They are invisible to these eigenvalues, not to operator norms or complete derivatives. No spectral-stability claim should be based on their omission.

ANODE's non-crossing/topology result concerns uniquely solvable continuous flows. Here clocks are frozen from each solve input, K1 can already be singular, SiLU is not globally injective, and the output is projected. That theorem is not this design's motivation; see the [prior-art review](augmented-state-prior-review.md).

## Budget and necessary controls

For the pruned family the dense MAC count is `2(d+r)(q+g)` and parameters add `2g+d+r` controller biases. Choosing half the dense budget for factors and half for controllers gives the single rule `q=g=floor(16384/[4(d+r)])`; for r128 it yields21. This rule is fixed, not a rank or width sweep.

| FFN | Dense MACs | Parameters |
|---|---:|---:|
| Original dense P64, controller32 | 16,384 | 16,576 |
| Unexpanded factorized r64,q=g32 | 16,384 | 16,576 |
| Unexpanded factorized r64,q=g21 | 10,752 | 10,922 |
| Expanded r128,q=g21 | 16,128 | 16,362 |
| Ordinary SiLU128 | 16,384 | 16,576 |
| SwiGLU128, actual85 channels | 16,320 | 16,554 |

The expanded Product16 model would total83,022 parameters/81,216 dense MACs. Counts are derived before implementation and require executed-operation verification. Factoring raises the number of linear calls from six to eight per FFN; larger SiLU states and pointwise gate work also cost time. Equal MACs do not establish comparable latency.

The same-budget factorized32 control is essential. Beating narrow21 alone uses1.5 times its dense computation. Beating the dense parent alone could reflect factorization or its initialization. The six-arm PhaseA proposal is exactly the table above on the existing three tasks, with the original data, objective, time sampling, optimizer, head, seed0 and budget. Native generic initial functions are not paired with the ODE family.

## Initialization and what is actually paired

Construct the same native64-wide parent reference first, preserving all reference RNG draws. For each original P use one declared rank-q balanced SVD:

```text
P=Q Sigma W^T
U=Q[:,:q] sqrt(Sigma[:q])
V=sqrt(Sigma[:q]) W[:,:q]^T.
```

The spectral truncation error is the next singular value; squared Frobenius error is the discarded singular-value energy. Those are matrix errors, not assurances of a small model error. Factorized32 and narrow21 generally change the parent's initial function. Specify SVD precision/casting once before evaluation; no fitted scaling or oracle weights are allowed.

Take the first g rows/biases of each native32-hidden controller's first affine map and retain constant-paired zero final controller affine maps. For expanded21, copy narrow21 into all common factor/controller coordinates. Set

```text
U1_extra = independent copy of U1_top
V2_extra = 0
controller2_first_weight_extra_columns = 0.
```

The first extra state becomes `SiLU(c U1_top V1 h)`, while the first d coordinates are `SiLU(h+c U1_top V1 h)`. Extra features need not equal the old features. They are initially ignored by the second receiving projection and controller, so the **complete FFN function matches narrow21 exactly**. The nonzero extra producer is essential: zeroing both producer and receiver would make that branch structurally dead at initialization.

Balanced SVD describes the common narrow factors. After copying the first producer's extra rows, its full expanded Gram is `U1^T U1=2*Sigma_q`, whereas `V1 V1^T=Sigma_q`; the expanded pair is not globally balanced. Do not silently rescale it to claim balance, because that would change the declared common factor coordinates and their initial parameter responses.

At the paired start, common parameter responses agree. The new `V2_extra` differential is

```text
delta_out = c U2 delta_V2_extra y_extra.
```

It is generically nonzero under a nonzero receiving head. U1_extra and the other isolated extra paths initially have zero responses until the receiver awakens; they are not permanently unused parameters. Numerical checks must expose the live receiver on an explicit nonzero-head example.

At such a head the expanded model adds parameter-response columns; equal initial functions do not imply equal tangent kernels or guarantee independent new directions. **At the actual zero final velocity head**, however, all backbone responses vanish and the shared output-head responses agree, so narrow/expanded initial complete NTKs can coincide. Both statements are needed; neither may be substituted for the other.

Balanced factors do not remove optimization changes. Under ordinary gradient flow for a matrix loss, `P=UV` induces `Pdot=-G V^T V-U U^T G`, where `G=grad_P L`; the factor gauge `(U,V)→(US,S^-1V)` changes these coordinate metrics without changing P. Current Adam/clipping adds further differences. No new factor regularizer is selected, and raw factor norms must not be presented as invariant generator spectra.

## A bounded decision, with credit for useful controls

Before training, PhaseA needs one shared practical screen: compare every potentially useful new arm against the parent and the existing generic frontier, preserving all task/Jacobian/SW2 guards. The established10% nonlinear primary improvement,5% other-primary and10% nonlinear Jacobian/SW2 margins are a suitable quality-lead rule. A claimed augmentation effect additionally needs a material advantage over both factorized32 and paired narrow21 on its qualifying task. Otherwise credit a successful factorized or narrow control under its actual mechanism, rather than discarding it because augmentation lost. A cheaper quality-preserving lead requires its own explicitly registered cost/quality criterion; do not invent that interpretation after outcomes.

Only a practical lead warrants PhaseB. For an augmentation lead, fixed clocks with q21 test the necessity of conditional modulation at lower compute; fixed clocks with q42 reallocate the saved gate budget into factors at the same16,128 MACs. Their budgets are8,064 and16,128 respectively. SwiGLU84 has10,752 MACs and matches **narrow21**, not fixed augmented21; retain that distinction. A winning unexpanded control needs the corresponding controls for its own structure, not automatic attribution to the augmented design. Independent holdout, transfer and isolated timing remain necessary after any screen.

Failure closes this fixed allocation: no rank/width/gate-hidden sweep, temperature, order, forcing, initializer, head or objective rescue. A successful prototype establishes correct algebra and fair costs, not practical usefulness. A later gain would establish a useful conditional factorized allocation; it would not by itself establish ODE novelty, topology benefits or a solution to the architecture goal.
