# Cycle1 registration: what generates D(h)?

Freeze A(h)=P D(h), F(h)=h+A(h)h+A(h)^2h/2. D is evaluated once at entrance;
training and full layer Jacobians retain its input derivatives. Backbone width64,
depth3, two solves with SiLU between, cached bank/data stream, seed0,20k updates,
batch512, independent linear FM objective, clipping1 and Heun32/SW2 remain unchanged.

Native parent: g(h)=W2 SiLU(W1 h+b1)+b2; d=.2+.8 sigmoid(g/10).
A. Affine controller, same bounded transform: replace g by Lh+b, initialized to
match the parent's logit value and input Jacobian at h=0. This tests the controller's
nonlinear features, not its scalar range. Vector controller parameters4160 versus
parent4192 per solve (32 fewer); record rather than claim exact parameter matching.
B. Same MLP and tensors, affine output: d=.6+.02 g. At zero logit it matches the
parent's value and slope, but differs away from zero; signed/unbounded rates are
allowed. This tests the output transform, not a new controller width. Do not also
change the optimizer or range within an arm.

Each family crosses selected-P AdamW.0003 and official Muon.006/.02; auxiliary
AdamW.0003/decay0 fixed. Run on Ring8/Lenovo and Product16/145 (12 new runs).
Each host first reruns three anchors: PD/AdamW.0003, scalar/Muon.02, SwiGLU128/AdamW.0003
(6 technical controls, not new statistical replicates). Retain every earlier rate
and both ordinary widths. New candidates compare with the same-optimizer PD parent.
No chosen-rate rescue follows a bad endpoint. Stop nonfinite failures and retain logs.

Mathematics: for F1=h+P(d(h)*h), J_F1=I+P[diag(d)+diag(h)J_d].
For K2, with q=A(h)h, dF[h][v]=v+A v+(dA[v])h
 +.5[(dA[v])q+A((dA[v])h+A v)], where dA[v]=P diag(J_d v).
An affine controller makes the logit Jacobian constant; it does not make J_d constant
under sigmoid. Affine output removes sigmoid saturation but retains MLP derivatives.
Removing the bound also removes any uniform bound on D. None guarantees accuracy.

Record P and actual-update spectra as before, plus A(h)=P D(h) spectra on two
actual hidden inputs per solve, gate signs/range, full local Jacobians and term sizes.
These conditional spectra are local probes, not global stability certificates.
If useful directions appear, isolate their cause before combining. If both fail,
consider direct input-derived D as a distinct construction, then the already planned
optimizer-by-order comparison. No family is accepted merely for increasing rank.

## Interpretation refinement before outcomes

Although the affine controller has almost the same parameter count, its64x64
matrix can become full rank. The parent logit Jacobian W2 diag(SiLU') W1 has
rank at most32 at every input. The affine initialization matches that rank32
tangent, but later training can increase its rank. Therefore this is a controller
budget/parameterization comparison, not an isolated proof that removing SiLU helps.
A successful affine arm requires a factorized-linear control with the same32-wide
bottleneck before attributing the gain to nonlinearity versus rank.

An exact local matching construction for that prospective control is
W1'=W1,b1'=b1,W2'=W2 diag(SiLU'(b1)),
b2'=b2+W2[SiLU(b1)-SiLU'(b1)*b1], with hidden activation replaced by identity.
This preserves the logit value and Jacobian at h=0, parameter count and rank cap;
it does not match the parent at all h. Keep all current results regardless.
