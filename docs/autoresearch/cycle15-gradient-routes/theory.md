# A large output-null loss does not directly push hidden P

## Frozen cotangents, live hidden computation

Let the final raw predictor be m=W[gamma*z(h)+beta]+b. With width w=64,
LayerNorm's normalized vector z is centered: Cz=z, C=I-11^T/w. Define
B=W diag(gamma) C. Let Q contain the two target-plane basis vectors, and let
T be the orthogonal projector onto ker(B^T) intersect ker(Q^T).

For any parameter theta upstream of this final LayerNorm (including all P,
clocks, frames and ordinary FFNs), T*(partial m/partial theta)=0. Indeed,
the derivative of z is centered too, and TB=0. The clean decoded velocity
v=(m-x)/delta has the same annihilation property because x and delta are
independent of hidden parameters. Consequently J_theta(v)^T*T*e=0 for
**any** residual e, including the noisy paired FM training label. No oracle
replacement or frozen hidden clock is required.

This statement holds with W,gamma,beta,b and T fixed at the checkpoint.
Final normalization affine parameters belong to the readout group; their
variations can move the affine output envelope. Differentiating a globally
recomputed projector with respect to the readout would add other terms.
Our diagnostic detaches T and measures cotangent routes at one checkpoint.

Rank(B)<=63, so rank(T)>=63. At zero output initialization B=0 and T has
rank126. Do not impose rank63 on cold checkpoints or divide a zero hidden
gradient by itself.

## Orthogonal errors are not orthogonal parameter gradients

Let A=QQ^T, N=I-A, and R=N-T. These are mutually orthogonal projectors with
A+T+R=I. Split e into e_A,e_T,e_R. For mean squared error with n scalar
outputs, g_j=(2/n)*J^T e_j and g=g_A+g_T+g_R. The loss energies add, but

    ||g||^2 = sum_j ||g_j||^2 + 2*sum_(i<j) <g_i,g_j>.

On hidden parameters g_T=0. Cancellation or reinforcement between g_A and
g_R remains possible. Reporting separate squared norms as causal gradient
fractions would be incorrect. A descriptive ratio
||g_A+g_R||^2/(||g_A||^2+||g_R||^2) includes their cross term.

## Indirect clipping and saved-moment coupling

Native clipping uses c=min(1,1/(||g||+1e-6)). Removing the forced cotangent
g_T leaves the hidden raw gradient unchanged but may change c through the
readout. Cross terms mean c need not move in a predetermined direction.
For plain gradient descent this changes only the hidden step magnitude.

For native AdamW with weight decay zero and saved moments m,v at step k,
consider only a **promoted float64 one-step counterfactual**:

    m+ = beta1*m+(1-beta1)*c*g
    v+ = beta2*v+(1-beta2)*(c*g)^2
    d = lr*[m+/(1-beta1^(k+1))]/[sqrt(v+/(1-beta2^(k+1)))+eps].

Changing c can alter coordinate-wise Adam direction due to existing moments.
This formula is not a replay of the native float32 optimizer kernel, a
counterfactual retraining history, or a proof that clipping caused a quality
gap. At cold initialization there are no historical moments.

## What would change our next action?

Annihilation rules out direct hidden-gradient explanations for this specific
forced component. Measured clipping/update differences establish a local
indirect route only. Neither justifies changing the loss, head or optimizer
on stopped candidates. A useful new ODE operator must still beat ordinary
controls and all existing quality guards on unchanged benchmarks.
