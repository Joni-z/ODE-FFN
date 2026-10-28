# What the image comparison can tell us about an ODE layer

The current image result is a disagreement between fitting and generation
metrics, not evidence that a hidden Taylor expansion solves the transport
problem more accurately. The following identities explain why that distinction
matters. They do not change cycle17's failed gates or authorize a cap, solver,
loss or architecture rescue.

## The relevant primary sources

| Paper | Status and inspected material | Relevant point | Limit for this project |
|---|---|---|---|
| Tianhong Li and Kaiming He, [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/html/2511.13720v1), 2025 | Original paper, Sections 3.2/4.3, Table 9 and Appendix B.1 | Separates direct prediction from loss space; explicitly caps division by 1-t at 0.05. | Does not propose a Taylor FFN or prove that higher hidden order improves image generation. |
| Maojiang Su, Jerry Yao-Chieh Hu, Yi-Chen Lee, Ning Zhu, Jui-Hui Chung, Shang Wu, Zhao Song, Minshuo Chen and Han Liu, [High-Order Flow Matching: Unified Framework and Sharp Statistical Rates](https://proceedings.neurips.cc/paper_files/paper/2025/file/4163873c9ad623a87989d0a6eefe9442-Paper-Conference.pdf), NeurIPS 2025 | Pages 1–8, especially Remark 3.1/Eq. 3.3 and Theorems 3.3–3.4 | Higher trajectory derivatives obey total-derivative consistency; conditional regression follows the specified augmented probability path. | Its order concerns sampling-time trajectories, not powers of a hidden P. We have not independently verified all proofs. |

These sources address distinct objects. The first motivates clean prediction;
the second requires consistency of derivatives of a transport flow. Neither
supplies a theorem turning the professor's internal P squared into external
image acceleration. This is a targeted interpretation check, not an exhaustive
new literature review. The retrieved SCORE preprint is not promoted to a
recommendation: its discrete recurrence is another question and supplies no
image evidence for our block. Exact-name searches still do not establish which
historical paper was called “Tied Flow”; no claim relies on that name.

## 1. What the regression loss identifies

Let X be a data image, E independent standard Gaussian noise, and
Y_t=t X+(1-t)E. Write m(y,t)=E[X | Y_t=y]. Let delta=max(1-t,epsilon),
with epsilon=.05. At each fixed t, the population loss satisfies

```text
R(D,t) = E[||D(Y_t,t)-X||^2] / delta^2
       = E[||D(Y_t,t)-m(Y_t,t)||^2] / delta^2
         + E[trace Cov(X | Y_t)] / delta^2.
```

The second term does not depend on D. Thus the unrestricted conditional
minimizer remains m even with the cap. This is a population identity; a finite
held-out batch has cross terms and sampling error. Its pixels or saved
checkpoints are not independent trained repetitions.

Without the cap and for t<1, the interpolation's marginal velocity is

```text
v_star(y,t) = E[X-E | Y_t=y] = (m(y,t)-y)/(1-t).
```

The capped implementation at the same optimum instead uses

```text
v_cap(y,t) = (m(y,t)-y)/delta
           = [(1-t)/delta] v_star(y,t).
```

After t=.95 the multiplier is below one. In general the original probability
path p_t no longer solves the continuity equation for v_cap: substituting it
leaves `(1-r(t))*partial_t p_t`, where r=(1-t)/delta. This is a statement
about the specified path, not a claim that the resulting trajectory can be
replaced by a simple time reparameterization: v_star is still evaluated at
the original t. The cap is a declared numerical convention shared by all arms.
It does not invalidate their fixed-protocol comparison.

## 2. An exact counterexample to a universal loss/FID argument

Consider one normal coordinate whose clean value is always zero. Its noisy
training value is Y_t=(1-t)Z, with Z standard Gaussian. The perfect clean
predictor is D_good=0 and its regression loss is zero. Its capped sampling
ODE is `y'=-y/delta`, initialized with y(0)=Z. Solving the two intervals gives

```text
y(t) = (1-t) Z                         for t <= 1-epsilon
y(t) = epsilon exp(-(t-1+epsilon)/epsilon) Z  afterwards
y(1) = a Z,     a=epsilon/exp(1).
```

For epsilon=.05, a=0.01839397 and the terminal variance is 0.0003383382.
Finer integration approaches this nonzero variance; it cannot remove the cap's
continuous-limit discrepancy to the point target.

Now consider the deliberately worse predictor D_bad(y,t)=-y, solely as an
analytic counterexample. Its clean regression loss is positive, whereas its
capped velocity is `-2y/delta`. Its terminal state is `a^2 Z`.
The squared Wasserstein distance to the zero point is `a^4`,
which is smaller than `a^2`. Better generation distance therefore need
not imply better paired regression under this convention, even in a linear
example with an exactly known continuous solution.

This is **not** an image FID result, a trained-model intervention, a proposed
negative-identity FFN, or an explanation of which current arm learns which
bias. The example rules out a universal inference from one metric to the
other. It does not reverse the registered K2 decision. No cap change is tested.

## 3. Which derivative is actually higher order?

For the complete clean predictor D, external material acceleration is

```text
t < .95:  a = (D_t + J_D v)/delta
t > .95:  a = (D_t + J_D v - v)/delta.
```

The professor's hidden solve computes `F_K(h)=T_K(s(h)P)h`, whose derivative
contains both the matrix polynomial and a clock-gradient outer product.
Its `P^2 h` term is a local linear-generator term. Identifying it with the
external acceleration above would require an additional relation between
hidden coordinates, the full time-conditioned predictor and the readout.
Our experiments have established no such relation. A larger quadratic RMS
or a well-behaved P spectrum does not establish it either.

## 4. Straight conditional paths need not have straight marginal flows

There is a second distinction when borrowing a high-order FM argument.
It is already established in the project's [complete-field derivation](../../research-architecture-search/complete-field-constraints.md); the example below restates it for this image comparison. Let
X and E be independent standard one-dimensional Gaussians. Each conditional
interpolation `Y_t=t X+(1-t)E` is straight and has zero second derivative.
Yet its marginal probability flow has

```text
q(t) = t^2+(1-t)^2
u(y,t) = [(2t-1)/q(t)] y
a(y,t) = partial_t u + (partial_y u) u = y/q(t)^2.
```

At t=.5, the marginal velocity is zero while its material acceleration is
`4y`, generally nonzero. This example is uncapped and does not involve a
trained model. Averaging zero conditional accelerations therefore cannot
be substituted for differentiating the marginal velocity field. The
conditioning state and derivative-consistency assumptions matter.

For this project, neither a conditional straight path nor a local hidden
`P^2 h` establishes the external acceleration needed by a higher-order
transport argument. This is a mathematical distinction, not a new loss
or a proposal to add an acceleration network.

## 5. Decisions this supports

1. Finish the all-model frozen resolution check. A ranking reversal would
   identify a numerical dependence. Persistence would rule out that specific
   32-step explanation on the tested grids, not establish exact-flow quality.
2. Keep fitting quality, finite-solver generation quality and local operator
   activity separate. The present K2 fails its fitting and order gates; the
   original ODE FID observation is still retained with its limits.
3. Before another architecture, require a task-related property of the local
   map that an ordinary matched control lacks and a measurement capable of
   falsifying that property. Existing stopped field, frame, recurrence and
   coefficient families remain stopped. No new training family is selected
   by this note.
