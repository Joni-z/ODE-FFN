# Professor-guided ODEFFN exploration

The working question is whether an input-dependent matrix polynomial can make a
useful FFN for learning flow-matching fields. The present experiments use a small
time-conditioned synthetic velocity network. They establish neither JiT image
quality nor an advantage from ODE terminology alone.

The professor proposed examining Muon and replacing a scalar s(h)P by P D(h).
The completed comparisons now distinguish three things that were initially mixed:
update-spectrum shape, matrix scale, and the function actually computed after
input-dependent gating.

Muon spreads update energy across more directions. In the original Product16
scalar model, the selected higher-rate Muon run has update stable rank32.7 versus
AdamW's3.1. P energy is also less concentrated, but its spectral norm grows19.2
versus3.4. A read-only row/column decomposition shows that most of this observed
growth accompanies increasing row magnitudes, with much less growth of normalized
row coherence. Conditioning does not improve uniformly. Matching ordinary momentum
to the current hypothetical Muon update norm does not reproduce all field gains;
the same direction benefit can also occur in SwiGLU. Tested weight decay reduces
P and local Jacobians but worsens fitting. These results support an optimization
interaction, not an ODE-specific spectral mechanism.

The diagonal question is broader than where to place D. We tested PD,DP and a
balanced sandwich; then held PD fixed while changing nonlinear versus affine
controllers, bounded versus affine outputs, factorized versus full linear logits,
and direct input diagonals. Most changes fail to improve strong controls. Direct
D(h)=diag(h) is the clearest remaining local lead: on Ring8, field MSE0.01210 and
SW2 0.17270 improve on the tested scalar ODE, but remain behind stronger ordinary
SwiGLU results. On Product16 the same construction improves some field comparisons
while worsening sampling. This is not a general architecture win.

The algebra explains why P alone is an incomplete object of study. In the affine-D
family, P'=P C^(-1),D'=C D gives exactly the same PD and polynomial. A trained
checkpoint demonstrates a P singular norm changing1.44 to7.56 while the complete
prediction changes only2.2e-15 in float64. This exact scale freedom generally does
not hold inside the original fixed sigmoid range. It also changes optimization
coordinates: even ideal Muon at the same nominal LR need not produce the same
effective PD step. The relevant diagnostics therefore include PD, controller input
derivatives and the complete predictor, not only raw weights.

A separate one-dimensional example makes the limitation concrete: P=-1 has
benign spectrum and condition number1, yet an allowed sharp bounded gate makes
the K2 input-dependent map's derivative negative. Entrance-frozen exact exponentials
can also fold. This does not establish that folding harms velocity prediction;
it shows why an eigenvalue constraint is not a complete argument for a learned
conditional block.

The raw-input lead computes explicit quadratic/cubic features. Its very large
hidden Jacobians do not translate directly into huge complete-field Jacobians,
and frozen float32/float64 checks do not explain its field gain as a precision
artifact. The registered order comparison removes the tied second-order term,
for both the original scalar and direct-input candidates. A smaller native
SwiGLU64 completes the parameter-budget comparison; SwiGLU128 already nearly
matches the direct K2 dense-MAC budget. All22 trials completed and replayed.
Scalar K1 improves field error in all eight paired settings. Ring8 raw K1 reaches
0.01091, slightly below the strongest ordinary0.01123, but its sampling remains
worse than matched SwiGLU64 (0.17792 versus0.15038); Product16 still loses to
ordinary controls. The quadratic feature interpretation is now more credible
than a benefit from the tied second-order Taylor term in this setting.

The frozen Heun32/64/128 check also completed. Ordering does not change, and the
Product16 raw-K2 sampling gap to SwiGLU persists. Coarse integration is not
supported as the explanation for that frozen discrepancy. Time-binned field
errors describe the remaining mismatch without proving its transport cause.

These completed tests weaken the second-order explanation. The remaining lead
concerns feature construction and optimization, with an unresolved gap between
field fitting and sampling. A matched ordinary FFN remains the necessary control.
No current result justifies image transfer or an FM-specific inductive-bias claim.

Cycle4 completed a smaller K1 feature contrast: constant1, bounded direct sigmoid,
and h+1 diagonals, holding P/common tensors and all training settings fixed.
The constant arm is an ordinary bias-free SiLU FFN under W=I+P; the shifted arm
adds quadratic curvature at the same local value/Jacobian at zero. Afterward,
a frozen parity diagnostic verified a known property of both population targets:
u(-x,t)=-u(x,t). The shift1 arm does not improve the best result; the constant
ordinary FFN remains a useful control. On Product16, excess even energy accounts
for74% of rawK1's paired-error gap to Swi128. This is a diagnostic decomposition,
not evidence that removing it improves sampled quality. A raw-versus-abs diagonal
has exactly identical conditional singular values at matched P,h but even versus
odd increments. That contrast has now completed with a matched ordinary bilinear
FFN and all earlier SwiGLU controls. Abs worsens all6 matched field comparisons
and does not consistently reduce the composed predictor's even error. Stop
this feature setting. No symmetry projection or extra loss was added.

A new ordinary bilinear case has poor field error but low random-projection SW2
on Product16. This does not alone establish either good generation or a broken
metric. A frozen audit now shows a concrete failure: its axis0 gap mass is32.45%
versus the population3.04%, despite favorable aggregate distances. Strong ordinary
models also retain smaller imperfections. Primary scores remain unchanged;
actual marginal examples are required alongside them.

Cycle6 completed all eight80k trajectories with exact archived20k states and
exact generated-array replay. Raw K1 improves matched-Swi64 field MSE by9.4%
on Ring8 and2.9% on Product16, but worsens sampling on both and does not beat
the wider ordinary frontier. The short-budget lead over Ring8 Swi192 disappears.
No asymptotic convergence is asserted.

Cycle7 then separates frozen singular-value flattening from scalar contraction
at the same leading singular value, preserving Frobenius norm on the shape
branch. All54cases and six exact anchors pass. Only a small RawK1/Ring20k shape
change helps slightly; that benefit reverses at80k. Strong flattening always
harms field fitting. Equal matrix energy also need not preserve feature energy:
E||Pq||^2=tr(P^T P E[qq^T]). This closes a post-hoc repair, not the separate
question of adaptation during constrained training.

Cycle8 completed that adaptation question: all12trajectories and native replays
pass; four20k native states/samples exactly match the archives. Shape relaxation
increases P stable rank in all four settings but worsens field MSE by3.5–11.2%.
Scalar contraction also worsens all four. Both modes worsen test SW2. Reject
this setting, without a new strength/rate sweep. Product16 scalar K2 also shows
a useful diagnostic contradiction: lower P norm accompanies a larger first-term
response. The subsequent actual-feature audit tests
that mismatch; its feature-metric derivation is not a new optimizer or quality
result.

Cycle9 completed the feature question rather than introducing whitening. For
q=D(h)h, response energy is tr(P E[qq^T] P^T). At the first Product16 scalar
solve, shape training reduces P's leading singular value19.46->16.13 while
first-term/input RMS rises3.29->4.39. Feature amplitude and alignment account
for the opposite movement. The decomposition is descriptive: it does not show
that this larger response causes the worse field score. The audit also finds
no uniform catastrophic cancellation between K2 terms.

Cycle10 completed one different question: preserve raw quadratic feature
directions, but soften their growth with d(h)=h/sqrt(1+mean(h^2)). At common
P,h, all conditional singular values scale together; spectral shape is unchanged.
The complete input derivative nevertheless changes through a rank-one term.
This is a more specific hypothesis than making every spectrum flatter. A fixed
h/sqrt(2) contrast tests whether simple amplitude/coordinate scaling suffices.
Raw and all three SwiGLU widths remain in the twelve-cell comparison. Radial
reduces mean Ring8 first-term/input RMS from970to30, but improves field error
only3.73% and barely changes sampling. Product16 field worsens9.22%. Neither
task qualifies under the predeclared raw/static5% criterion, so this setting
is closed rather than advanced to80k. Numerical moderation alone is not enough.

Cycle11 then closes the missing rank32-controller/affine-output corner. All four
corners are now present at each of six fixed optimizer/task settings. On
Product16/AdamW, sigmoid-controller linearization barely changes field fitting,
while affine-controller linearization improves it49%. This is a real fixed-budget
interaction, not a general advantage: all six new candidates lose to Swi128
on field and sampling, at almost identical parameter count. The next diagnostic
must distinguish an immediate output-transform effect from the function learned
through training. No new method or longer-run rescue is accepted.

Cycle12 tests an immediate-output explanation at fixed weights: swapping
sigmoid/affine on all24factorial checkpoints. All Product16swaps worsen fitting
and sampling. Only two small Ring8field improvements occur, both below Swi128.
Native arrays and restoration are exact, and all saved-array distances reproduce.
This rules out that frozen repair; it does not establish the full cause of
training differences or promote a new method. Raw P spectra remain identical
while the conditional function changes, reinforcing the need to examine D and
the full predictor rather than raw weights alone.

Primary artifacts: [question-to-decision map](research-map.md),
[logic and derivations](logic-chain.md), [spectral evidence](spectrum/findings.md),
[controller/range results](cycle1-results/findings.md),
[direct-diagonal results](cycle2-results/findings.md),
[order/control results](cycle3-results/findings.md),
[deterministic features](cycle4-results/findings.md),
[abs rejection](cycle5-results/findings.md),
[80k comparison](cycle6-results/findings.md),
[frozen spectral interventions](cycle7-results/findings.md),
[training constraints](cycle8-results/findings.md),
[actual feature responses](cycle9-results/findings.md),
[radial results](cycle10-results/findings.md), [radial derivation](radial-diagonal.md)
and [complete controller/output comparison](cycle11-results/findings.md) and
[frozen output-swap analysis](cycle12-results/analysis-report.md).

The latest [bounded-odd D study](cycle13-results/analysis-report.md) provides a
more competitive candidate: AdamW ISRU-D reaches Ring8 MSE.009773 versus
Swi192.011231. Yet ordinary free ISRU is.009998, and onProduct16 it is better
than ISRU-D (.015356 versus.016059). The structural lead is not established;
the5%free-control continuation threshold fails. The useful distinction is
feature-response benefit versus an ODE mechanism, with an optimizer interaction
that the present spectra describe but do not causally explain.
