# Does an ODE constraint improve the learned generative field?

Current disposition: the coefficient/state/sharing/schedule controls and
full refined transport calibration are complete. The general first-order
endpoint explanation is not supported at the trained scale.
[Completed decision and next bounded diagnosis](target-geometry/decision-after-calibration.md).
The original plan below is retained; do not relaunch its completed controls.

16 November 2026. Research plan, not a training submission.

**Priority: identify the contribution of the exponential constraint before
another response-only CelebA run.** Hold ISRU fixed for the mechanism study;
its selection here is an anchor to existing evidence, not a claim that it
is the best response. Arctan remains a later transfer candidate.

## Question, hypothesis and alternatives

Main question: does the exponential's prescribed composition improve
generalization or error propagation of a learned flow-matching field, beyond
the shared matrix operations and nonlinear response already present?

**H1, current architecture:** at matched initial function and dense compute,
fixing the exponential coefficients produces a better trained field and
generated distribution than letting those coefficients adapt. Proposed
mechanism: the constraint changes the placement/direction of field errors
so less harmful error reaches the terminal distribution.

**H2, conditional follow-up:** if the current input-conditioned exponential
does not provide useful flow structure, updating the response along a true
hidden ODE trajectory may help. To support a continuous-dynamics explanation,
any gain must survive convergence toward its actual ODE solution and appear
in the complete generative field and samples.

Alternative explanations include generic shared polynomial features,
optimization changes, state feedback, and numerical-scheme-specific effects.
Merely satisfying a flow identity is not evidence of task usefulness.

## Why the existing experiments have not answered this

Existing K8 checkpoints use their high-order terms and already approximate
their specified conditional exponentials closely on inspected states.
Removing terms after training damages them; independently trained K1 often
matches or improves K8. This establishes use of the supplied computation,
not necessity or value of exponential coefficients.

The latest response screen changes q. It cannot attribute a response gain to
ODE structure. Earlier state-clock midpoint and cubic-flow trials also
failed useful controls; refreshing a state is not an untested universal fix.
See [earlier term audit](../2026-11-15-order-response/findings.md),
[completed screen](../2026-11-15-response-screen/analysis-report.md), and
[earlier feedback discussion](../2026-11-14-next-architecture/recommendation.md).

## Experiment A: release only the exponential coefficients

Use the existing two-solve ISRU FFN with entrance-frozen A=P D(h), original
projections and middle SiLU. Evaluate each Taylor term by the native recurrence.

| Arm | Local map | Purpose |
|---|---|---|
| Fixed exponential K8 | h+Ah+A^2h/2!+...+A^8h/8! | Existing structural constraint |
| Free polynomial K8 | h+Ah+sum_(j=2)^8 alpha_j A^j h/j! | Release only coefficient tying to the exponential |
| Existing K1 | h+Ah | First-order endpoint reference |
| Ordinary SwiGLU and Free ISRU | Existing native definitions | Task utility and non-ODE nonlinear control |

Initialize every alpha_j to1. Shared tensors, hidden outputs, full outputs,
input gradients and gradients to original parameters must match the fixed
arm at initialization. Keep the identity and first-order coefficients fixed
to remove the simplest generator/step-scale ambiguity. There are seven
additional scalars per local solve,42 per width64/depth3 model. Each K8 arm
still uses eight P applications per solve. Share the response, seed0, native
20k-update budget, batch512, AdamW0.0003, no weight decay and grad clip1.
Do not introduce an alpha regularizer or an optimizer sweep.

The free arm contains the fixed arm as a function class. If fixed wins
held-out evaluation despite free matching or improving training fit, that is
a candidate inductive-bias result. If free has worse training fit too, first
report optimization failure; it does not establish an essential ODE benefit.
If free matches/wins while its feature-weighted coefficient effects depart
from the exponential, H1 is weakened or rejected for these tasks and budget.
Alpha remaining near1 is inconclusive without nontrivial gradients and
feature-weighted changes: raw coefficients alone are not a mechanism metric.

Measure coefficient contributions on fixed probes, not just raw alpha.
Fixed-checkpoint replacement alpha->1 is a sensitivity diagnostic; its loss
increase alone cannot prove which architecture would train better.

**Cost: two new trajectories**, one per task. Existing fixed K1/K8 and
ordinary CPU checkpoints can be reused if the new adapter with alpha=1
replays the frozen source, data and initial tensors exactly. Otherwise stop
and register any required replay before training; do not silently mix sources.

## Experiment B: conditional test of actual state evolution

Before training, inspect existing K8 layer entrances at physical times
0.100,0.300,0.500,0.700,0.900. Separate exact-exponential truncation error from
the restart/composition defect caused by recomputing D. Compare the induced
full-model prediction change with field error and numerical repeatability.
If the defect is negligible on trained states and no complete-field effect
is resolved, B is not prioritized merely because its equation looks cleaner.

If that diagnostic supports testing feedback, add one architecture:
`dy/ds=P ISRU(y)`, integrated by four explicit-midpoint steps (eight P
applications per local solve). It has the fixed arm's trainable tensors and
dense matrix work; pointwise work differs and is timed. Initial parameters
match, but its initial hidden function generally differs. Consequently this
comparison is not as clean an attribution test as A.

**At most two further trajectories**, Ring8 and Product16 under the same
20k recipe. At frozen checkpoints evaluate4/8/16 midpoint substeps on the
same internal interval[0,1] and an adaptive high-accuracy reference on a
bounded probe subset. Also test one-versus-two-segment composition at fixed
resolution. If a quality gain disappears on refinement, describe it as a
discrete-computation result. If flow identities hold but sample quality does
not improve, that is a correctly solved but unhelpful hidden ODE.

Do not compare K1 with K8 as the sole test here. Do not label repeated Euler
steps as an ODE-free control: they are also an ODE discretization. Shared
versus unshared full matrices are not in this minimal study; they would
change parameter budgets substantially. A result here cannot isolate the
benefit of weight sharing itself.

## Measurements linking either intervention to flow matching

Both existing datasets have exact marginal velocities. Use common oracle
trajectories and compare the complete v_theta(x,t), including every gate and
projection. Under the unchanged synthetic trainer this is direct velocity;
under JiT any later diagnostic must use its original image-to-velocity
conversion, not just the image-output Jacobian.

1. **Field errors:** ordinary oracle MSE and errors along common oracle
   trajectories, split into five equal physical-time intervals. Record
   training-fit curves too. No arbitrary assertion that smaller ||P|| is better.
2. **Transport of errors:** compute the oracle variational equation and its
   directional integral of model error, as derived in
   [mathematical-basis.md](mathematical-basis.md). Check the first-order
   approximation with mixtures v*+lambda(v_theta-v*) at lambda0.125/0.250/0.500,
   then quantify its residual at1.000. A poor approximation cannot explain
   the final model by extrapolation.
3. **Intervene:** on frozen weights replace the learned velocity with the
   oracle in each of the five time intervals, one interval at a time, and
   compare paired terminal displacement and mode allocation. Integrate
   separately at interval boundaries. Rank predictions before these
   interventions. Agreement supports the error-location explanation;
   disagreement refutes that proposed mechanism, even if one architecture wins.
4. **Samples and cost:** Heun128 pooled SW2 from eight common4096 banks,
   Ring8 mode balance, Product16 gap mass and measured training/sample time.
   Use Heun64/256/512 on a fixed subset to separate discretization error from
   learned-field error; record actual function evaluations, not just step labels.

These observations test how architectural changes affect the actual field.
A lower hidden semigroup defect by itself is never the promotion criterion.
Nor does lower endpoint error on a particular coupling automatically imply
better SW2; both must be observed. Correlation between Jacobian statistics and
quality alone is insufficient for a mechanism claim.

## Evaluation and decisions fixed before new training

Keep training seed0. Generate new diagnostic and confirmation banks before
training, with recorded seeds, hashes and oracle inputs. Previously opened
screen banks are development evidence. Reevaluate reused controls on the new
banks too. Use20k as the primary endpoint;15k is the specified trajectory
check, not an opportunity to choose the best checkpoint after seeing scores.

Report all pooled scores to three decimals; retain full precision in CSV/JSON.
Use paired bank differences with conditional intervals, which describe
evaluation variation of these weights. No training-repeat uncertainty is
inferred from the banks. Define the numerical resolution for each metric by
the fixed solver-refinement check; improvements below that resolution remain
undecided. A predeclared conservative advancement rule is lower pooled SW2
beyond that resolution on both tasks, paired bank evidence in the same
direction, and no resolved mode/gap regression. This is an engineering
decision rule, not a significance theorem or an arbitrary5% gain requirement.

| Outcome | Interpretation | Decision |
|---|---|---|
| Free coefficients match or improve the fixed exponential | Factorial tying has no demonstrated benefit here | Stop promoting exponential accuracy as the explanation |
| Fixed exponential improves held-out transport and samples with comparable training fit | Candidate benefit of this coefficient constraint | Check the proposed error-transport mechanism; then consider image transfer |
| State-evolving model wins but changes materially under refinement | Learned discretization/feedback may help | Do not claim a continuous-flow advantage |
| State-evolving model retains its gain under refinement and improves measured transport and samples | Candidate useful hidden dynamics | Compare with ordinary baselines and measured cost before CelebA |
| Both constrained designs lose to ordinary controls | Current hidden-ODE route is unhelpful at this budget | End this architecture family rather than add responses, order or penalties |

For image promotion, a within-ODE win is insufficient: ordinary SwiGLU and
parameter-matched Free ISRU remain practical references. Show the quality/time
frontier as well as equal-update scores. No automatic CelebA job follows this plan.

## What could become a research contribution

The possible contribution is an identified architectural constraint, a
specific prediction about transport of field error, and interventions that
support that prediction across tasks and survive a qualified image test.
Matrix exponentials, state-evolving neural ODEs, midpoint solvers and flow
composition identities already exist. The plan does not certify novelty.

If hidden dynamics have no measured task benefit, a more direct structural
connection would require learning physical-time transition maps, where
composition is an actual target property. That would change the output and
objective and must be compared with existing flow-map/consistency methods;
it is a separate research decision, not a renamed FFN modification.

Immediate next action after this plan: run the bounded frozen-checkpoint
diagnosis and the two coefficient-control trajectories. Decide on B from
those results. No response-family search is included.
