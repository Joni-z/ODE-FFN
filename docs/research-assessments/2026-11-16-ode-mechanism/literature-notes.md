# Primary-source checks for the mechanism plan

Checked16 November2026. This is a targeted methodological review, not an
exhaustive novelty search. No paper's reported performance is substituted
for evidence about our FFN.

## Flow-map structure belongs to a transition map

Boffi, Albergo and Vanden-Eijnden, *How to build a consistency model:
Learning flow maps via self-distillation*, NeurIPS2025.
[Full text](https://arxiv.org/html/2505.18825v2), section2.2,
Proposition2.2 and AppendixE.

The paper characterizes a two-time physical flow map through its tangent
condition and equivalent Lagrangian, Eulerian or composition conditions.
It shows where the relationship to generation is explicit: the modeled
object transports states between physical times. Its composition condition
does not automatically apply to a hidden feature map inside an instantaneous
velocity predictor. Using that property as a diagnostic is sensible; calling
it a new idea or imposing it on arbitrary hidden features without measuring
task benefit is not supported. A transition-map pivot would enter established
consistency/flow-map work and require comparison with it.

## A numerically trained block may depend on its discretization

Ott, Katiyar, Hennig and Tiemann, *ResNet After All? Neural ODEs and Their
Numerical Solution*, arXiv2007.15386v2.
[Full text](https://arxiv.org/html/2007.15386v2), introduction and
step-size/convergence discussion.

The authors observe that switching to a more accurate solver can hurt a
model trained with coarse integration. This motivates testing whether our
state-evolving block's result survives internal solver refinement. Such a
check distinguishes a continuous-flow interpretation from a useful particular
discrete computation. The work does not predict that refreshing our ISRU
response will improve flow matching.

Sander, Ablin and Peyre, *Do Residual Neural Networks discretize Neural
Ordinary Differential Equations?*, NeurIPS2022.
[Proceedings abstract](https://papers.nips.cc/paper_files/paper/2022/hash/ecc38927fe5148c66bee64ee8fed1e76-Abstract-Conference.html).
The theoretical result needs depth smoothness for convergence to a continuous
model. Evidence inspected here is the proceedings abstract, not a full proof
audit. It supports caution about identifying repeated residual computation
with an arbitrary continuous model; the plan does not rely on its theorem.

## Continuous sensitivity versus discrete numerical stability

Soderlind, *The logarithmic norm. History and modern theory*, BIT2006.
[Author institution record](https://portal.research.lu.se/en/publications/the-logarithmic-norm-history-and-modern-theory/).
This is a provenance pointer for the classical concept. The plan supplies
its own variational-equation derivation and checks; no claim of a newly
discovered error-amplification principle is made.

Tao and Choi, *On the Role of Strain and Vorticity in Numerical Integration
Error for Flow Matching*, arXiv2605.06680v1, preliminary preprint.
[Full text](https://arxiv.org/html/2605.06680v1), section3.1 and AppendixA.

This recent paper is directly relevant to velocity-Jacobian diagnostics,
but its displayed proof step `||I+hJ|| <= exp(h*mu_2(J))` is false in general.
For J=[[0,-1],[1,0]], mu_2(J)=0 while ||I+hJ||=sqrt(1+h^2)>1 for h>0.
The continuous exponential bound is valid; the discrete Euler expression
needs higher-order terms. We therefore do not rely on that paper's stated
finite-step bound or transfer it to Heun. This local objection does not audit
all its experiments or exclude an asymptotic bound with corrected terms.

## Research consequence

Established mathematics gives useful identities to test, not a guarantee
that an ODE FFN outperforms SwiGLU. The contribution must identify which
constraint matters in this architecture, predict a change in the actual
learned field, and connect it to samples under controlled interventions.
The exponential-coefficient release and the conditional state-evolution test
are designed for that purpose. They are not claimed as novel numerical methods.
