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

## Additional checks: what autonomy and composition do not guarantee

Tong et al., *Neural ODE Transformers: Analyzing Internal Dynamics and
Adaptive Fine-tuning*, arXiv2503.01329v1 (March2025).
[Primary full text](https://arxiv.org/html/2503.01329v1), sections4--6.
The model generates attention and FFN weights as functions of continuous
depth, deliberately relaxing shared weights. Its spectral observations and
language-model evaluations challenge treating shared-weight dynamics as a
universal description of transformers. This is whole-model language work,
not a matched flow-matching FFN comparison. Consequence for our study:
shared P is a testable restriction; "more autonomous" alone does not predict
better generation. Our stationary-versus-stage controls must establish that
benefit on their own tasks.

He, *Neural Vector Field Transformations: Learning the Flow of Autonomous
Systems by Conjugation to Explicitly Integrable Reference Flows*,
arXiv2610.29039v1 (21October2026), preliminary preprint.
[Primary record](https://arxiv.org/abs/2610.29039), abstract.
The architecture conjugates an explicit reference flow through an invertible
network, obtaining a flow for every parameter value. Conjugacy also preserves
dynamical invariants, so the reference family restricts the learned dynamics.
The abstract reports a setting where a time-input network is more accurate
inside its training horizon. Scope here is the abstract, not a theorem audit
or generative-image replication. This is an example of a precise structural
guarantee with an expressivity tradeoff, rather than evidence that a valid
flow necessarily fits a task better. We do not add a conjugacy architecture
to the current experiment family.

Bajars, *Locally-symplectic neural networks for learning volume-preserving
dynamics*, arXiv2109.09151v2 / Journal of Computational Physics2023.
[Primary full text](https://arxiv.org/html/2109.09151v2), sections2.4/3.3.
The symmetric construction composes learned modules with their adjoints and
shares parameters, targeting time reversal and volume-preserving dynamics.
Thus symmetric learned compositions are established prior work. A possible
future palindrome-versus-alternating schedule would be a controlled mechanism
test, not a new numerical principle. Their volume-preserving assumptions do
not hold automatically for our feature transformations or FM target flow;
no volume constraint is imported from this paper.

## Research consequence

Established mathematics gives useful identities to test, not a guarantee
that an ODE FFN outperforms SwiGLU. The contribution must identify which
constraint matters in this architecture, predict a change in the actual
learned field, and connect it to samples under controlled interventions.
The exponential-coefficient release and the conditional state-evolution test
are designed for that purpose. They are not claimed as novel numerical methods.
