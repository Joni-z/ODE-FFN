# What the solver-bias result can contribute

Useful coarse-solver bias is established prior work. Our remaining claim needs
an unfitted prediction of its effect on the complete field and target metric.
Six primary records and relevant sections were rechecked17November2026; this is
a focused scope audit, not an exhaustive novelty certification. The registered
full transport cohort keeps running unchanged.

## Verified comparisons

| Primary paper | Inspected evidence | Consequence for this project |
|---|---|---|
| Ott et al., *ResNet After All?* (2020;v2,2023), [Sections2.2/2.4](https://arxiv.org/html/2007.15386v2) | Biased coarse numerical errors can become useful features, including without trajectory crossing. Classification solver substitutions are tested. | Solver dependence and a useful coarse bias are established phenomena. Observing them in our FFN is insufficient by itself for a new mechanism claim. |
| Zhu et al., *On Numerical Integration in Neural ODEs*, ICML2022, [Theorems3.1/3.2](https://proceedings.mlr.press/v162/zhu22f/zhu22f.pdf) | Inverse modified equations describe fitting a known flow map under analytic/boundedness and uniform-neighborhood error assumptions. | Our hidden dynamics have no supervised true feature trajectory. Use forward defect algebra, with the correct sign; do not import the inverse learning guarantee. |
| Tong et al., *Neural ODE Transformers*, ICLR2025, [Sections4.1/6.1](https://arxiv.org/html/2503.01329v2), [official proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/82e3509393670d9d478c359b4ce6f950-Abstract-Conference.html) | Depth-dependent attention/FFN weights and language-model perplexity comparisons, including a shared-weight baseline. | Continuous depth does not require a shared autonomous P. Their language results do not establish a flow-matching FFN benefit. |
| Zhou and Liu, *An Error Analysis of Flow Matching*, ICML2025, [Assumption1.1/Theorems1.6/1.7](https://proceedings.mlr.press/v267/zhou25l.html) | Wasserstein consistency under bounded target support, specified network classes, early stopping and step schedules. | Upper bounds and consistency are not a monotone ranking of finite-checkpoint MSE and SW2. Our Gaussian-mixture targets have unbounded support; these guarantees cannot simply be cited as quantitative guarantees here. |
| Tao and Choi, *Strain and Vorticity…* (2026preprint), [Section3.1/AppendixA.5/Section9](https://arxiv.org/html/2605.06680v1) | Continuous logarithmic-norm reasoning is used to justify a finite Euler norm step; higher-order solvers remain future work. | That displayed finite-step inequality is false. Retain continuous sensitivity analysis, but do not use this proof to justify a Heun bound or a new spectral penalty. |
| Zhang et al., *Contrastive Flow Map Matching*, ICML2026, [Theorem3.1/Section4](https://proceedings.mlr.press/v306/zhang26do.html) | Joint-KL upper bound under absolute-continuity conditions for physical-time intermediate/endpoint distributions, followed by regression/contrastive surrogates. | This is a direct physical-transition formulation. Applying its argument to an arbitrary hidden feature map requires a separate derivation; changing our objective to FMM would be a different study. |

## Two mathematical distinctions that matter

The forward defect of a fixed numerical method satisfies
`method(f)-exact_flow(f)=eta^(p+1)*g+...`; its modified field starts with
`f+eta^p*g`. Zhu et al.'s inverse construction compensates a known target map
and starts with the opposite correction. Their learning theorem uses a
neighborhood-uniform map error, not our measured finite-sample FM field MSE.
Our existing midpoint/RK4 coefficient and sign checks remain necessary.

For the rotation matrix J=[[0,-1],[1,0]], the continuous logarithmic norm is0
and exp(hJ) has norm1. In contrast:

    ||I+hJ||_2 = sqrt(1+h^2)
    ||I+hJ+h^2*J^2/2||_2 = sqrt(1+h^4/4)

Both exceed1 for positive h. This directly contradicts the finite Euler
inequality displayed in the cited preprint and rejects its naive extension to
our Heun sampler. It does not reject the continuous bound or every asymptotic
bound with additional remainder terms. At h=.5, the three norms are
1.118/1.008/1.000 (Euler/Heun/exact). The
[independent numerical check](literature-rotation-counterexample.json) retains
all three evaluated step sizes.

## Research decision

The current experiments answer a discrete-computation question at fixed native
Heun128. Keep that scope. A claim about a continuous generated flow would also
need outer-sampler resolution evidence; hidden-solver accuracy cannot supply it.
No new refinement run is launched by this audit.

Our inference from these sources and the completed controls is that ODE value
must be attached to a useful architectural constraint, rather than accuracy
alone. The next evidence must connect prescribed hidden computation to an
unfitted full-field change, then to the actual target metric. The current
learned-field tangent test validates the latter link before replacing its
measured forcing by the independently derived ODE defect. Ordinary FFN controls
still decide architecture utility. Keep the original primary gate and every
failed numerical or explanatory case.

The six-source [coverage receipt](literature-scope-coverage.json) records scope
and unresolved leads. New PDFs were streamed from publisher-linked URLs to the
existing local text extractor; none was added to the repository or paper library.
Final Reports1/2 are unchanged.
