# Evidence and boundaries

ER-20261029-image-mismatch-01: the sealed image cold-control analysis records
native K1 weighted MSE0.160267/FID173.44, Swi86 .159944/195.31 and fixed06
.160392/174.71. Source: ../image-clock-training/analysis-report.md. This
supports a mismatch in measured regression and generation ordering, not a
cause or a new ODE-specific gain. The hypothesis is exploratory.

ER-20261029-empirical-separation-01: Ye et al., authors' arXiv2511.03202v2,
Sections3–4, Proposition4.1. https://arxiv.org/html/2511.03202v2 . The paper
distinguishes the finite-data score minimizing empirical denoising risk from
the population score; the gap is a Fisher divergence under its specified law.
It blocks silently treating an empirical-bank reference as the population
target. Its OU process and data regularity differ from our linear interpolant;
we derive the latter directly. It neither diagnoses our trained models nor
justifies importing its pruning strategy. OpenReview final-PDF review was
blocked by a browser challenge; final-publication listing was verified on
the author's publication page, https://minshuochen.github.io/publications .

The exact Gaussian-conditioning expression here is a direct finite-sum
Bayes calculation, not new mathematics or a Gaussian-reference rescue. Raw
image corruption satisfies the observation law; normalized hidden tokens do
not automatically do so. No learned mixture is fitted or sampled.

ER-20261029-state-risk-result-01: complete frozen cohort and independent audit
sealedf42623697307ebda1d45751e873db30784bfde0307b4c0cfa55df07515023861.
K1/Swi86 equal-time bank-reference ratios are.988757paired-validation,
.996893generated-K1,.977018generated-Swi86; no registered reversal. The
reference has11.33/11.74times K1/Swi86 actual-image error at115/128 and fails
the use guard at4/5times. Native predictions, reference arrays and16,000
per-image quantities replay. These are exact empirical-reference diagnostics,
not population-risk estimates or an explanation of neural FID.

ER-20261029-balanced-representations-01: Zhang et al., ICLR2026, authors'
arXiv2512.20963v2, Sections2.2–3.2, Theorem3.1/Corollary3.3 read at
https://arxiv.org/html/2512.20963v2 ; final venue verified at
https://proceedings.iclr.cc/paper_files/paper/2026/hash/a5e3cf29c269b041ccd644b6beaf5c42-Abstract-Conference.html .
Their theory studies local minimizers of a fixed-noise, regularized two-layer
ReLU DAE under separated-mixture assumptions; balanced refers to data-dependent
representations, not equal singular values of an arbitrary hidden P. It supplies
a useful distinction but no transferable JiT constraint or selected method.
The covariance/projected-coordinate identity in next-question.md is separately
derived; no theorem of theirs is applied to our transformer.
