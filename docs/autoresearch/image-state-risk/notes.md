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
