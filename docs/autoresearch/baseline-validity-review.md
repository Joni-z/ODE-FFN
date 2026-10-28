# The image screen does not yet support architecture conclusions

October 29, 2026. This assessment supersedes performance interpretations of
the current small CIFAR-10 screen. It preserves the recorded numbers and all
sealed artifacts. No new training was launched for this review.

The ordinary baseline was not first shown to be a competent generator or to
have converged under a defensible recipe. Consequently, FID differences such
as 172.04 versus 183.57 do not establish the intended ODE architecture claim.
A predeclared 5% margin and exact numerical replay do not repair that missing
premise. Earlier decisions to prioritize mechanisms based on these image
rankings were too strong.

## What the actual records show

- The model has three transformer blocks, width 64, 4x4 patches on 32x32
  images and about 211k parameters. Its capacity was chosen for a cheap screen;
  it was not validated as a sufficient image-generation baseline. See the
  [bridge configuration](cycle16-image-bridge/feasibility-report.md).
- Training uses 20,000 updates at batch 64, a constant 1e-4 learning rate and
  the recorded native recipe. This is 1.28 million sampled training examples,
  equivalent to 28.44 times the 45k training-set size. A fixed update budget
  is not a convergence criterion. See the
  [training protocol](image-budget-continuation/protocol.md).
- Regression is still improving over the final measured interval. From 15k
  to 20k, native K1 weighted MSE falls from 0.162137 to 0.160267 and Swi129
  from 0.159583 to 0.157935. FID worsens from 10k to 20k across all seven
  original arms. The generated examples remain coarse. These observations
  do not separate inadequate capacity, optimization, objective/sampler
  mismatch, or other pipeline problems, and do not establish convergence.
  See the [curves and cases](image-budget-continuation/analysis-report.md).
- FID uses 1,024 generated and 2,048 validation-reference images. No ordinary
  baseline reproduction with an independent competence/quality target was
  required before using this evaluator to select architecture directions.
  The [latest comparison](cold-clock-source/results/analysis-report.md) is
  therefore a fixed-budget observation in this screen, not an established
  comparison of adequately trained generators.

A high FID alone does not diagnose non-convergence: a converged but inadequate
model or a flawed recipe can also have poor FID. Merely extending training
would therefore be an unsupported remedy. The missing work is baseline and
pipeline validation, not choosing a more favorable checkpoint.

## What survives this correction

Algebraic identities, local derivative checks, actual parameter counts and
measured matrix work retain their stated scope. For example, a fixed local
clock allows the professor's core to be written with ordinary matrices
`W=I+cP`. This identity does not imply an advantage in generation. Likewise,
ODE notation or a correct Taylor expression does not prove that the clean-
image regression objective benefits from that internal computation.

Synthetic comparisons retain only their own task/budget scope and require
their own baseline-adequacy checks. They do not supply the missing image
baseline or justify transferring an image-quality claim. Passing code/data
audits establishes reproducibility of a calculation, not scientific validity
of the architecture inference.

## Consequence for the research loop

Do not use the current image ranking to accept an ODE variant, reject an
architecture family, or claim that half local matrix work preserves useful
generation quality. Keep the numbers as historical observations. The recent
clock-source experiment had an empirical premise drawn partly from this weak
screen; its structural mathematics did not establish expected effectiveness.

The next priority is one credible ordinary JiT/SwiGLU baseline with a
source-supported training recipe, a checked training/sampling/evaluation
pipeline, adequate capacity and documented quality/convergence assessment.
Its budget and acceptance criteria must be justified before looking for an
ODE advantage. Only then should a paired FFN change be assessed under the
same conditions. This review selects no model scale, new hyperparameter grid,
training extension, or new experiment. Repairing the benchmark is separate
from rescuing a failed ODE variant. Report1 and sealed studies are unchanged.
