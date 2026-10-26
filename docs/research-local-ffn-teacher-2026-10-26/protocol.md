# Fixed local FFN teacher experiment

Registered before student outcomes, 26 October 2026. Accountable project owner: Zhizhe Zhang. Authorized use: local synthetic-model research. No personal or controlled data and no external data transfer.

## Observation and question
The [complete-network realizable-teacher experiment](../research-realizable-teacher-2026-10-26/analysis-report.md) found original ODE relative squared error 0.629% on its own frozen teacher and 10.652% on a SwiGLU128 teacher. That does not distinguish a local FFN fitting problem from representation coordination and optimization across the full network. Existing frozen-feature probes and small-bank tangent fits do not answer the local-function question.

For each of the three block positions, does a cold original ODEFFN fit the frozen SwiGLU FFN on common inputs as accurately as generic FFNs at the declared budget? Does it fit a known representable original-ODE FFN function on those same inputs?

## Objects and common coordinates
Use the unchanged direct Product16 ODE and SwiGLU128 checkpoints from the conservative-readout study at step 5000. Define the input Z_j to position j as the actual post-normalization, post-FiLM tensor consumed by the SwiGLU128 teacher's FFN. Capture it by a native pre-hook while evaluating that full teacher. All students and both local teachers at position j receive exactly these same 64-dimensional coordinates. No hidden states are transplanted between trained networks.

The six targets are T_{j,a}(z) = FFN_{j,a}(z), for three positions and a in {original ODE K2, SwiGLU128}. For the ODE targets, this deliberately evaluates the ODE teacher module on the common SwiGLU input distribution, rather than its native prefix distribution. Teacher output/derivative energies must be reported; equal amplitude is not assumed to mean equal complexity.

Each own-family teacher is an explicit exact representer in its corresponding student's architecture. Restore it only in separate procedural checks, never into a cold student.

## Data and allocation
Training inputs: first 100000 x/t rows of the already sealed realizable-teacher input cache, flattened in its original order. They derive from training targets and fresh noise/time, without using validation or test labels. The common frozen SwiGLU prefix maps them to each Z_j. Cache inputs and the two deterministic local targets once per position. Validation uses all existing 20000 uniform-time field inputs. Secondary fixed-time evaluation uses all five existing 512-input panels. No fresh test set is opened.

Within every cell, sample 512 training-bank indices per update using a dedicated generator seeded 0. All eighteen cells consume identical index sequences. This is a new fixed-bank local regression, not a relabeling or extension of the earlier complete-network runs. Prefix features are fixed and never updated.

## Eighteen cold cells
Two teacher families × three positions × three students (original scalar K2 ODEFFN, nominal SwiGLU128, nominal SwiGLU192). Obtain each cold module from the unchanged full constructor at its original position, and verify it against the historical step-0 checkpoint. Only that FFN's parameters are optimized. No surrounding network, target weights, output decoder, normalization, or auxiliary loss is trained.

Fit mean squared error over all 64 outputs with 5000 updates, batch 512, AdamW learning rate 3e-4/default moments, weight decay 0, global gradient clip 1, no schedule. This preserves the established finite budget without optimizing it for the diagnostic. Record actual parameters, forward MACs and forward/backward MACs for these isolated modules; exclude pointwise work explicitly.

## Outcomes and interpretation
Primary: all-validation MSE and MSE divided by the corresponding local teacher's mean squared output. Secondary: all five time-panel risks, full 64×64 input-Jacobian error on the first 128 fixed validation inputs, training/validation curves, and every original P trajectory. The Jacobian panel is a conditional descriptive diagnostic, not a full-distribution derivative risk.

Before outcomes, define a consistent local-gap pattern as: on at least two of the three SwiGLU targets, ODE relative MSE exceeds 5%, both generic students are at or below 1%, and ODE error is at least four times each generic error; meanwhile all three ODE self-target relative errors are at or below 5%. These cutoffs operationalize a substantial finite-budget contrast, not a statistical hypothesis test or an architecture acceptance gate. Report every cell, including exceptions and failed procedural controls. Layer-specific patterns remain exploratory if the uniform pattern does not hold.

If that pattern appears, the gap exists inside a single FFN under the common teacher-prefix distribution and does not require full-network coordination to appear. It still does not prove an expressivity lower bound or identify the cause. If local fitting is accurate while complete-network fitting is poor, the simple local-function explanation weakens; representation distribution, conditioning, and joint optimization remain alternatives. If ODE self-fitting is poor, local learnability under this budget remains unresolved and prevents a clean capacity interpretation. If generic students also fail, target difficulty is not isolated to the ODE module.

No outcome nominates a generative architecture, justifies teacher supervision in a deployed method, or replaces ordinary FM, transfer, and measured-cost validation. No post-result learning-rate, target-strength, budget, activation, or gate rescue is allowed in this setting.

## Measurement and execution checks
Validate native pre-hook capture and replay of the historical full-teacher validation outputs, unchanged cold weights/buffers, exact teacher representers, and teacher immutability. Compare separate explicit formulas and analytic local Jacobians with native autograd in float64 at initialization and trained teacher weights. Establish exact three-update versus two-update-plus-resume histories, including optimizer, RNG, and actual input/label digest chains. Teacher cache replay is bitwise under the identical preparation batch boundaries; arbitrary reordered minibatches permit float32 roundoff up to 2e-6 times max(1, output absolute maximum), declared before checking.

Snapshot step 0/1000/2500/5000, including weights, optimizer, runtime buffers, global/sampling RNG, and actual consumed-stream digests. Record metrics every 500 steps, status every 100, and real process CPU/RSS every five seconds. Run at most two single-thread CPU children. Final audit must reconstruct every validation output, every endpoint time/Jacobian panel, checkpoint spectra, and actual replay chains against hash-bound caches. Freeze training sources before one formal launch; preserve any technical failure rather than overwriting outputs.

## Provenance
The hypothesis-generation, experimental-design, planning-with-files, and daily-coding workflows inform this diagnostic. Their procedural source is [Kassis, Agarwal, He, Patel and Brueckner (2026), Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065); this is not scientific evidence for an ODE architecture. The related primary literature and its limitations are in the [latest review](../literature-ode-practical-2026-10-26/review.md).
