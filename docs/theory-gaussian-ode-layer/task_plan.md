# Task plan: Gaussian denoising, flow matching, and the original ODELayer

## Objective

Start from a tractable simple distribution, rigorously derive the optimal denoiser and velocity field, and determine what these results imply for the professor's original ODELayer matrix P. Establish the mathematical argument before launching training.

## Scope and evidence

- Use independent Gaussian noise and the JiT interpolation z_t = t*x + (1-t)*e.
- Start with a centered Gaussian supported on a low-dimensional linear subspace; retain general covariance eigenvalues to expose direction and time dependence.
- Analyze the original user-provided ODELayer, including its scalar input-dependent gate, Taylor truncation, and identity term. Do not substitute later architectures.
- Distinguish the denoiser, velocity field, sampling flow map, and hidden-feature map.
- Prove representational statements with their assumptions and counterexamples; do not claim an architectural advantage from task structure alone.
- Verify algebra symbolically and with deterministic small matrix calculations. These checks are not training experiments.
- Preserve existing dirty repository work. No training launch or remote job submission is planned in this derivation phase.

## Phases

- [x] Inspect repository instructions, goal, and current state.
- [x] Derive conditional mean, posterior covariance, velocity, and exact sampling flow.
- [x] Derive the precise relation and obstructions for K=1, K=2, exponential limits, gates, and hidden coordinates.
- [x] Cross-check primary literature and run independent algebra/matrix verification.
- [x] Write an English technical derivation plus a readable Chinese explanation and a narrowly justified experimental decision.
- [x] Audit deliverables against the full objective and prepare the conclusions for delivery.

## Decisions

- Gaussian subspace data are a controlled analytic model, not a claim about all natural images or curved manifolds.
- Separate new analytic conclusions from previously collected empirical results; this task does not re-launch the architecture search.

## Errors encountered

- The default python3 has no SymPy. Resolved by using the existing /Users/mr.z/ode-workspace-prep/.venv-test/bin/python (SymPy 1.14.0, NumPy 2.2.6); no environment modification was needed.

## Status

Complete: the derivation and its verification are finished. No training has been started. This objective requests a rigorous mathematical basis before experiments, not an unconditional new training launch; the derivation does not establish an ODE-specific advantage warranting another architecture sweep.

## Completion audit

| Requested requirement | Current evidence | Assessment |
|---|---|---|
| Start with a simple distribution that can be solved exactly | derivation.md Sections 1 and 5 specify Sigma=diag(4,1,0), its rotated generalization, and a positive-thickness control | Complete |
| Strictly derive the optimal denoising function | Sections 2–3 prove squared-loss optimality and Gaussian conditioning, including singular Sigma; symbolic posterior and risk checks pass | Complete |
| Strictly derive the optimal velocity field | Sections 4–6 derive conditional velocity, irreducible risk, directional rates, and the exact flow map; continuity/covariance/flow checks pass | Complete |
| Relate the result to the original professor's P | Sections 7–11 analyze the original scalar gate and recurrence, give the exact K1 relation and K2 matrix equation, prove scoped obstructions, and separate physical/hidden/optimization interpretations | Complete |
| Think through the connection before training | Section 12 and explanation-zh.md give the conclusion and launch decision. Verification uses only symbolic algebra and deterministic matrix calculations; no training or remote submission was executed | Complete |
| Verify rather than rely on a plausible narrative | math-checks.json records 107 passing checks, including 48 symbolic checks. A subsequent audit confirms its input hashes match the current derivation and script, validates local links and Python syntax, and verifies whitespace | Complete |

The direct-layer impossibility statements are mathematical arguments in the text. The checks corroborate identities and examples, not global neural-network claims. No superiority over SwiGLU, new architecture, or empirical performance claim is made.
