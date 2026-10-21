# Anchored gate geometry: protocol fixed before execution

Question: at the same state and time, does input-gate dependence improve the complete velocity network's local derivative relative to the analytic FM field? Is any improvement concentrated in uncertain/expansive regions?

Checkpoints: the latest completed CPU gate study on 145, primary step 5000, seed 0. All five kinds (input ODE K2, fixed ODE K2, scalar ODE K2, SiLU, SwiGLU) on anisotropic Gaussian and Ring8. No checkpoint selection, new training, architecture change, or test-target lookup. Record checkpoint/config/data/source hashes. Older GPU runs are not pooled with these models.

Evaluation times: 0.1, 0.3, 0.5, 0.7, 0.9, with equal descriptive weight. Distribution A: 512 validation targets (first indices), paired to independent standard Gaussian draws from numpy default_rng(0), reusing these pairs at all times and for all models. This is a probability-sampled panel of the interpolation marginal, not a grid over unusually ambiguous states. Distribution B: 128 common standard Gaussian initial points (a fresh default_rng(0) stream), transported separately by each model with Heun100, captured at the same times. Trajectory panels differ by model and cannot be described as matched-state architecture comparisons. Their source indices are matched. These are finite numerical trajectories, not exact ODE solutions.

Computation: evaluate in float64 after loading the saved float32 weights; record forward cast drift against the original float32 computation. Check the analytic velocity against the repository oracle, and its Jacobian against autograd/finite differences. Do not rank generation quality by these diagnostics. Retain all panel rows.

At each anchor x,t, compute native gate values and native full-network Jacobian J. Hold the six scalar gate values at their anchor values while differentiating neighboring inputs, yielding J0. The forward output at the anchor must agree with the native model. This is an anchored local derivative decomposition; detaching dynamically recomputed gates alone would not define the same finite-difference comparison. Let C=J-J0 and E=Jstar-J0. Primary response:

    gain = ||J0-Jstar||_F^2 - ||J-Jstar||_F^2
         = 2 <E,C>_F - ||C||_F^2.

Positive gain means the gate derivative improves the full-field local Jacobian at that input; it does not show improved forward velocity, training, or endpoint distribution. Report mean gain, mean gate-free/native derivative errors and fraction with gain>1e-10. Normalized gain is mean gain / mean gate-free error, with a numerical denominator guard. Report correction/error alignment and symmetric/skew parts as descriptive companions. Fixed/scalar/ordinary baselines must have C=0.

Oracle geometry: store full Jstar, eigenvalues and eigenvectors. Report squared directional derivative errors along its maximum and minimum eigen-directions; symmetric maximum eigenvalue error and skew norm. Categorize target geometry by eigenvalues (contracting, mixed, expanding; zero tolerance 1e-8). Ring8 posterior component entropy divided by log(8): low<0.25, middle in [0.25,0.75], high>0.75. Always report counts and panel fractions; do not turn small/empty strata into conclusions. Gaussian entropy is not applicable. Keep log-density for rare-state inspection, without excluding or upweighting rare regions. No threshold tuning after seeing outcomes.

Validation: anchored output equality, polynomial/Jacobian identities, finite differences for native and anchored maps on a fixed small prefix, strict checkpoint/config identity, no-gradient parameter changes, and frozen-source hashes. Baselines provide zero-correction controls. A CPU preflight measures a representative batch before the full run. Monitor elapsed time, progress, CPU and RSS to files. Refuse to overwrite any completed output directory.

Interpretation: a consistent positive gain in uncertain/mixed regions would support the proposed local mechanism for this trained network. Uniformly negative/negligible gain, or no concentration in the predicted regions, weakens that specific mechanism. Either outcome remains a local explanation; it cannot distinguish additional gate capacity from other training effects, nor establish ODE superiority or transfer to JiT. No architecture is automatically selected after this diagnostic.
