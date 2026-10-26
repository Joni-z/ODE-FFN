# Evidence and mathematical implementation notes

The complete orientation audit is PASS. Product16 ODE A/B field MSE is .037320/.361073, while SwiGLU192 A/B is .033826/.032544. Fixed-weight switches in either direction substantially worsen nonlinear ODE predictions. The full transposed sensitivity, not generic input-derivative computation, is the next object to understand. Hard conservativity as a unique cause remains unresolved.

Each original ODE adjoint contains a transposed polynomial and a rank-one gate-derivative term. The layer identity was verified against autograd on36 actual learned parameter sets, with maximum adjoint error1.78e-15. That verifier used deterministic algebra inputs, not task hidden states. This new study extends it to the complete conditioned network and actual task fields.

The two-clocks literature review does not itself justify a new cold run: the existing recurrent-depth and time-generator screens are negative. The current experiment asks a different implementation and mechanism question while keeping the useful learned A function fixed.

Seven focused explicit-adjoint unit cases pass, including complete input/time/all-parameter gradients, actual AdamW updates, no_grad/inference_mode, batch independence and a gate-omission finite difference. The full evaluator's first compile caught an extra closing parenthesis before launch; inspecting the actual panel bank also established the oracle_jacobian key. Both were corrected before any evaluation output. No task result existed at that point.
