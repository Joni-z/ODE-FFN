# Exact composition of the complete adjoint

Registered before decomposition measurements. This is a mathematical/frozen-weight experiment prompted by the cold gate-removal result, not a new trained architecture. Use all eighteen existing A/B endpoints: original ODE K2, SwiGLU128/192, Gaussian2D/Ring8/Product16. No state or exposed validation input is selected by a favorable partial result.

## Exact expansion

The unchanged network is `G=output(norm(B3(B2(B1(input(x))))))`. At each actual forward state, write a residual block derivative `J_Bi=I+Di`, including its normalization, FiLM and complete FFN derivatives. Set `q=J_norm^T*Wout^T*x` at the actual final hidden state. Then

```text
J_G^T x = Win^T (I+D1^T)(I+D2^T)(I+D3^T) q.
```

Expand by the number of selected residual derivative factors. Degree0 is the identity path, degree1 the sum of three single-factor paths, degree2 the three correctly ordered pairs, and degree3 the triple path. The exact potential field A=.5*(G+J_G^T x) splits into four components C0..C3; C0 includes .5*G plus the degree0 reverse term. All components retain their actual input dependence and the seed x. Degrees count cross-block derivative factors, not Taylor order in P or external sampler order.

Compute all components with ordinary tensor formulas, sharing one primal pass and propagating the polynomial coefficients of the reverse product. Sum them and compare to the already independently verified full A readout. Component input Jacobians also sum to the complete Jacobian. Every partial sum `Rk=C0+...+Ck` uses the same actual forward states; R0 is not a forward network with all FFNs removed.

## Mathematical checks before interpretation

1. For noncommuting symmetric positive definite A and B, the local gradient maps Ax and Bx compose to BAx, whose Jacobian is not symmetric. They can also each be exact flows of symmetric linear generators log(A),log(B). This refutes inferring a global gradient map from locally gradient updates or flows.
2. A symmetric P alone does not make the original state-clock map a gradient: `J_L=I+aP+a²/2P²+(Ph+aP²h)grad(a)^T`. Give a deterministic bounded-clock counterexample with a nonzero skew term.
3. At small width, directly enumerate all2^3 residual-factor paths and compare components, full input Jacobians and all parameter gradients to the ordinary potential construction. Check the full x-dependent reverse seed. Reject unsupported layouts.

## Saved-weight measurements

For all eighteen states, evaluate all four partial sums on the complete uniform-time validation field bank and every five512-input panel. Reproduce full A values and Jacobians from the independently saved canonical crossover arrays. Record individual component energies, all cross-component inner products, skew-Jacobian components/cancellation and partial/full field and derivative errors. Save per-input predictions, not just summary norms. Exact finite-change risk polarization must account for the change from full A to every partial sum. No individual component norm is a causal importance score.

Report all four sums and both training orientations. The B-trained states distinguish patterns of an A-coadapted model from generic raw-network behavior; their A-evaluation is contextual and not a candidate. Do not tune signs, multipliers, subsets, truncation degree or local replacements after looking at these outputs. A fixed-weight truncation penalty cannot prove that a newly trained simpler architecture will fail or succeed. Local symmetry counterexamples rule out an invalid guarantee, not all local designs.

No training, new solver, fresh test consumption, isolated latency claim or hardware allocation is part of this experiment. Mac mini one-thread bounded analysis with actual process/memory telemetry; Lenovo is presently offline. Source/checkpoint/bank hashes and all identities must pass before reporting. An observation timeout does not terminate or restart a live process.
