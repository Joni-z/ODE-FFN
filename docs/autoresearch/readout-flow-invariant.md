# What the final readout conserves during sampling

This post-cycle07 mathematical review led to the completed [cycle08 verification](cycle08-readout-invariant/analysis.md). All18 fitted endpoints satisfy the predicted source-wise identities; six fixed-readout zero-P interventions and two older dimension controls also pass. The original derivation below is retained. This is not a selected architecture or training experiment.

## Exact output subspace

The raw native output is `W*LayerNorm(h)+b`. With learned gamma,beta and width64, define `C=I-11^T/64`,

```text
B=W*diag(gamma)*C,    c0=W*beta+b
N_theta(x,t)=B*z(x,t)+c0,    rank(B)<=63.
```

The normalized z has zero sum, including at finite epsilon. B,c0 stay fixed throughout inference even though z changes. For an orthonormal basis L of ker(B^T), `L^T*N_theta=L^T*c0` at every state/time, independently of every hidden P.

Direct prediction gives projected outer dynamics `y'=L^T*c0`. Hence `y(1)=y(0)+L^T*c0`: standard-Gaussian variance is preserved in at least65 directions. Heun preserves the same affine identity exactly in real arithmetic.

Both skip heads instead give `y'=-y/delta(t)+alpha(t)*L^T*c0`, where alpha=1/delta for clean and1 for shifted. The endpoint variance multiplier is r² for both, `r=exp(-integral dt/delta)=.05/e`. Biases alter the mean, not the variance. This is a trained complete-predictor restriction, not an ODE-FFN-specific property.

## The actual Heun prediction

For step h, rate `a_i=-1/delta(t_i)` and forcing `f_i=alpha(t_i)*L^T*c0`,

```text
R_i=1+h*(a_i+a_(i+1)+h*a_i*a_(i+1))/2
d_i=h*((1+h*a_(i+1))*f_i+f_(i+1))/2
y_(i+1)=R_i*y_i+d_i.
```

The32-step product of R_i differs from continuous r and was computed before examining projected fitted samples. Cycle08 tested this **source-wise identity** for all18 final cycle07 checkpoints, using actual W,gamma,beta,b, common sources and saved samples. Every older artifact was preserved.

## Consequence and limits

The intersection with the126D target-normal space has dimension at least63. There direct prediction retains Gaussian standard deviation1, while skip prediction has standard deviation r (or its Heun counterpart). The target marginal is Gaussian with standard deviation sqrt(q_N), including zero for thin data. Projection gives a full squared-Wasserstein lower bound `dim(intersection)*(r-sqrt(q_N))²`, plus squared mean displacement; for direct use r=1.

This bounds **full W2**, not the128-direction empirical SW2 in the benchmark. They must not be compared as interchangeable metrics. Finite-sample energy also differs from population variance; the deterministic source-wise identity is the first check.

The verified result localizes another complete-network restriction that hidden P constraints cannot repair. The older2D/16D controls have full-row-rank readouts, so this obstruction does not explain their negative results. An analytic input skip is already an ordinary control and JiT uses this decoupling; renaming it an ODE method adds no novelty. No learned radial skip, wider readout, changed cap or head-training rescue is selected. Any different architecture needs separate justification and the older benchmarks remain mandatory.
