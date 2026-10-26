# Evidence boundary and derivation notes

The prior frozen-model diagnosis finds the nonlinear odd component accounts for approximately99–100% of the direct ODE–SwiGLU excess, while the actual middle SiLUs are far from the centered special case. This does not identify an architecture improvement or prove a lack of representational capacity.

Earlier population-mean training does not close the K1 ODE gap; feature/head refits and small-anchor parameter kernels already test other representation questions. A broad full-kernel rerun or a relabeled frozen-SwiGLU recurrence would not supply a new mechanism. Current research concerns a different exact operator: the K2 dependence on one P at a fixed input-selected duration.

For one original layer, Q_a(P)=I+aP+a²P²/2, with the input gate computed once and independent of this layer's own P. For a perturbation H,

```text
D Q_a(P)[H] = aH + a²(PH+HP)/2.
```

At P=-I/a this derivative vanishes for every H although Q_a(P)=I/2. This is a real stationarity witness, not evidence that any trained layer is near it. For a real matrix P, singularity of this Sylvester operator occurs if some eigenvalue pair satisfies1+a*(lambda_i+lambda_j)/2=0. Eigenvalue distance is not singular-value conditioning for nonnormal P.

Search of existing project notes and primary sources is used to locate overlap. A public search also located local-linearization/exponential-integrator and data-controlled ODE prior art; it did not select a new architecture. The exact frozen-SwiGLU coupled generator had already been reviewed and rejected in coupled-generator-prior-review.md, so it is not implemented here.

The project search locates the Sylvester differential in research-2026-10-21-reassessment/notes.md and the scalar stationary point in research-2026-10-23-direction/mathematical-audit.md. These are earlier theoretical observations, not newly claimed theory. No current trained-K2 order-route cancellation measurement was found in that bounded search. The old shared-controller kernel diagnosis used K1 generator arms and small anchor sets; it is not repeated here.

All new computations remain local and use already authorized synthetic models/data. Hypothesis-generation, experimental-design, daily-coding and planning-with-files procedures guide the work; they do not supply empirical support or external approval. No new safety or data-governance gate is involved in this local synthetic study.

Preflight passed on the first attempt: all five old native panels, all six positive solve captures, full local derivative witnesses, and the projection-envelope identity. Frozen source contains eight files. Main session22703 and independent replay72328 both terminated with exit0; no process was restarted. No primary layer shows severe cancellation. The actual nonlinear local ratios are.8909–1.2958 and coherent ratios.7809–1.3729. The full90-row spectral context and60-row objective/time table are retained.

After observing positive bounds at all recorded duration quantiles, an algebraic corollary was computed over the entire declared interval using each final P's symmetric minimum and spectral norm. All mu>-1, yielding a uniform normalized singular lower bound at least.12234 and conservative condition upper bounds at most26.2918. This is explicitly post-result context, not a changed criterion or a complete-field stability claim. The proof does not require normal P. Independent final verification also recomputes these bounds with Torch eigensolvers.

Workflow provenance, with metadata verified earlier on26October2026: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner, Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents (2026), https://doi.org/10.48550/arXiv.2609.00065 . This is an acknowledgment of research procedures, not evidence for the model or hypothesis.
