# Research question

- Question: how does the professor's literal ODELayer compare with ordinary
  FFNs when token attention and real image patches are present?
- Why now: prior vector-only screens did not produce a useful successor;
  their high-dimensional global-readout restriction differs from a patch head.
- Scope: new baseline coverage only. No failed operator is rescued or nominated.
- Proposed baseline cohort: original scalar input-clock K1/K2, ordinary SwiGLU
  covering ODE parameters, and a separate SwiGLU covering executed dense FFN
  matrix work. Width64/depth3/4x4patches on32x32RGB; unconditional, seed0.
- Missing evidence: feasibility/actual costs, fixed quality protocol, trained
  image endpoints, generation quality and independent audit. None exists yet.
- First falsification boundary: any condition embedding driving the professor's
  clock is the wrong operator. Stop before training and correct the adapter.
- First action: prepare official data and perform source/budget/derivative/
  resume/throughput preflight. No quality outcome is inferred from these checks.
