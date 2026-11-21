# Efficient internal solvers on Mac mini

This fixed-budget autoresearch cycle keeps the full-width ISRU drift and tests whether three evaluations of it are enough. The Product16 short screen selects a solver. Its full-budget models then use the separate confirmation arrays.

The two-step Adams–Bashforth variant starts with one explicit midpoint step. Its second step uses 1.5 times the current slope minus 0.5 times the initial slope, already computed during the first step. With two half-steps this makes three matrix calls per local solve. Ralston uses two steps and four calls. SSPRK3 uses one step and three calls. The external flow-matching sampler stays Heun128 for SW2.

A follow-up tests checkpoint averaging as a separate run on the original two-step models. It equally averages the parameters at 25k, 30k and the registered full-budget endpoint, with no additional training. The earlier Ring8 result showed a better velocity fit and worse sector balance late in training. If averaging suppresses late oscillations, confirmation SW2 and sector balance should improve.

[Protocol and trainer](../../../studies/muon_pd_20261102/efficient_solvers_20261122/)
