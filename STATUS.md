# Current project status

## More accurate hidden ODE solves did not improve generation

The matched-solver training comparison is complete. With the same parameters,
initialization, training stream and eight P applications per local solve,
RK4 reduces normalized hidden solver error by 45.580x on Ring8 and 22.245x on
Product16. Confirmation pooled SW2 is 0.091 vs 0.091 on Ring8 and 0.065 vs 0.064 on
Product16 (RK4 vs midpoint, three decimals). The two-task generation-gain
prediction fails while the hidden-fidelity prediction succeeds.

[Complete analysis and figures](docs/research-assessments/2026-11-16-ode-mechanism/rk4-results/analysis-report.md),
[exact numeric data](docs/research-assessments/2026-11-16-ode-mechanism/rk4-results/summary.json),
[source/execution/protected-file audit](docs/research-assessments/2026-11-16-ode-mechanism/rk4-audit.json).

Both 20k training trajectories, four common-bank evaluations and eight hidden
observer/verifier commands have actual exit 0. All 1,920 hidden states qualify.
NumPy independently reconstructs all 16 model/split metric rows (max 2.78e-17);
separate integer-CDF statistics agree within 2.71e-20. Cost checks confirm equal
parameters and dense MACs, with descriptive same-process CPU timings retained.
All original checkpoints, samples, update spectra and logs remain outside Git.

## Next mathematical question

A more faithful internal solution is not sufficient by itself. Next distinguish
weak exposure of hidden bias at the complete field from solver-specific learned
compensation, using a crossed weight/solver comparison and the exact risk identity
R_cont-R_num=2 mean<error,delta>+mean||delta||². This is a frozen-weight diagnostic,
not a new response or image architecture. The full4096-input diagnosis is complete: all four native/independent commands
and their supervisor exit0. Both Ring8weight sets prefer midpoint, contradicting
the strong own-solver compensation prediction. Native midpoint-to-continuous
replacement raises fieldMSE6.121%on Ring8 and1.762%on Product16for midpoint-trained
weights. Every point is numerically qualified; field bootstrap is independently
verified. The next step tests whether these field-risk rankings survive actual
common-bank sampling, using the same retained weights and ordinary controls.
[Field result and figures](docs/research-assessments/2026-11-16-ode-mechanism/field-exposure-results/analysis-report.md).

[Protocol and falsifiable predictions](docs/research-assessments/2026-11-16-ode-mechanism/rk4-followup-protocol.md).

## Earlier controls and preserved image benchmark

The parameter-free modified-equation calculation predicts the trained midpoint
bias (B=.831--.971). The equal-dimension null rejects special tangential sensitivity
in all six models. Uniform 512/1024 sampling qualifies all 12 pilot comparisons,
but target-quality shifts are small and mixed. All earlier failed gates remain.

CelebA128 at 160 epochs remains native 24.304/K1 27.789/K8 28.083 FID50k.
[Matched image table](docs/celeba128-wide-20261114/continuation-160/matched-results.md).
Reports1/2, 97 protected files, 11 Report2 bindings and five native image source files
are unchanged. Mac CPU environment 04f11f75 has no package changes. No current study
worker remains live. 145 physical 0-3 are prohibited;4-7 were occupied and unused.
Raw root: /Users/mr.z/ode-workspace-prep/coefficient-20261116/rk4-control/.
