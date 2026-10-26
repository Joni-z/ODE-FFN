# Measurement and provenance

The experimental unit is a saved trained function. The variants are deterministic counterfactual evaluations on the same inputs, not independent training runs. No p-value or training-variability claim is made from repeated inputs, blocks or times.

Primary field MSE is the elementwise mean squared error against the complete existing validation bank's oracle field. Jacobian error is the mean squared Frobenius difference to the oracle input Jacobian, averaged over t=.1,.3,.5,.7,.9, each containing 512 inputs. The different normalizations are retained in comparison.csv.

Native return/centered-quadratic energy ratios sum energies over the same 2560 inputs per block before taking a ratio. The mean-return operator norm is ||CPu|| times ||u^T P||; it omits the input-dependent duration factor and does not characterize the full conditioned backbone Jacobian. Actual per-input traces, gates and vectors are saved in the raw native-terms files.

Full-field and panel NPZ files retain every prediction, input Jacobian where applicable, native/altered error and the identity

```text
new_error-old_error = 2 mean((native-truth)*(new-native)) + mean((new-native)^2).
```

The source computes explicit return subtraction; the independent path edits actual original matrices to CP and differentiates via vmap(jacrev). The all-second-order reference edits the second solve's actual order to K1. The final-mean control is independently projected with a normalized uniform vector. audit.json records all counts and tolerances. All stated checks passed under the registered bounds; none was relaxed.

The immutable runtime is /Users/mr.z/ode-workspace-prep/architecture-search/mean_return_20261026. Its 30-file analysis-source manifest includes the original source snapshot plus the new module, test, runner and protocol. provenance.json binds source and 72 input artifacts (banks, checkpoints and previous saved panel diagnostics); raw-artifact-sha256.json binds all 172 current output files. Counts in that manifest are checked directly when assembling this report.

No fresh holdout, sample-generation claim or trained constrained architecture is evaluated. Only global drop_return was eligible; drop_secondorder is a specificity reference. decision.json applies the registered comparison and derivative rules. No candidate passed, so no further compute allocation follows from this setting.
