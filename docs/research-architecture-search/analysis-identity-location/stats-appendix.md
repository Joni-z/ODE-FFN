# Measures and analysis scope

This is a fixed-budget architecture screen, with no inferential testing.
Primary oracle MSE averages squared coordinate errors on the existing
validation FM inputs. Their labels use the analytic population conditional
velocity. Training still uses independent cached-target-bank pair labels;
the population oracle is not the empirical bank's conditional mean.

The separate common-point diagnostic uses the first512 validation targets and
one shared Gaussian source array, at t=.1,.3,.5,.7,.9. Field MSE averages
coordinatewise squared error. Jacobian error averages squared Frobenius
errors, then averages the five times; it is not dimension-normalized like
field MSE. All models see identical diagnostic inputs. Saved float64 oracle
and predicted arrays support independent replay.

Generation uses2048 fixed Gaussian starts, Heun32,2048 validation targets
and128 fixed sliced directions. SW2 is the saved implementation's sliced
Wasserstein metric. It is a validation comparison, not a fresh holdout or FID.
The better oracle, Jacobian and SW2 baseline can be a different SwiGLU arm;
the screen selects the better baseline separately for each metric.

For a metric Y, the first-removal contrast is
((Y_IM-Y_MM)+(Y_II-Y_MI))/2; second-removal is
((Y_MI-Y_MM)+(Y_II-Y_IM))/2. Interaction is Y_II-Y_IM-Y_MI+Y_MM.
All contrasts and cells are reported descriptively without selecting a
favorable subset, confidence intervals, or causal claims beyond the defined
implementation changes under this training setup.

Initial raw/centered spectral summaries average six solves per model.
The fixed reference map gamma*I+.6P is a representation diagnostic. It is
not the actual map at learned D(h), its full input Jacobian, or the complete
velocity Jacobian. Full singular values and spectral export hashes are kept
with the raw centered-operator JSONs.

Native checks pass180 checkpoint identities and150 common-point arrays.
Independent Mac replay passes all30 final checkpoints and150 arrays, with
maximum prediction difference9.99e-15 and derivative difference5.29e-14.
Initial/intermediate states remain on Lenovo; selected raw collection includes
final states. Replays do not claim optimization or generation replication.
