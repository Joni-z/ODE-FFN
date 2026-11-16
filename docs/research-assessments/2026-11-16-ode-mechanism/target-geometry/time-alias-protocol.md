# Check temporal aliasing before interpreting continuous-path diagnostics

Registered source audit, separate from the unchanged trajectory calibration.
The native synthetic TimeEmbedding uses16frequencies from1to approximately
1000cycles per unit t (reference/synthetic/models.py lines542-551). Existence
of that feature does not show that the learned model uses it.

For any Heun stage t=i/N with Ndividing1024, replacing a frequency f by f-1024
leaves both sin(2*pi*f*t) and cos(2*pi*f*t) identical in exact arithmetic.
Therefore128/256/512/1024step samplers cannot distinguish the corresponding
continuous fields, regardless of the network weights. This is a classical
sampling identity, not a new theorem or evidence of an architecture advantage.

Test the highest feature only on all twelve existing checkpoints. At FP64,
compare native-grid predictions on1025fixed grid times, cached continuous-time
field predictions on every original selection field pair, and native Heun128
endpoints for the first two sources from every bank. Change only this one
nonpersistent frequency buffer, then restore it exactly and verify all tensor
hashes. No weights, training recipe, native code or scientific checkpoint is
changed. Retain complete predictions outsideGit and publish their hashes.

Also retain the original selection diagnostic's128source/target pairs and five
times, measuring full velocity-Jacobian error for both fields. This is a
counterexample check for metric identifiability, not a newly selected primary
performance measure. Preserve both directions even if their ordering changes.

Repeat the native-grid and endpoint checks in FP32to quantify rounding:
grid atol/rtol1e-4, endpoint atol/rtol3e-4. The mathematical identity is exact
over real numbers; FP32andFP64have separately reported numerical residuals.

Require native-grid/output endpoint agreement within1e-9absolute/relative,
while measuring off-grid RMSchange relative to the model's actual oracle fitting
error. The intervention is a mathematical counterexample, not a proposed model.
If the learned output does not respond materially, the frequency's existence
does not explain the numerical issue. If it does respond, register a separate
non-dyadic/high-resolution qualification before a continuous-flow conclusion.
Keep all previous native-grid scores and all unresolved calibration rows.
