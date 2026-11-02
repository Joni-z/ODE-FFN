# Evidence and decisions

Prior36+30 trajectories are complete and replayed. Muon produces higher-rank updates
but larger P; matched-magnitude momentum does not explain all field gains. Selected
matrix decay0.1/0.5 harms fitting. PD/DP/sandwich ordering alone does not beat SwiGLU.
Prior Stage8 tested wider-positive and affine DP atK1/5k/AdamW3e-4, with negative
results (docs/research-architecture-search/clock-range-readout.md). It did not test
PD/K2/Muon or replacing the nonlinear controller with a locally matched affine one.
The new test is an interaction, not a rediscovery of affine gates.

145 GPU5 is now occupied.3November fresh snapshot shows GPU0/1/2/3 free; choose GPU0
UUID GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 after another check before launch.
Lenovo GPU0 remains desktop-only415MiB. No environment changes are needed.

Additional read-only decomposition registered before execution: archived scalar P
at0/1k/2.5k/5k/10k/15k/20k, both AdamW rates and both Muon rates, Ring8/Product16.
Normalize rows and columns separately and measure resulting spectral norms alongside
row/column magnitudes. P=diag(r)R gives ||P||2<=max(r)||R||2; this is an inequality,
not an exact multiplicative attribution. Checkpoint/config hashes are retained.
This asks whether our P growth resembles magnitude drift discussed by Muown, without
introducing Muown or any new optimizer trick. CPU-only readout; training is unchanged.

Before candidate outcomes, identified a controller-rank confound: full affine can
learn rank64; original32-hidden MLP logit Jacobian has rank<=32. Record this explicitly,
then distinguish using a factorized-linear control if affine suggests a lead.
Do not label a win as evidence that nonlinear gate features are unnecessary.
