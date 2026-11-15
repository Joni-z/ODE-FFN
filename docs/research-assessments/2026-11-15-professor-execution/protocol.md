# Registered professor follow-up

## Scope and computation
Use145 physical GPUs4-7 only, one CPU thread per synthetic process. GPU4:
Ring8 tanh; GPU5: Product16 tanh; GPU6: Ring8 ISRU-D then SwiGLU64/Muon replay; GPU7: Ring8 Free ISRU replay.
These become the frozen-checkpoint diagnosis cohort after completion.
Reuse historical data and source-verified native training/evaluation. No package
rebuild, additional seed, optimizer sweep or image training in this wave.

## Activation control
Train only pd_tanh_k1 on Ring8 and Product16: seed0,20kupdates,batch512,
AdamW3e-4, unchanged auxiliary optimizer/clipping/backbone/data streams.
Replace q(h)=h/sqrt(1+1.2h^2) by tanh(sqrt(1.2)h)/sqrt(1.2).
Both are odd, have derivative1 at zero and saturation1/sqrt(1.2).
Their cubic coefficients differ (-0.6 versus-0.4); this isolates response
shape, not a claim that all local derivatives are matched. Keep K1 and all
parameter initializations unchanged. Verify8->16 resume against16continuous.
Use existing linear, ISRU-D and ordinary controls. Final comparisons use
shared eight4096sample banks, Heun128 and256directions, including pooledSW2.
The native Heun32 trainer output is a separate historical compatibility metric.

## C8 symmetry and sampling
Restore Ring8 ISRU-D, Free ISRU and SwiGLU64/Muon at15k and20k. Check
v(Rx,t)=Rv(x,t) for45degree rotationR. Validate against the exact mixture
oracle. Evaluate the group projection v_sym(x,t)=mean_k R^-k v(R^k x,t),
with the same native Heun128 sampler and eight IID banks as cycle14.
Use rotated trajectory pairs for the equivariance check; do not replace the
IID evaluation bank with orbit-balanced points. Save distributions, mode
counts and trajectories. C8 symmetry alone does not imply zero sector
crossings. Keep paired bank differences distinct from training variability.

## Learned spectra
From raw update_spectra records, report actualW andDeltaW separately: singular
spectra, stable rank, entropy effective rank, minimum singular value and
condition number. Keep layer identities. AdamW3e-4/Muon.02 are selected
historical settings, not matched update magnitudes. No inference from
high-rank updates alone to high-rank weights; no spectral constraint added.

## Decision
If tanh and ISRU behave alike, ISRU's specific nonlinear shape is not needed
for the observed gain. If C8projection reduces mode imbalance and improves
sampling while preserving or improving field fit, investigate symmetry/error
allocation before increasing Taylor order. If these diagnostics do not explain
the gap, retain the negative result and compare responses along trajectories.
Choose further image experiments only after this evidence is available.

## Checkpoint recovery decision
Lenovo is unavailable and the retained local cycle14/15 archives contain
samples rather than full weights. Replay the three Ring8 references from
seed0 using the pinned cycle13 recipe on145, keeping15k/20k states. These are
new same-recipe trajectories, not recovered historical weights. Compare their
native metrics to archived results before interpreting the C8 intervention.
This raises the bounded training count from two to five; no additional seed
or architecture is introduced. Product16 original weights remain available.
