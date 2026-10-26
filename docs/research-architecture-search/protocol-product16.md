# Stage6: fixed architectures on a genuinely multidirectional population

Registered after the failed Ring8 clock-sharing screen and before any model is trained on product16. This stage adds no architecture. Test the unchanged scalarK1, vectorK1, full SwiGLU and budget SwiGLU on the frozen population defined and derived in product16-mathematics.md. Exclude the failed tied/reused-controller variants.

## Purpose and competing explanations

The existing vectorK1 lead has not established a practical benefit. The rotated Ring8 target also did not test more than two state-dependent posterior directions, despite its16 observed coordinates. Product16 supplies a better discriminating test of the original directional-adaptation hypothesis, including an early/mid-path region with about6–11 effective directions and late-path sparse ambiguity. No hidden-network rank obstruction is asserted for the scalar model: its intervening activations and complete backbone can supply other state dependence.

The scalar/vector pair nearly matches parameter/MAC budget; their non-gate initial tensors and RNG states are paired. Full SwiGLU is the practical reference at the same approximate budget; budget SwiGLU checks whether a narrower ordinary block can explain the comparison. These four choices are frozen before outcomes. A win by both ODE variants would support a different explanation than a vector-specific win. A small velocity-MSE change without generation/derivative benefit is insufficient.

## Fixed data and training

Population: the orthogonally rotated product of16 symmetric two-Gaussian mixtures, mu=geomspace(.75,2.5), within variance.04, fixed stored rotation. Gaussian source, independent source/target draws and uniform external t. No latent signs, component labels or rotation enter the network.

Exact v1 data manifest SHA256: f15e3bebdc673a1c51eb643723f863559192409d6ba558e2272739dc7dcbbb05. Train100000, validation20000, test20000, probes2048; independent streams, seed0. Population calibration used validation targets only; there is no preceding product16 model outcome.

Train all four frozen architectures fresh for20000updates, seed0, same width64/depth3/batch512/AdamW3e-4/wd0/clip1 settings and existing evaluation/checkpoint intervals. Final20000 is the primary endpoint. Intermediate5000/10000 curves are descriptive and cannot select the winning checkpoint. Two monitored CPU workers, immutable mirrored source and no automatic resubmission.

## Readout

Primary: final random-time validation oracle MSE. Secondary: common512-state/five-time velocity and squared-Frobenius Jacobian error, broken down by time and latent coordinate; validation SW2; P/gate trajectories and measured compute.

After all final checkpoints exist, run the frozen test comparison for all four arms once. Test field evaluation uses common512 source/target pairs at t=.1,.3,.5,.7,.9. Sampling uses8192 common Gaussian sources,128 common projection directions, and Heun32/128. Use two disjoint8192-target test subsets for an empirical reference. Compute SW2, latent marginal1D W2, fraction of coordinates more than.6 from either well, sign balance, off-diagonal sign dependence, covariance error and target log density. Preserve raw field/sample banks and hashes. These metrics are diagnostics and never extra training objectives. Full joint-mode coverage is not an acceptance metric with65536 possible sign patterns.

For a materially useful directional lead, seek at least10% lower validation oracle error versus both scalarK1 and full SwiGLU, without more than5% degradation in common Jacobian error or test marginal1D W2, and with meaningful evidence from full SW2/well/sign checks. These are engineering criteria, not p-values or permission to ignore unfavorable metrics. If only a small metric improves, retain the entire comparison and leave the goal unachieved. Existing Ring8/rotated16 failures remain in the report; the new task cannot erase them.

No new activation, spectral restriction, gate function, width, optimizer or loss is tuned from this stage. An architecture that fails these requirements returns the research to operator design rather than triggering a rescue sweep. Passing still needs a measured latency/throughput comparison and an independently recomputed final analysis before any completion claim.
