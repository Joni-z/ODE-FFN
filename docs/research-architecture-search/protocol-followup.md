# Stage3: frozen lead, longer learning and nonlinear16D verification

Registered after the scalar/vector screen and before these follow-up outcomes. The lead is the unchanged vector-rateK1 block, gate hidden32; scalarK1 and SwiGLU remain references. No additional architectural change, spectral constraint, loss term or learning-rate schedule is introduced.

## Why extend the budget

At5000steps Ring8 validation oracle MSE is .022457 scalarK1, .019438 vectorK1 and .016221 SwiGLU. All three curves are still improving. VectorK1 passes the declared directional-adaptation screen but does not yet establish an advantage over SwiGLU. Its Gaussian16 validation regression versus scalarK1 is4.61%, while the common-panel regression is6.54%; retain both observations.

Continue each Ring8 reference from its exact5000-step model, optimizer and RNG state to20000steps, preserving the original config bytes, data manifest, learning rate and sampling. Copy historical runs into a new output tree; do not overwrite the source study. Record continuation source/config/checkpoint provenance, remove the copied stale COMPLETE marker while training, and retain all checkpoints/metrics. Primary endpoint20000;5000 and10000 are descriptive learning-curve comparisons, not cherry-picked winners.

## Fixed nonlinear higher-dimensional task

Define y=Qz in16dimensions. The first two coordinates of z follow the existing radius3/eight-component Ring8 mixture with within-mode variance.04. The remaining14 coordinates are independent centered Gaussians with variances geometrically spaced from.04 to4. Q is one fixed orthogonal matrix generated with seed0 and is stored in the dataset specification. The network receives neither Q nor component labels.

All component variances are positive, so this is a smooth full-dimensional mixture, not a singular-manifold example. The source remains isotropic Gaussian under rotation. Therefore `v_y(x,t)=Q v_z(Q^T x,t)` and `J_y=Q J_z Q^T`; density and posterior component entropy transform consistently. Unit tests check the rotated Jacobian by finite differences and existing endpoint/moment checks.

Dataset is independently generated from this declared population:100000training targets,20000validation,20000test,2048probes,seed0. Train the same three frozen architectures for20000updates with the existing optimizer/batch/time settings. Use the new test split only for the final frozen comparison; do not tune the architecture on its result.

## Readout and acceptance

Report oracle velocity error and Jacobian error at shared inputs, heldout generation under matched Heun resolutions, parameters/MACs and learning curves. On the rotated task also inspect the known two-dimensional mode subspace and its14-dimensional complement; a mean over16 coordinates must not hide failure to recover modes. Separate fixed validation screening from final heldout evidence.

Evidence of directional adaptation versus the scalar parent is insufficient for goal completion. The lead must show a material useful quality/compute tradeoff against SwiGLU and retain evidence on nonlinear16D. If it remains dominated, keep the goal active and use the measured failure to choose the next hypothesis; do not relabel a negative result as success.
