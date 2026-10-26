# Nonlinear drift and returned-state screen

The nonlinear increment is a mathematically defined internal Euler update, but it fails the practical screen against the matched SwiGLU reference. No candidate is extended and the implemented nonlinearK2 coefficient is not trained as a rescue.

| Task | Split-linear map | Nonlinear map | Nonlinear increment | Fixed-rate MLP | Budget SwiGLU | Full SwiGLU |
|---|---:|---:|---:|---:|---:|---:|
| gaussian_aniso_2d | 0.00374642 | 0.00466227 | 0.00394035 | 0.00397248 | 0.00430438 | 0.00392798 |
| gmm_ring8_2d | 0.02302461 | 0.02814329 | 0.02793461 | 0.03017669 | 0.01518420 | 0.01622070 |
| rotated_product_bimodal_16d | 0.30115736 | 0.30457803 | 0.14813975 | 0.15629325 | 0.09083060 | 0.10823417 |

Values are final5000-step validation oracle MSE. All input-gated operator arms use the same P1/P2, retained first controller, initial surrounding tensors and69282 parameters/67904 MACs at dim2. The nonlinear map/increment pair has identical initial trainable tensors and differs only by adding the internal input to its returned update. The fixed-rate nonlinear increment is exactly a bias-free ordinary SiLU MLP and uses less compute; it is deliberately retained as a generic rival.

Returning the increment materially helps product16 relative to returning the nonlinear map (.30458→.14814). The same-parameter budget SwiGLU remains better at.09083. Ring8 does not improve over the split-linear reference, and its derivative error.91004 remains well above budget SwiGLU.39370. These results fail the registered continuation rule before any high-order training is considered.

The correct nonlinear second coefficient .5*D0*P2[SiLU'(P1x)*(P1*d1)] matches an exact field-JVP, and its small-horizon local error scaling was verified against RK4. This verifies the implementation's internal numerical meaning, not a benefit for the FM objective. All trained arms in this screen wereK1. P1/P2 are logged as nonlinear field factors; their individual spectra do not equal the nonlinear field's Jacobian spectrum.

Both frozen-source audits passed, including common initial-tensor mapping, all saved checkpoints, source/data checks and oracle recomputation. Evidence is under evidence/nonlinear_drift_v1_20261024 and evidence/nonlinear_drift_product16_20261024. Raw outputs remain on Lenovo under the same stage names, and exact source snapshots are mirrored locally.
