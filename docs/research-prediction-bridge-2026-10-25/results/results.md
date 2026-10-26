# Prediction-space bridge: results

All24 registered cells completed. The saved field, derivative and sampling metrics were recomputed; common input streams and initialization controls passed.

| Dataset | Arm | Primary field MSE | Native-time MSE | SW2 |
|---|---|---:|---:|---:|
| gaussian_aniso_2d | ode_k1_v | 0.0335744 | 0.00373556 | 0.11287 |
| gaussian_aniso_2d | ode_k1_x | 0.580471 | 0.00812199 | 0.575256 |
| gaussian_aniso_2d | ode_k2_v | 0.0347834 | 0.0040608 | 0.113004 |
| gaussian_aniso_2d | ode_k2_x | 0.603489 | 0.00911585 | 0.566915 |
| gaussian_aniso_2d | swiglu128_v | 0.0352382 | 0.00484037 | 0.11533 |
| gaussian_aniso_2d | swiglu128_x | 0.576203 | 0.0105592 | 0.588256 |
| gaussian_aniso_2d | swiglu192_v | 0.0370078 | 0.00512012 | 0.123105 |
| gaussian_aniso_2d | swiglu192_x | 0.539965 | 0.011141 | 0.567634 |
| gmm_ring8_2d | ode_k1_v | 0.158008 | 0.0204292 | 0.318 |
| gmm_ring8_2d | ode_k1_x | 0.398881 | 0.0485465 | 0.331156 |
| gmm_ring8_2d | ode_k2_v | 0.153102 | 0.0204696 | 0.305043 |
| gmm_ring8_2d | ode_k2_x | 0.406955 | 0.0493285 | 0.336106 |
| gmm_ring8_2d | swiglu128_v | 0.0900478 | 0.0119927 | 0.312801 |
| gmm_ring8_2d | swiglu128_x | 0.397841 | 0.0492004 | 0.327491 |
| gmm_ring8_2d | swiglu192_v | 0.0831709 | 0.0119598 | 0.311403 |
| gmm_ring8_2d | swiglu192_x | 0.406987 | 0.0495969 | 0.326978 |
| rotated_product_bimodal_16d | ode_k1_v | 0.332418 | 0.126155 | 0.160418 |
| rotated_product_bimodal_16d | ode_k1_x | 0.55271 | 0.136987 | 0.174091 |
| rotated_product_bimodal_16d | ode_k2_v | 0.375083 | 0.145029 | 0.174358 |
| rotated_product_bimodal_16d | ode_k2_x | 0.556444 | 0.158232 | 0.152357 |
| rotated_product_bimodal_16d | swiglu128_v | 0.202016 | 0.0577546 | 0.144881 |
| rotated_product_bimodal_16d | swiglu128_x | 0.328133 | 0.0302413 | 0.127078 |
| rotated_product_bimodal_16d | swiglu192_v | 0.199982 | 0.0582203 | 0.152864 |
| rotated_product_bimodal_16d | swiglu192_x | 0.36364 | 0.0280144 | 0.141466 |

Decision: no ODE arm passes the predeclared quality/transfer/cost screen; stop this fixed setting.

This comparison concerns tiny synthetic backbones under the declared common JiT-style prediction/time recipe. It does not establish full JiT image-generation performance or attribute any generic prediction-space effect specifically to an ODE block. Concurrent training wall intervals are retained as records, not isolated latency measurements.
