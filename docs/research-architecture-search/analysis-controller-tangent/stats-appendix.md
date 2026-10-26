# Statistical appendix: frozen controller tangent diagnosis

All outputs are descriptive, conditional on the saved weights, anchors and parameter coordinates. There are no significance tests, confidence intervals or independent time replicates. This is an exposed-anchor diagnostic using privileged population-oracle labels, not a fresh generalization estimate or a finite nonlinear update. Every predeclared checkpoint and ridge value is retained.

## Estimands and normalization

For residual e = oracle − prediction, the native parameter kernel is K = J Jᵀ. Fit anchors comprise 16 paths at five times (80 anchors); output-coordinate counts are 160 for Ring8 and 1,280 for Product16. Let c = trace(K_fit,fit)/n. The normalized blocks are K/c. For each fixed λ, the dual coefficient is `(K_fit,fit/c + λI)^−1 e_fit`; multiplying it by the corresponding normalized cross-kernel gives the predicted evaluation correction. MSE averages over anchors and output coordinates. Each residual share divides by the original MSE on the same split.

The recorded native update norm is `sqrt(alphaᵀ (K_fit,fit/c) alpha / c)`, the Euclidean norm of the implied native parameter change. It is not a fractional parameter change or proof that linearization is valid at that finite displacement. The receiver-RMS counterfactual has only a feature-space norm. It is not a fixed parameter-coordinate change. Participation rank is `(sum eigenvalues)^2 / sum(eigenvalues^2)`, not the number of nonzero eigenvalues.

The free-gate floor uses all 160 anchors and unbounded independent controls. Numerical rank keeps singular values above `max(1e−12, largest singular value × 1e−12)`. Its residual share is the ratio of summed squared projected residuals to summed squared original residuals. Tiny negative eigenvalues are checked against a relative numerical tolerance and clipped to zero in the frozen ridge routine; no case has zero trace. Per-layer controller traces use all 160 anchors, whereas component mean diagonals use calibration anchors only; do not sum the former to reproduce the latter.

Complete FFN excludes non-FFN parameters. Original ODE/SwiGLU FiLM initializations were unpaired, so cross-family references are not an isolated FFN causal comparison. Time rows reuse the same 16 paths. All λ=.1 examples in the report were fixed illustrations, not selected minima.

## Case and kernel tables

### Ring8

| Model | Update | All-anchor MSE | Gate rank min | Gate rank max | Free-gate floor | Controller trace share |
| --- | --- | --- | --- | --- | --- | --- |
| scalar_mul | 1000 | 0.163865 | 2 | 2 | 1.2076e-31 | 0.00697605 |
| scalar_mul | 5000 | 0.0221769 | 2 | 2 | 1.51515e-31 | 0.0150065 |
| scalar_add | 1000 | 0.162419 | 2 | 2 | 1.38305e-31 | 0.00150806 |
| scalar_add | 5000 | 0.0199095 | 2 | 2 | 1.20938e-31 | 0.0147399 |
| vector_mul | 1000 | 0.163472 | 2 | 2 | 8.3554e-32 | 0.00938889 |
| vector_mul | 5000 | 0.018059 | 2 | 2 | 1.37387e-31 | 0.0773303 |
| vector_add | 1000 | 0.164283 | 2 | 2 | 1.21338e-31 | 0.00647352 |
| vector_add | 5000 | 0.0158931 | 2 | 2 | 1.63227e-31 | 0.046098 |
| swiglu_budget | 1000 | 0.13677 | — | — | — | — |
| swiglu_budget | 5000 | 0.015761 | — | — | — | — |
| swiglu | 1000 | 0.136724 | — | — | — | — |
| swiglu | 5000 | 0.0136671 | — | — | — | — |

### Product16

| Model | Update | All-anchor MSE | Gate rank min | Gate rank max | Free-gate floor | Controller trace share |
| --- | --- | --- | --- | --- | --- | --- |
| scalar_mul | 1000 | 0.544294 | 6 | 6 | 0.614015 | 0.000236625 |
| scalar_mul | 5000 | 0.279396 | 6 | 6 | 0.699591 | 0.0178169 |
| scalar_add | 1000 | 0.56324 | 6 | 6 | 0.602406 | 0.000344975 |
| scalar_add | 5000 | 0.30502 | 6 | 6 | 0.703586 | 0.0152073 |
| vector_mul | 1000 | 0.554647 | 16 | 16 | 1.20917e-30 | 0.00899667 |
| vector_mul | 5000 | 0.24739 | 16 | 16 | 1.04434e-30 | 0.0311849 |
| vector_add | 1000 | 0.559713 | 16 | 16 | 1.10927e-30 | 0.0128613 |
| vector_add | 5000 | 0.166852 | 16 | 16 | 9.4118e-31 | 0.124212 |
| swiglu_budget | 1000 | 0.279533 | — | — | — | — |
| swiglu_budget | 5000 | 0.0923534 | — | — | — | — |
| swiglu | 1000 | 0.292767 | — | — | — | — |
| swiglu | 5000 | 0.104452 | — | — | — | — |

| Task | Model | Update | Component | Native mean diagonal | Participation rank | Min normalized eigenvalue | Max normalized eigenvalue |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Product16 | scalar_mul | 1000 | full_ffn | 37.179 | 50.9422 | 0.000967344 | 65.5862 |
| Product16 | scalar_mul | 1000 | controller | 0.00879747 | 10.9078 | -4.0301e-14 | 231.291 |
| Product16 | scalar_mul | 1000 | controller_value_rms_equalized | 0.0094471 | 9.74675 | -2.81857e-14 | 311.094 |
| Product16 | scalar_mul | 1000 | projection_P | 37.1702 | 50.9259 | 0.000958807 | 65.6005 |
| Product16 | scalar_mul | 5000 | full_ffn | 378.666 | 38.9688 | 0.000554944 | 104.841 |
| Product16 | scalar_mul | 5000 | controller | 6.74666 | 3.73915 | -3.99861e-14 | 499.336 |
| Product16 | scalar_mul | 5000 | controller_value_rms_equalized | 1.07791 | 3.66493 | -5.64919e-14 | 571.812 |
| Product16 | scalar_mul | 5000 | projection_P | 371.919 | 39.4866 | 0.000559859 | 105.51 |
| Product16 | scalar_add | 1000 | full_ffn | 19.771 | 46.8589 | 0.000356409 | 69.9186 |
| Product16 | scalar_add | 1000 | controller | 0.0068205 | 6.81941 | -3.35653e-14 | 369.877 |
| Product16 | scalar_add | 1000 | controller_value_rms_equalized | 0.0031262 | 7.22105 | -4.65445e-14 | 339.309 |
| Product16 | scalar_add | 1000 | projection_P | 19.7642 | 46.8467 | 0.000355593 | 69.9387 |
| Product16 | scalar_add | 5000 | full_ffn | 307.047 | 34.9104 | 0.000435284 | 119.178 |
| Product16 | scalar_add | 5000 | controller | 4.66936 | 4.08593 | -3.91606e-14 | 533.889 |
| Product16 | scalar_add | 5000 | controller_value_rms_equalized | 0.143083 | 5.07427 | -7.23578e-14 | 406.396 |
| Product16 | scalar_add | 5000 | projection_P | 302.378 | 35.1613 | 0.000427332 | 119.688 |
| Product16 | vector_mul | 1000 | full_ffn | 27.0613 | 44.701 | 0.000731573 | 75.5143 |
| Product16 | vector_mul | 1000 | controller | 0.243462 | 39.166 | 0.000421546 | 81.8096 |
| Product16 | vector_mul | 1000 | controller_value_rms_equalized | 0.0880456 | 33.8063 | 8.89263e-05 | 90.6517 |
| Product16 | vector_mul | 1000 | projection_P | 26.8179 | 44.6442 | 0.000716432 | 75.7229 |
| Product16 | vector_mul | 5000 | full_ffn | 445.154 | 24.9255 | 0.000519891 | 147.347 |
| Product16 | vector_mul | 5000 | controller | 13.8821 | 12.3961 | 0.000149629 | 302.626 |
| Product16 | vector_mul | 5000 | controller_value_rms_equalized | 0.766712 | 20.1116 | 0.000111434 | 195.736 |
| Product16 | vector_mul | 5000 | projection_P | 431.272 | 24.6184 | 0.000481952 | 150.848 |
| Product16 | vector_add | 1000 | full_ffn | 19.5123 | 39.5849 | 0.000152021 | 76.7535 |
| Product16 | vector_add | 1000 | controller | 0.250954 | 27.9278 | 1.6526e-05 | 101.654 |
| Product16 | vector_add | 1000 | controller_value_rms_equalized | 0.0630164 | 33.1833 | 1.9029e-06 | 97.6718 |
| Product16 | vector_add | 1000 | projection_P | 19.2614 | 39.6131 | 0.000147167 | 76.8472 |
| Product16 | vector_add | 5000 | full_ffn | 553.709 | 33.3395 | 0.000277574 | 112.322 |
| Product16 | vector_add | 5000 | controller | 68.7774 | 18.8129 | 2.91216e-05 | 152.72 |
| Product16 | vector_add | 5000 | controller_value_rms_equalized | 1.08996 | 22.4654 | 4.9628e-05 | 156.524 |
| Product16 | vector_add | 5000 | projection_P | 484.932 | 33.1435 | 0.000252993 | 114.427 |
| Product16 | swiglu_budget | 1000 | full_ffn | 440.998 | 77.4965 | 0.00125681 | 68.6126 |
| Product16 | swiglu_budget | 5000 | full_ffn | 870.795 | 84.6999 | 0.00107215 | 69.74 |
| Product16 | swiglu | 1000 | full_ffn | 350.54 | 88.6717 | 0.00182782 | 59.2587 |
| Product16 | swiglu | 5000 | full_ffn | 656.625 | 88.0862 | 0.00133609 | 56.2108 |
| Ring8 | scalar_mul | 1000 | full_ffn | 19.995 | 10.6154 | 7.57879e-09 | 32.6289 |
| Ring8 | scalar_mul | 1000 | controller | 0.139486 | 3.04448 | 5.557e-09 | 86.375 |
| Ring8 | scalar_mul | 1000 | controller_value_rms_equalized | 0.081345 | 2.47264 | 2.53068e-09 | 98.4196 |
| Ring8 | scalar_mul | 1000 | projection_P | 19.8555 | 10.6667 | 7.15368e-09 | 32.323 |
| Ring8 | scalar_mul | 5000 | full_ffn | 190.425 | 11.9997 | 3.70315e-09 | 23.5775 |
| Ring8 | scalar_mul | 5000 | controller | 2.8576 | 5.15854 | 2.96475e-10 | 52.9148 |
| Ring8 | scalar_mul | 5000 | controller_value_rms_equalized | 0.369776 | 3.58505 | 5.26889e-10 | 79.2222 |
| Ring8 | scalar_mul | 5000 | projection_P | 187.567 | 11.9177 | 3.21899e-09 | 23.8588 |
| Ring8 | scalar_add | 1000 | full_ffn | 13.5674 | 10.7038 | 2.22436e-09 | 31.9451 |
| Ring8 | scalar_add | 1000 | controller | 0.0204604 | 3.07667 | 2.40242e-10 | 86.5779 |
| Ring8 | scalar_add | 1000 | controller_value_rms_equalized | 0.0117937 | 2.49672 | 4.98908e-11 | 97.577 |
| Ring8 | scalar_add | 1000 | projection_P | 13.5469 | 10.7072 | 2.1884e-09 | 31.9339 |
| Ring8 | scalar_add | 5000 | full_ffn | 110.817 | 10.2112 | 3.90537e-10 | 32.372 |
| Ring8 | scalar_add | 5000 | controller | 1.63343 | 7.70565 | 4.78404e-13 | 39.755 |
| Ring8 | scalar_add | 5000 | controller_value_rms_equalized | 0.157963 | 7.81403 | 7.38959e-13 | 37.4526 |
| Ring8 | scalar_add | 5000 | projection_P | 109.184 | 10.1096 | 3.927e-10 | 32.6504 |
| Ring8 | vector_mul | 1000 | full_ffn | 16.5903 | 10.6002 | 3.80816e-09 | 33.3118 |
| Ring8 | vector_mul | 1000 | controller | 0.155765 | 6.44854 | 2.22171e-08 | 55.5957 |
| Ring8 | vector_mul | 1000 | controller_value_rms_equalized | 0.0569909 | 6.35912 | 1.3859e-08 | 55.6009 |
| Ring8 | vector_mul | 1000 | projection_P | 16.4346 | 10.6184 | 3.20726e-09 | 33.1522 |
| Ring8 | vector_mul | 5000 | full_ffn | 140.463 | 11.7069 | 1.32783e-08 | 25.4072 |
| Ring8 | vector_mul | 5000 | controller | 10.862 | 7.41298 | 3.05838e-08 | 38.412 |
| Ring8 | vector_mul | 5000 | controller_value_rms_equalized | 0.576107 | 8.52056 | 1.41679e-07 | 37.5738 |
| Ring8 | vector_mul | 5000 | projection_P | 129.601 | 11.6955 | 1.67233e-09 | 25.3573 |
| Ring8 | vector_add | 1000 | full_ffn | 14.0926 | 11.0717 | 5.30696e-09 | 31.051 |
| Ring8 | vector_add | 1000 | controller | 0.0912289 | 6.69404 | 6.10465e-08 | 45.719 |
| Ring8 | vector_add | 1000 | controller_value_rms_equalized | 0.0271587 | 9.0716 | 2.27431e-08 | 30.8595 |
| Ring8 | vector_add | 1000 | projection_P | 14.0014 | 11.0369 | 2.94561e-09 | 31.1331 |
| Ring8 | vector_add | 5000 | full_ffn | 98.6972 | 11.5753 | 3.45161e-09 | 25.5313 |
| Ring8 | vector_add | 5000 | controller | 4.54975 | 8.71045 | 1.66921e-08 | 33.3473 |
| Ring8 | vector_add | 5000 | controller_value_rms_equalized | 0.421159 | 9.02739 | 6.46455e-08 | 35.0004 |
| Ring8 | vector_add | 5000 | projection_P | 94.1475 | 11.5515 | 4.08269e-10 | 25.7687 |
| Ring8 | swiglu_budget | 1000 | full_ffn | 76.1273 | 13.243 | 1.07924e-07 | 26.9286 |
| Ring8 | swiglu_budget | 5000 | full_ffn | 121.65 | 16.8313 | 6.32392e-08 | 21.6719 |
| Ring8 | swiglu | 1000 | full_ffn | 80.0788 | 15.7316 | 1.97206e-07 | 23.1773 |
| Ring8 | swiglu | 5000 | full_ffn | 108.834 | 20.5043 | 1.1516e-07 | 17.3665 |

## Every ridge row

Each table states the original MSE for both splits. Norms are native parameter norms except for the explicitly named RMS-equalized feature counterfactual. Exact values and per-time MSEs are in the CSV exports.

### Ring8 / scalar_mul / update 1000

Original calibration MSE: 0.170246098; original evaluation MSE: 0.157484066.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 0.00250004 | 0.704615 | 0.0146849 | 4.4742 | 15.7399 | 0.721599 |
| full_ffn | 0.001 | 0.0097055 | 0.544275 | 0.0570086 | 3.45607 | 8.50775 | 0.621293 |
| full_ffn | 0.01 | 0.0318233 | 0.311028 | 0.186925 | 1.97498 | 3.49383 | 0.417821 |
| full_ffn | 0.1 | 0.0692199 | 0.227145 | 0.406587 | 1.44233 | 1.24241 | 0.267747 |
| full_ffn | 1 | 0.113485 | 0.190791 | 0.666592 | 1.21149 | 0.352143 | 0.15355 |
| full_ffn | 10 | 0.14836 | 0.169309 | 0.871444 | 1.07509 | 0.0813573 | 0.0758164 |
| controller | 0.0001 | 0.00987281 | 0.668468 | 0.0579914 | 4.24467 | 358.357 | 0.743641 |
| controller | 0.001 | 0.0375774 | 0.307372 | 0.220724 | 1.95177 | 136.195 | 0.417081 |
| controller | 0.01 | 0.0760164 | 0.242073 | 0.446509 | 1.53713 | 45.9702 | 0.268135 |
| controller | 0.1 | 0.118344 | 0.212593 | 0.695134 | 1.34993 | 12.3063 | 0.19342 |
| controller | 1 | 0.147795 | 0.172532 | 0.868127 | 1.09555 | 2.95688 | 0.106419 |
| controller | 10 | 0.164264 | 0.157845 | 0.964863 | 1.00229 | 0.541638 | 0.0323128 |
| controller_value_rms_equalized | 0.0001 | 0.0105845 | 0.668341 | 0.062172 | 4.24387 | 461.792 | 0.731651 |
| controller_value_rms_equalized | 0.001 | 0.0374879 | 0.310447 | 0.220199 | 1.97129 | 174.091 | 0.414671 |
| controller_value_rms_equalized | 0.01 | 0.0737666 | 0.238806 | 0.433294 | 1.51638 | 57.3555 | 0.269356 |
| controller_value_rms_equalized | 0.1 | 0.112257 | 0.209945 | 0.659381 | 1.33312 | 16.6041 | 0.185154 |
| controller_value_rms_equalized | 1 | 0.144398 | 0.17509 | 0.848171 | 1.1118 | 4.1936 | 0.102082 |
| controller_value_rms_equalized | 10 | 0.163651 | 0.15892 | 0.961259 | 1.00912 | 0.753296 | 0.0303529 |
| projection_P | 0.0001 | 0.00250473 | 0.706686 | 0.0147124 | 4.48735 | 15.8153 | 0.723112 |
| projection_P | 0.001 | 0.00972968 | 0.544953 | 0.0571507 | 3.46037 | 8.54193 | 0.621996 |
| projection_P | 0.01 | 0.0318767 | 0.311433 | 0.187239 | 1.97755 | 3.50718 | 0.418069 |
| projection_P | 0.1 | 0.0692768 | 0.227358 | 0.406922 | 1.44369 | 1.24558 | 0.26792 |
| projection_P | 1 | 0.113455 | 0.190875 | 0.666419 | 1.21202 | 0.353237 | 0.153641 |
| projection_P | 10 | 0.148312 | 0.169374 | 0.871161 | 1.0755 | 0.0816879 | 0.0760316 |

### Ring8 / scalar_add / update 1000

Original calibration MSE: 0.168543652; original evaluation MSE: 0.156293695.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 0.00270748 | 0.734325 | 0.016064 | 4.69837 | 19.5759 | 0.757411 |
| full_ffn | 0.001 | 0.0101523 | 0.52503 | 0.0602354 | 3.35925 | 10.6215 | 0.611477 |
| full_ffn | 0.01 | 0.033752 | 0.296649 | 0.200257 | 1.89802 | 4.27428 | 0.407884 |
| full_ffn | 0.1 | 0.0710804 | 0.221721 | 0.421733 | 1.41862 | 1.46695 | 0.26557 |
| full_ffn | 1 | 0.113161 | 0.187919 | 0.671402 | 1.20235 | 0.42435 | 0.148371 |
| full_ffn | 10 | 0.147723 | 0.169026 | 0.876466 | 1.08146 | 0.0967789 | 0.0751899 |
| controller | 0.0001 | 0.00458161 | 0.744533 | 0.0271835 | 4.76368 | 778.843 | 0.797218 |
| controller | 0.001 | 0.0241615 | 0.410577 | 0.143355 | 2.62696 | 361.75 | 0.540457 |
| controller | 0.01 | 0.0643724 | 0.240954 | 0.381933 | 1.54168 | 120.819 | 0.319408 |
| controller | 0.1 | 0.107872 | 0.196442 | 0.640024 | 1.25688 | 34.6004 | 0.196445 |
| controller | 1 | 0.142411 | 0.17063 | 0.844951 | 1.09173 | 8.39168 | 0.0992295 |
| controller | 10 | 0.161532 | 0.158247 | 0.958396 | 1.01249 | 1.51578 | 0.0335694 |
| controller_value_rms_equalized | 0.0001 | 0.0067395 | 0.627095 | 0.0399867 | 4.01228 | 1139.15 | 0.709683 |
| controller_value_rms_equalized | 0.001 | 0.0311236 | 0.373559 | 0.184662 | 2.39011 | 475.182 | 0.506088 |
| controller_value_rms_equalized | 0.01 | 0.0701158 | 0.2388 | 0.416009 | 1.52789 | 154.952 | 0.312403 |
| controller_value_rms_equalized | 0.1 | 0.111786 | 0.190211 | 0.663244 | 1.21701 | 44.613 | 0.171889 |
| controller_value_rms_equalized | 1 | 0.144665 | 0.169107 | 0.858326 | 1.08198 | 10.7563 | 0.0830782 |
| controller_value_rms_equalized | 10 | 0.16271 | 0.157936 | 0.965385 | 1.01051 | 1.83109 | 0.0285548 |
| projection_P | 0.0001 | 0.00273671 | 0.73634 | 0.0162374 | 4.71126 | 19.5898 | 0.758517 |
| projection_P | 0.001 | 0.01018 | 0.525654 | 0.0604 | 3.36324 | 10.6278 | 0.612012 |
| projection_P | 0.01 | 0.033772 | 0.296889 | 0.200376 | 1.89956 | 4.27692 | 0.408141 |
| projection_P | 0.1 | 0.0710913 | 0.22179 | 0.421797 | 1.41906 | 1.46779 | 0.265653 |
| projection_P | 1 | 0.113155 | 0.187919 | 0.67137 | 1.20235 | 0.424641 | 0.148409 |
| projection_P | 10 | 0.147714 | 0.169036 | 0.876414 | 1.08153 | 0.0968623 | 0.0752435 |

### Ring8 / vector_mul / update 1000

Original calibration MSE: 0.169316854; original evaluation MSE: 0.157627746.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 0.00270061 | 0.782112 | 0.01595 | 4.96177 | 18.2654 | 0.781727 |
| full_ffn | 0.001 | 0.0107239 | 0.553364 | 0.0633362 | 3.51058 | 9.47089 | 0.635163 |
| full_ffn | 0.01 | 0.03358 | 0.295846 | 0.198327 | 1.87686 | 3.85888 | 0.405087 |
| full_ffn | 0.1 | 0.0710167 | 0.2209 | 0.419431 | 1.40141 | 1.35001 | 0.259467 |
| full_ffn | 1 | 0.114856 | 0.186882 | 0.678349 | 1.18559 | 0.386496 | 0.146224 |
| full_ffn | 10 | 0.149428 | 0.168136 | 0.882535 | 1.06666 | 0.0861037 | 0.0726424 |
| controller | 0.0001 | 0.00163051 | 0.396395 | 0.00962993 | 2.51476 | 182.121 | 0.535375 |
| controller | 0.001 | 0.00923758 | 0.349721 | 0.054558 | 2.21865 | 101.834 | 0.471326 |
| controller | 0.01 | 0.0341871 | 0.252037 | 0.201912 | 1.59894 | 39.6065 | 0.348748 |
| controller | 0.1 | 0.0708269 | 0.2138 | 0.41831 | 1.35636 | 13.6509 | 0.247586 |
| controller | 1 | 0.112139 | 0.19739 | 0.662303 | 1.25226 | 3.85433 | 0.171597 |
| controller | 10 | 0.145607 | 0.171434 | 0.859966 | 1.08759 | 0.976377 | 0.0788655 |
| controller_value_rms_equalized | 0.0001 | 0.00139401 | 0.40242 | 0.00823315 | 2.55298 | 281.294 | 0.538658 |
| controller_value_rms_equalized | 0.001 | 0.00768709 | 0.377717 | 0.0454006 | 2.39626 | 160.405 | 0.498265 |
| controller_value_rms_equalized | 0.01 | 0.0308801 | 0.269854 | 0.182381 | 1.71197 | 64.9141 | 0.369788 |
| controller_value_rms_equalized | 0.1 | 0.0667017 | 0.217555 | 0.393946 | 1.38018 | 22.9098 | 0.254429 |
| controller_value_rms_equalized | 1 | 0.109948 | 0.1946 | 0.649365 | 1.23455 | 6.68972 | 0.163723 |
| controller_value_rms_equalized | 10 | 0.146687 | 0.169458 | 0.866348 | 1.07505 | 1.5984 | 0.0700293 |
| projection_P | 0.0001 | 0.00276848 | 0.792696 | 0.0163509 | 5.02891 | 18.4125 | 0.788479 |
| projection_P | 0.001 | 0.0108466 | 0.557983 | 0.0640611 | 3.53988 | 9.52518 | 0.639409 |
| projection_P | 0.01 | 0.033742 | 0.297219 | 0.199283 | 1.88557 | 3.8765 | 0.407082 |
| projection_P | 0.1 | 0.0711637 | 0.220953 | 0.420299 | 1.40174 | 1.35568 | 0.259704 |
| projection_P | 1 | 0.11496 | 0.186749 | 0.678964 | 1.18475 | 0.388072 | 0.145933 |
| projection_P | 10 | 0.149476 | 0.168112 | 0.882818 | 1.06651 | 0.0863881 | 0.0726153 |

### Ring8 / vector_add / update 1000

Original calibration MSE: 0.170739469; original evaluation MSE: 0.157826372.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 0.0030799 | 0.788788 | 0.0180386 | 4.99782 | 19.2505 | 0.790053 |
| full_ffn | 0.001 | 0.0104593 | 0.540608 | 0.0612588 | 3.42533 | 10.3216 | 0.629528 |
| full_ffn | 0.01 | 0.0336005 | 0.306485 | 0.196794 | 1.94191 | 4.21443 | 0.422128 |
| full_ffn | 0.1 | 0.0715421 | 0.224639 | 0.419013 | 1.42333 | 1.45481 | 0.267044 |
| full_ffn | 1 | 0.114513 | 0.189094 | 0.670691 | 1.19811 | 0.42389 | 0.150196 |
| full_ffn | 10 | 0.150338 | 0.169058 | 0.880509 | 1.07116 | 0.0951 | 0.0719666 |
| controller | 0.0001 | 0.0020919 | 0.645433 | 0.012252 | 4.08951 | 253.73 | 0.715401 |
| controller | 0.001 | 0.0109418 | 0.497298 | 0.0640846 | 3.15092 | 138.264 | 0.596511 |
| controller | 0.01 | 0.0369758 | 0.287436 | 0.216563 | 1.82122 | 53.6734 | 0.389767 |
| controller | 0.1 | 0.0776804 | 0.203874 | 0.454965 | 1.29176 | 18.8597 | 0.230327 |
| controller | 1 | 0.12409 | 0.1824 | 0.726782 | 1.1557 | 4.86637 | 0.137096 |
| controller | 10 | 0.153931 | 0.165958 | 0.901555 | 1.05152 | 1.0711 | 0.0637174 |
| controller_value_rms_equalized | 0.0001 | 0.00147575 | 0.596525 | 0.0086433 | 3.77963 | 424.464 | 0.678549 |
| controller_value_rms_equalized | 0.001 | 0.00844758 | 0.512468 | 0.0494764 | 3.24704 | 241.56 | 0.604504 |
| controller_value_rms_equalized | 0.01 | 0.0329833 | 0.301738 | 0.193179 | 1.91184 | 95.3113 | 0.409156 |
| controller_value_rms_equalized | 0.1 | 0.070328 | 0.209817 | 0.411902 | 1.32942 | 34.3753 | 0.250751 |
| controller_value_rms_equalized | 1 | 0.117327 | 0.183628 | 0.687169 | 1.16348 | 9.45459 | 0.138781 |
| controller_value_rms_equalized | 10 | 0.150762 | 0.168686 | 0.882992 | 1.06881 | 2.10699 | 0.0721618 |
| projection_P | 0.0001 | 0.00312321 | 0.795115 | 0.0182922 | 5.03791 | 19.3214 | 0.793726 |
| projection_P | 0.001 | 0.0105069 | 0.540055 | 0.0615374 | 3.42183 | 10.3585 | 0.629083 |
| projection_P | 0.01 | 0.0336677 | 0.306686 | 0.197188 | 1.94319 | 4.23094 | 0.422074 |
| projection_P | 0.1 | 0.0716478 | 0.224807 | 0.419632 | 1.42439 | 1.4579 | 0.267043 |
| projection_P | 1 | 0.114512 | 0.189144 | 0.67068 | 1.19843 | 0.42509 | 0.150294 |
| projection_P | 10 | 0.150322 | 0.169074 | 0.880415 | 1.07126 | 0.095448 | 0.0720102 |

### Ring8 / swiglu_budget / update 1000

Original calibration MSE: 0.137406899; original evaluation MSE: 0.136132781.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 0.000643598 | 0.256006 | 0.00468388 | 1.88056 | 5.84802 | 0.444898 |
| full_ffn | 0.001 | 0.0044498 | 0.220605 | 0.0323841 | 1.62052 | 2.98213 | 0.384701 |
| full_ffn | 0.01 | 0.0147365 | 0.193312 | 0.107247 | 1.42003 | 1.32286 | 0.312723 |
| full_ffn | 0.1 | 0.0348313 | 0.192178 | 0.25349 | 1.4117 | 0.499424 | 0.260595 |
| full_ffn | 1 | 0.0618193 | 0.179298 | 0.4499 | 1.31708 | 0.179426 | 0.204004 |
| full_ffn | 10 | 0.100663 | 0.145516 | 0.732589 | 1.06892 | 0.0547856 | 0.0815051 |

### Ring8 / swiglu / update 1000

Original calibration MSE: 0.132383783; original evaluation MSE: 0.141064045.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 0.00149646 | 0.275514 | 0.011304 | 1.95311 | 5.40807 | 0.454496 |
| full_ffn | 0.001 | 0.0044482 | 0.185502 | 0.0336008 | 1.31502 | 2.44378 | 0.329857 |
| full_ffn | 0.01 | 0.0119665 | 0.162581 | 0.0903926 | 1.15253 | 1.26206 | 0.272346 |
| full_ffn | 0.1 | 0.0309897 | 0.161702 | 0.23409 | 1.1463 | 0.47652 | 0.223669 |
| full_ffn | 1 | 0.057537 | 0.15837 | 0.434623 | 1.12268 | 0.1788 | 0.169079 |
| full_ffn | 10 | 0.0971741 | 0.143219 | 0.734033 | 1.01527 | 0.0522167 | 0.0692127 |

### Ring8 / scalar_mul / update 5000

Original calibration MSE: 0.0204794205; original evaluation MSE: 0.0238744329.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 6.66979e-06 | 0.0114713 | 0.000325682 | 0.480484 | 0.544351 | 0.136731 |
| full_ffn | 0.001 | 7.38821e-05 | 0.00993048 | 0.00360763 | 0.415946 | 0.376549 | 0.132061 |
| full_ffn | 0.01 | 0.000447802 | 0.00927466 | 0.021866 | 0.388477 | 0.227719 | 0.125744 |
| full_ffn | 0.1 | 0.00190513 | 0.00946565 | 0.0930266 | 0.396477 | 0.11751 | 0.108237 |
| full_ffn | 1 | 0.00601973 | 0.0124803 | 0.29394 | 0.522745 | 0.0433651 | 0.0817719 |
| full_ffn | 10 | 0.0111798 | 0.0169902 | 0.545902 | 0.711647 | 0.0153253 | 0.0473677 |
| controller | 0.0001 | 0.000346134 | 0.0340613 | 0.0169016 | 1.42668 | 18.2039 | 0.20546 |
| controller | 0.001 | 0.00177876 | 0.0268812 | 0.0868558 | 1.12594 | 8.44036 | 0.182312 |
| controller | 0.01 | 0.00488075 | 0.0139365 | 0.238325 | 0.583742 | 3.21847 | 0.137914 |
| controller | 0.1 | 0.0092066 | 0.0130521 | 0.449554 | 0.5467 | 1.06987 | 0.100493 |
| controller | 1 | 0.0140022 | 0.0173582 | 0.683718 | 0.727063 | 0.280773 | 0.0599438 |
| controller | 10 | 0.0170622 | 0.0207277 | 0.833137 | 0.868197 | 0.0778144 | 0.0337859 |
| controller_value_rms_equalized | 0.0001 | 0.00022684 | 0.040205 | 0.0110765 | 1.68402 | 45.202 | 0.229497 |
| controller_value_rms_equalized | 0.001 | 0.00135311 | 0.0248787 | 0.0660715 | 1.04207 | 21.941 | 0.191756 |
| controller_value_rms_equalized | 0.01 | 0.00411982 | 0.0153615 | 0.201169 | 0.64343 | 8.4512 | 0.147797 |
| controller_value_rms_equalized | 0.1 | 0.00790018 | 0.0133504 | 0.385762 | 0.559192 | 2.93026 | 0.106416 |
| controller_value_rms_equalized | 1 | 0.0127096 | 0.0161551 | 0.620603 | 0.676668 | 0.919897 | 0.0627918 |
| controller_value_rms_equalized | 10 | 0.0171387 | 0.0206958 | 0.836873 | 0.866861 | 0.232206 | 0.0282367 |
| projection_P | 0.0001 | 7.03505e-06 | 0.0112683 | 0.000343518 | 0.471981 | 0.553774 | 0.140532 |
| projection_P | 0.001 | 7.57264e-05 | 0.009681 | 0.00369769 | 0.405497 | 0.381699 | 0.134464 |
| projection_P | 0.01 | 0.000455351 | 0.00917418 | 0.0222346 | 0.384268 | 0.230095 | 0.12742 |
| projection_P | 0.1 | 0.0019225 | 0.00941626 | 0.0938746 | 0.394408 | 0.118548 | 0.109052 |
| projection_P | 1 | 0.00604251 | 0.0124727 | 0.295053 | 0.52243 | 0.0434733 | 0.0823568 |
| projection_P | 10 | 0.0111492 | 0.0169707 | 0.544409 | 0.710831 | 0.0154425 | 0.04751 |

### Ring8 / scalar_add / update 5000

Original calibration MSE: 0.0206624369; original evaluation MSE: 0.0191565352.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 5.94189e-06 | 0.00939887 | 0.00028757 | 0.490635 | 0.621654 | 0.127865 |
| full_ffn | 0.001 | 5.29032e-05 | 0.00861571 | 0.00256036 | 0.449753 | 0.456988 | 0.122996 |
| full_ffn | 0.01 | 0.000387843 | 0.00769383 | 0.0187704 | 0.401629 | 0.27218 | 0.111312 |
| full_ffn | 0.1 | 0.00162469 | 0.00808748 | 0.0786302 | 0.422179 | 0.125073 | 0.102177 |
| full_ffn | 1 | 0.00419453 | 0.00985939 | 0.203003 | 0.514675 | 0.0495755 | 0.0910646 |
| full_ffn | 10 | 0.00814542 | 0.0119751 | 0.394214 | 0.625119 | 0.0220521 | 0.0636626 |
| controller | 0.0001 | 0.000146384 | 0.0338823 | 0.00708453 | 1.76871 | 13.6607 | 0.226444 |
| controller | 0.001 | 0.000591268 | 0.0219079 | 0.0286156 | 1.14362 | 6.82342 | 0.180376 |
| controller | 0.01 | 0.00173743 | 0.0132611 | 0.0840866 | 0.692251 | 2.98835 | 0.131455 |
| controller | 0.1 | 0.00392295 | 0.0122647 | 0.189859 | 0.640233 | 1.10359 | 0.109535 |
| controller | 1 | 0.00668139 | 0.0130099 | 0.323359 | 0.679137 | 0.437422 | 0.100304 |
| controller | 10 | 0.0116606 | 0.0135085 | 0.564337 | 0.705166 | 0.164535 | 0.0631489 |
| controller_value_rms_equalized | 0.0001 | 0.000109613 | 0.0196365 | 0.00530493 | 1.02505 | 39.2887 | 0.182265 |
| controller_value_rms_equalized | 0.001 | 0.000458921 | 0.0189975 | 0.0222104 | 0.991696 | 20.1096 | 0.171004 |
| controller_value_rms_equalized | 0.01 | 0.00141679 | 0.0141608 | 0.0685683 | 0.739217 | 9.35036 | 0.136793 |
| controller_value_rms_equalized | 0.1 | 0.00345216 | 0.0121882 | 0.167074 | 0.636244 | 3.97893 | 0.115019 |
| controller_value_rms_equalized | 1 | 0.00721579 | 0.0126223 | 0.349223 | 0.658905 | 1.42995 | 0.106468 |
| controller_value_rms_equalized | 10 | 0.011986 | 0.0122266 | 0.580084 | 0.638249 | 0.510055 | 0.0745407 |
| projection_P | 0.0001 | 5.97545e-06 | 0.00955289 | 0.000289194 | 0.498675 | 0.63485 | 0.129666 |
| projection_P | 0.001 | 5.51184e-05 | 0.00876106 | 0.00266757 | 0.457341 | 0.463979 | 0.124689 |
| projection_P | 0.01 | 0.000395389 | 0.00770675 | 0.0191356 | 0.402304 | 0.274504 | 0.112069 |
| projection_P | 0.1 | 0.00163535 | 0.00808637 | 0.0791461 | 0.42212 | 0.125735 | 0.102477 |
| projection_P | 1 | 0.0041926 | 0.00985652 | 0.202909 | 0.514525 | 0.0498658 | 0.0910948 |
| projection_P | 10 | 0.00812713 | 0.0119719 | 0.393329 | 0.624949 | 0.0222081 | 0.0636913 |

### Ring8 / vector_mul / update 5000

Original calibration MSE: 0.0194440955; original evaluation MSE: 0.0166738359.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 7.4219e-06 | 0.0113796 | 0.000381705 | 0.682483 | 0.608452 | 0.130331 |
| full_ffn | 0.001 | 7.3455e-05 | 0.0108868 | 0.00377775 | 0.652929 | 0.400828 | 0.126507 |
| full_ffn | 0.01 | 0.000375513 | 0.00890898 | 0.0193124 | 0.534309 | 0.242426 | 0.111317 |
| full_ffn | 0.1 | 0.00160547 | 0.0079014 | 0.0825688 | 0.47388 | 0.125586 | 0.0921733 |
| full_ffn | 1 | 0.00505978 | 0.0105424 | 0.260222 | 0.632269 | 0.0475747 | 0.0819714 |
| full_ffn | 10 | 0.00959802 | 0.0126518 | 0.493621 | 0.758779 | 0.0179321 | 0.0567127 |
| controller | 0.0001 | 1.29667e-05 | 0.0371965 | 0.00066687 | 2.23083 | 3.43415 | 0.23128 |
| controller | 0.001 | 0.000172702 | 0.0155016 | 0.00888195 | 0.929696 | 2.39058 | 0.16047 |
| controller | 0.01 | 0.00111716 | 0.00810196 | 0.0574551 | 0.485909 | 1.28305 | 0.104228 |
| controller | 0.1 | 0.00395625 | 0.00962434 | 0.203468 | 0.577212 | 0.434255 | 0.0847986 |
| controller | 1 | 0.00672428 | 0.0121835 | 0.345827 | 0.730694 | 0.133272 | 0.0810029 |
| controller | 10 | 0.00953511 | 0.0135883 | 0.490386 | 0.814948 | 0.0603437 | 0.0653613 |
| controller_value_rms_equalized | 0.0001 | 2.51993e-06 | 0.0169169 | 0.000129599 | 1.01458 | 9.00535 | 0.166907 |
| controller_value_rms_equalized | 0.001 | 5.19628e-05 | 0.0131313 | 0.00267242 | 0.787542 | 6.97786 | 0.146618 |
| controller_value_rms_equalized | 0.01 | 0.000434543 | 0.00856294 | 0.0223483 | 0.513555 | 4.23924 | 0.110937 |
| controller_value_rms_equalized | 0.1 | 0.00204441 | 0.00837781 | 0.105143 | 0.502452 | 2.08894 | 0.0899547 |
| controller_value_rms_equalized | 1 | 0.00588358 | 0.0114381 | 0.30259 | 0.685989 | 0.710568 | 0.0784223 |
| controller_value_rms_equalized | 10 | 0.00995626 | 0.0136077 | 0.512045 | 0.816112 | 0.266931 | 0.0604005 |
| projection_P | 0.0001 | 8.2083e-06 | 0.0113827 | 0.000422149 | 0.682668 | 0.641564 | 0.127127 |
| projection_P | 0.001 | 7.6314e-05 | 0.0107662 | 0.00392479 | 0.645693 | 0.416829 | 0.123615 |
| projection_P | 0.01 | 0.000375635 | 0.00899403 | 0.0193187 | 0.53941 | 0.253707 | 0.1098 |
| projection_P | 0.1 | 0.00162686 | 0.00796843 | 0.0836684 | 0.477901 | 0.130836 | 0.0915334 |
| projection_P | 1 | 0.00506762 | 0.0104871 | 0.260625 | 0.628956 | 0.0499581 | 0.0811149 |
| projection_P | 10 | 0.00971206 | 0.012657 | 0.499486 | 0.759094 | 0.0186699 | 0.0555356 |

### Ring8 / vector_add / update 5000

Original calibration MSE: 0.0156043612; original evaluation MSE: 0.0161817574.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 5.57458e-06 | 0.00897262 | 0.000357245 | 0.55449 | 0.635464 | 0.126076 |
| full_ffn | 0.001 | 4.40374e-05 | 0.00907701 | 0.00282212 | 0.560941 | 0.482476 | 0.123637 |
| full_ffn | 0.01 | 0.000358353 | 0.00867626 | 0.0229649 | 0.536175 | 0.311326 | 0.114657 |
| full_ffn | 0.1 | 0.00184991 | 0.00785631 | 0.118551 | 0.485504 | 0.143961 | 0.0914854 |
| full_ffn | 1 | 0.00493988 | 0.0098035 | 0.316571 | 0.605836 | 0.052375 | 0.0663728 |
| full_ffn | 10 | 0.00885481 | 0.0122522 | 0.567457 | 0.757163 | 0.0181508 | 0.0398582 |
| controller | 0.0001 | 6.99124e-06 | 0.0230451 | 0.000448031 | 1.42414 | 4.67215 | 0.181711 |
| controller | 0.001 | 0.000118688 | 0.016646 | 0.00760607 | 1.02869 | 3.49969 | 0.157638 |
| controller | 0.01 | 0.000951058 | 0.0105179 | 0.0609482 | 0.649988 | 1.93756 | 0.124283 |
| controller | 0.1 | 0.00373181 | 0.00877835 | 0.239152 | 0.542484 | 0.719485 | 0.0832268 |
| controller | 1 | 0.00693764 | 0.0109658 | 0.444596 | 0.677665 | 0.213738 | 0.0602869 |
| controller | 10 | 0.0101055 | 0.0131444 | 0.647608 | 0.8123 | 0.0754159 | 0.035881 |
| controller_value_rms_equalized | 0.0001 | 1.86842e-06 | 0.0178812 | 0.000119737 | 1.10502 | 10.4259 | 0.165271 |
| controller_value_rms_equalized | 0.001 | 4.32319e-05 | 0.0156708 | 0.0027705 | 0.968421 | 8.49071 | 0.154848 |
| controller_value_rms_equalized | 0.01 | 0.000466972 | 0.0113538 | 0.0299257 | 0.701645 | 5.43638 | 0.129623 |
| controller_value_rms_equalized | 0.1 | 0.00245719 | 0.00842539 | 0.157468 | 0.520672 | 2.33879 | 0.0886965 |
| controller_value_rms_equalized | 1 | 0.0058104 | 0.0104288 | 0.372357 | 0.644479 | 0.79288 | 0.0634901 |
| controller_value_rms_equalized | 10 | 0.00977623 | 0.0128475 | 0.626506 | 0.793952 | 0.258955 | 0.0380021 |
| projection_P | 0.0001 | 6.39789e-06 | 0.00944842 | 0.000410007 | 0.583893 | 0.66852 | 0.126368 |
| projection_P | 0.001 | 4.78078e-05 | 0.00924705 | 0.00306375 | 0.571449 | 0.505842 | 0.12297 |
| projection_P | 0.01 | 0.000382306 | 0.00866802 | 0.0244999 | 0.535666 | 0.319177 | 0.11385 |
| projection_P | 0.1 | 0.00187339 | 0.00789417 | 0.120055 | 0.487844 | 0.146126 | 0.0914621 |
| projection_P | 1 | 0.00490653 | 0.00978203 | 0.314433 | 0.60451 | 0.0535913 | 0.0666003 |
| projection_P | 10 | 0.00882387 | 0.012228 | 0.565475 | 0.755666 | 0.0186328 | 0.039982 |

### Ring8 / swiglu_budget / update 5000

Original calibration MSE: 0.0148953162; original evaluation MSE: 0.0166267761.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 4.72688e-07 | 0.00885781 | 3.1734e-05 | 0.532744 | 0.285625 | 0.131759 |
| full_ffn | 0.001 | 6.93153e-06 | 0.00807143 | 0.00046535 | 0.485448 | 0.249121 | 0.12807 |
| full_ffn | 0.01 | 8.94192e-05 | 0.00755239 | 0.00600317 | 0.45423 | 0.184441 | 0.121653 |
| full_ffn | 0.1 | 0.000639588 | 0.00784759 | 0.0429389 | 0.471985 | 0.117334 | 0.107539 |
| full_ffn | 1 | 0.00328213 | 0.00908987 | 0.220346 | 0.546701 | 0.0564819 | 0.0763385 |
| full_ffn | 10 | 0.00931148 | 0.0125525 | 0.625128 | 0.754958 | 0.0162745 | 0.0342076 |

### Ring8 / swiglu / update 5000

Original calibration MSE: 0.0133216869; original evaluation MSE: 0.014012579.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 7.80337e-07 | 0.00758617 | 5.85764e-05 | 0.541383 | 0.289257 | 0.124032 |
| full_ffn | 0.001 | 8.27282e-06 | 0.00728952 | 0.000621004 | 0.520212 | 0.23522 | 0.119509 |
| full_ffn | 0.01 | 7.70375e-05 | 0.00694383 | 0.00578286 | 0.495543 | 0.16727 | 0.113512 |
| full_ffn | 0.1 | 0.000483657 | 0.00688483 | 0.036306 | 0.491332 | 0.105501 | 0.101593 |
| full_ffn | 1 | 0.00232659 | 0.00769959 | 0.174647 | 0.549477 | 0.053207 | 0.0742908 |
| full_ffn | 10 | 0.00730515 | 0.0107403 | 0.548365 | 0.766472 | 0.017986 | 0.0306358 |

### Product16 / scalar_mul / update 1000

Original calibration MSE: 0.489898076; original evaluation MSE: 0.598689233.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 2.37751e-05 | 1.72178 | 4.85306e-05 | 2.87591 | 22.0835 | 1.06989 |
| full_ffn | 0.001 | 0.00154315 | 1.59461 | 0.00314994 | 2.6635 | 19.5924 | 1.01077 |
| full_ffn | 0.01 | 0.0286522 | 1.2338 | 0.0584861 | 2.06084 | 12.0103 | 0.808265 |
| full_ffn | 0.1 | 0.134153 | 0.887841 | 0.273838 | 1.48297 | 4.81515 | 0.53812 |
| full_ffn | 1 | 0.294879 | 0.669958 | 0.601919 | 1.11904 | 1.43317 | 0.292116 |
| full_ffn | 10 | 0.428204 | 0.604178 | 0.874067 | 1.00917 | 0.292166 | 0.11179 |
| controller | 0.0001 | 0.31715 | 1.51616 | 0.64738 | 2.53247 | 2340.05 | 0.999711 |
| controller | 0.001 | 0.325929 | 0.916647 | 0.6653 | 1.53109 | 1177.64 | 0.610834 |
| controller | 0.01 | 0.34931 | 0.693735 | 0.713026 | 1.15876 | 495.247 | 0.390705 |
| controller | 0.1 | 0.390136 | 0.634004 | 0.796362 | 1.05899 | 165.941 | 0.268491 |
| controller | 1 | 0.433403 | 0.610467 | 0.88468 | 1.01967 | 45.6347 | 0.173742 |
| controller | 10 | 0.465874 | 0.599683 | 0.950961 | 1.00166 | 10.8874 | 0.093594 |
| controller_value_rms_equalized | 0.0001 | 0.321071 | 1.65145 | 0.655383 | 2.75844 | 2300.15 | 1.0696 |
| controller_value_rms_equalized | 0.001 | 0.330101 | 0.920042 | 0.673816 | 1.53676 | 1005.39 | 0.618251 |
| controller_value_rms_equalized | 0.01 | 0.347885 | 0.710853 | 0.710117 | 1.18735 | 444.686 | 0.406321 |
| controller_value_rms_equalized | 0.1 | 0.383467 | 0.637207 | 0.782749 | 1.06434 | 158.091 | 0.281179 |
| controller_value_rms_equalized | 1 | 0.426364 | 0.61094 | 0.870311 | 1.02046 | 45.892 | 0.18762 |
| controller_value_rms_equalized | 10 | 0.461883 | 0.59907 | 0.942815 | 1.00064 | 11.3161 | 0.104541 |
| projection_P | 0.0001 | 2.40795e-05 | 1.73112 | 4.9152e-05 | 2.89151 | 22.1469 | 1.07393 |
| projection_P | 0.001 | 0.00155956 | 1.60267 | 0.00318344 | 2.67696 | 19.6341 | 1.01443 |
| projection_P | 0.01 | 0.0288018 | 1.23727 | 0.0587915 | 2.06664 | 12.0143 | 0.810244 |
| projection_P | 0.1 | 0.134313 | 0.888324 | 0.274166 | 1.48378 | 4.81407 | 0.538513 |
| projection_P | 1 | 0.294933 | 0.670008 | 0.60203 | 1.11912 | 1.43309 | 0.292138 |
| projection_P | 10 | 0.428214 | 0.604184 | 0.874088 | 1.00918 | 0.292169 | 0.111789 |

### Product16 / scalar_add / update 1000

Original calibration MSE: 0.50922183; original evaluation MSE: 0.617257917.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 0.000124942 | 1.79079 | 0.000245358 | 2.90121 | 38.8627 | 1.11636 |
| full_ffn | 0.001 | 0.00428054 | 1.58454 | 0.00840605 | 2.56706 | 30.0366 | 1.01634 |
| full_ffn | 0.01 | 0.0380658 | 1.25858 | 0.074753 | 2.03899 | 16.488 | 0.821861 |
| full_ffn | 0.1 | 0.142886 | 0.927478 | 0.280597 | 1.50258 | 6.61825 | 0.561307 |
| full_ffn | 1 | 0.303971 | 0.694171 | 0.596932 | 1.12461 | 1.97857 | 0.306775 |
| full_ffn | 10 | 0.440553 | 0.622436 | 0.865149 | 1.00839 | 0.420999 | 0.117355 |
| controller | 0.0001 | 0.328374 | 3.18618 | 0.644855 | 5.16183 | 3592.24 | 1.62123 |
| controller | 0.001 | 0.345162 | 1.21463 | 0.677822 | 1.96778 | 1592.41 | 0.782874 |
| controller | 0.01 | 0.377718 | 0.791927 | 0.741756 | 1.28298 | 582.51 | 0.437763 |
| controller | 0.1 | 0.42033 | 0.69555 | 0.825436 | 1.12684 | 184.34 | 0.300527 |
| controller | 1 | 0.462332 | 0.633838 | 0.907918 | 1.02686 | 50.3139 | 0.171392 |
| controller | 10 | 0.491854 | 0.615765 | 0.965892 | 0.997581 | 10.0269 | 0.0835299 |
| controller_value_rms_equalized | 0.0001 | 0.332745 | 3.30154 | 0.653438 | 5.34872 | 4589.73 | 1.66078 |
| controller_value_rms_equalized | 0.001 | 0.344758 | 1.28523 | 0.677028 | 2.08216 | 2077.45 | 0.826987 |
| controller_value_rms_equalized | 0.01 | 0.370166 | 0.82621 | 0.726925 | 1.33852 | 815.856 | 0.47502 |
| controller_value_rms_equalized | 0.1 | 0.408969 | 0.716591 | 0.803126 | 1.16093 | 274.413 | 0.339657 |
| controller_value_rms_equalized | 1 | 0.452026 | 0.644066 | 0.88768 | 1.04343 | 79.6068 | 0.208146 |
| controller_value_rms_equalized | 10 | 0.486718 | 0.61601 | 0.955808 | 0.997979 | 17.0808 | 0.101397 |
| projection_P | 0.0001 | 0.000125662 | 1.79377 | 0.000246772 | 2.90603 | 38.934 | 1.11733 |
| projection_P | 0.001 | 0.0043013 | 1.58595 | 0.0084468 | 2.56935 | 30.0751 | 1.01684 |
| projection_P | 0.01 | 0.0381696 | 1.25845 | 0.0749568 | 2.03877 | 16.4909 | 0.82193 |
| projection_P | 0.1 | 0.142973 | 0.927387 | 0.280768 | 1.50243 | 6.61864 | 0.561361 |
| projection_P | 1 | 0.304023 | 0.694193 | 0.597035 | 1.12464 | 1.97862 | 0.306787 |
| projection_P | 10 | 0.440565 | 0.622453 | 0.865173 | 1.00842 | 0.421027 | 0.117373 |

### Product16 / vector_mul / update 1000

Original calibration MSE: 0.502770539; original evaluation MSE: 0.606523708.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 4.26235e-05 | 1.69763 | 8.47772e-05 | 2.79895 | 28.9558 | 1.06855 |
| full_ffn | 0.001 | 0.00239041 | 1.55177 | 0.00475448 | 2.55846 | 24.7274 | 0.994903 |
| full_ffn | 0.01 | 0.0342666 | 1.21557 | 0.0681555 | 2.00416 | 14.2797 | 0.791886 |
| full_ffn | 0.1 | 0.141863 | 0.891891 | 0.282163 | 1.4705 | 5.71991 | 0.534007 |
| full_ffn | 1 | 0.307723 | 0.676091 | 0.612055 | 1.1147 | 1.68747 | 0.288174 |
| full_ffn | 10 | 0.441259 | 0.612249 | 0.877655 | 1.00944 | 0.339489 | 0.113294 |
| controller | 0.0001 | 0.00016421 | 1.69706 | 0.000326611 | 2.798 | 444.678 | 1.05932 |
| controller | 0.001 | 0.00675533 | 1.4235 | 0.0134362 | 2.34699 | 349.716 | 0.919076 |
| controller | 0.01 | 0.0667494 | 1.05021 | 0.132763 | 1.73152 | 176.97 | 0.672453 |
| controller | 0.1 | 0.212715 | 0.791685 | 0.423086 | 1.30528 | 58.8818 | 0.435096 |
| controller | 1 | 0.364446 | 0.65493 | 0.724875 | 1.07981 | 15.21 | 0.238076 |
| controller | 10 | 0.460008 | 0.612858 | 0.914946 | 1.01044 | 2.92434 | 0.104089 |
| controller_value_rms_equalized | 0.0001 | 0.000814521 | 1.67373 | 0.00162007 | 2.75954 | 796.784 | 1.06826 |
| controller_value_rms_equalized | 0.001 | 0.0104828 | 1.37723 | 0.0208501 | 2.27069 | 501.479 | 0.892883 |
| controller_value_rms_equalized | 0.01 | 0.0519114 | 1.10367 | 0.103251 | 1.81967 | 254.677 | 0.701041 |
| controller_value_rms_equalized | 0.1 | 0.162959 | 0.857602 | 0.324122 | 1.41396 | 99.6887 | 0.4869 |
| controller_value_rms_equalized | 1 | 0.326004 | 0.667013 | 0.648414 | 1.09973 | 28.3024 | 0.263816 |
| controller_value_rms_equalized | 10 | 0.447087 | 0.611734 | 0.889246 | 1.00859 | 5.56475 | 0.114613 |
| projection_P | 0.0001 | 4.46746e-05 | 1.71002 | 8.88569e-05 | 2.81937 | 29.3175 | 1.07415 |
| projection_P | 0.001 | 0.00246998 | 1.56002 | 0.00491273 | 2.57207 | 24.9438 | 0.99857 |
| projection_P | 0.01 | 0.0346259 | 1.21906 | 0.0688701 | 2.00991 | 14.3416 | 0.793451 |
| projection_P | 0.1 | 0.14212 | 0.892729 | 0.282674 | 1.47188 | 5.74011 | 0.534566 |
| projection_P | 1 | 0.307666 | 0.676226 | 0.611942 | 1.11492 | 1.69468 | 0.288501 |
| projection_P | 10 | 0.441165 | 0.612253 | 0.877469 | 1.00945 | 0.341253 | 0.113412 |

### Product16 / vector_add / update 1000

Original calibration MSE: 0.502191178; original evaluation MSE: 0.617234746.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 0.000373279 | 1.86173 | 0.000743301 | 3.01625 | 46.1457 | 1.12819 |
| full_ffn | 0.001 | 0.00722397 | 1.62244 | 0.0143849 | 2.62857 | 31.4716 | 1.02053 |
| full_ffn | 0.01 | 0.043418 | 1.26446 | 0.0864571 | 2.04858 | 16.4086 | 0.819935 |
| full_ffn | 0.1 | 0.145376 | 0.921203 | 0.289483 | 1.49247 | 6.60226 | 0.552723 |
| full_ffn | 1 | 0.304568 | 0.693367 | 0.606477 | 1.12334 | 1.97985 | 0.297131 |
| full_ffn | 10 | 0.438718 | 0.623876 | 0.873608 | 1.01076 | 0.408741 | 0.110918 |
| controller | 0.0001 | 0.00524758 | 2.14398 | 0.0104494 | 3.47352 | 722.643 | 1.27488 |
| controller | 0.001 | 0.0289082 | 1.32006 | 0.0575641 | 2.13867 | 392.852 | 0.884699 |
| controller | 0.01 | 0.104432 | 0.917468 | 0.207952 | 1.48642 | 168.62 | 0.580461 |
| controller | 0.1 | 0.238279 | 0.760247 | 0.474478 | 1.2317 | 54.6705 | 0.394652 |
| controller | 1 | 0.373491 | 0.669458 | 0.743724 | 1.08461 | 14.4483 | 0.240021 |
| controller | 10 | 0.463109 | 0.625877 | 0.922176 | 1.014 | 2.78128 | 0.0998213 |
| controller_value_rms_equalized | 0.0001 | 0.0209976 | 2.76296 | 0.0418119 | 4.47635 | 1491.49 | 1.51165 |
| controller_value_rms_equalized | 0.001 | 0.0454032 | 1.2746 | 0.0904102 | 2.06501 | 614.741 | 0.851686 |
| controller_value_rms_equalized | 0.01 | 0.0891845 | 0.951532 | 0.177591 | 1.5416 | 291.018 | 0.604662 |
| controller_value_rms_equalized | 0.1 | 0.192194 | 0.792203 | 0.382711 | 1.28347 | 110.883 | 0.429901 |
| controller_value_rms_equalized | 1 | 0.335855 | 0.685465 | 0.668779 | 1.11054 | 31.8412 | 0.271512 |
| controller_value_rms_equalized | 10 | 0.447102 | 0.628616 | 0.890302 | 1.01844 | 6.61468 | 0.114071 |
| projection_P | 0.0001 | 0.000404081 | 1.88971 | 0.000804635 | 3.06158 | 46.9477 | 1.14032 |
| projection_P | 0.001 | 0.00744037 | 1.63911 | 0.0148158 | 2.65557 | 31.7512 | 1.02845 |
| projection_P | 0.01 | 0.0437974 | 1.27466 | 0.0872126 | 2.06511 | 16.5171 | 0.825439 |
| projection_P | 0.1 | 0.145771 | 0.924026 | 0.29027 | 1.49704 | 6.63693 | 0.554802 |
| projection_P | 1 | 0.304514 | 0.693733 | 0.606371 | 1.12394 | 1.99137 | 0.297737 |
| projection_P | 10 | 0.438548 | 0.623911 | 0.87327 | 1.01082 | 0.41185 | 0.111138 |

### Product16 / swiglu_budget / update 1000

Original calibration MSE: 0.25554922; original evaluation MSE: 0.30351765.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 7.41202e-06 | 0.620708 | 2.90043e-05 | 2.04505 | 3.59363 | 0.58626 |
| full_ffn | 0.001 | 0.00047199 | 0.580668 | 0.00184696 | 1.91313 | 3.19773 | 0.551527 |
| full_ffn | 0.01 | 0.00842853 | 0.478439 | 0.032982 | 1.57631 | 2.0678 | 0.452051 |
| full_ffn | 0.1 | 0.0445339 | 0.390346 | 0.174268 | 1.28607 | 0.986883 | 0.330704 |
| full_ffn | 1 | 0.128987 | 0.321245 | 0.504744 | 1.05841 | 0.335234 | 0.186002 |
| full_ffn | 10 | 0.216383 | 0.298696 | 0.846736 | 0.984116 | 0.0693135 | 0.0611289 |

### Product16 / swiglu / update 1000

Original calibration MSE: 0.260635276; original evaluation MSE: 0.324899674.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 3.68836e-06 | 0.634453 | 1.41514e-05 | 1.95277 | 3.53144 | 0.569993 |
| full_ffn | 0.001 | 0.000269504 | 0.600328 | 0.00103403 | 1.84773 | 3.25406 | 0.541259 |
| full_ffn | 0.01 | 0.00643616 | 0.491591 | 0.0246941 | 1.51306 | 2.26091 | 0.438174 |
| full_ffn | 0.1 | 0.0411147 | 0.403627 | 0.157748 | 1.24231 | 1.10725 | 0.317389 |
| full_ffn | 1 | 0.125333 | 0.348284 | 0.480873 | 1.07197 | 0.387352 | 0.193445 |
| full_ffn | 10 | 0.219133 | 0.322606 | 0.840765 | 0.99294 | 0.0804841 | 0.0654322 |

### Product16 / scalar_mul / update 5000

Original calibration MSE: 0.243639099; original evaluation MSE: 0.315153876.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 5.70458e-05 | 0.874644 | 0.00023414 | 2.77529 | 6.40523 | 0.742603 |
| full_ffn | 0.001 | 0.00227466 | 0.75785 | 0.0093362 | 2.4047 | 4.95272 | 0.656716 |
| full_ffn | 0.01 | 0.0199808 | 0.593496 | 0.08201 | 1.88319 | 2.61893 | 0.515359 |
| full_ffn | 0.1 | 0.0705549 | 0.453028 | 0.289588 | 1.43748 | 1.03518 | 0.371026 |
| full_ffn | 1 | 0.145567 | 0.353148 | 0.597469 | 1.12056 | 0.310153 | 0.216659 |
| full_ffn | 10 | 0.21023 | 0.316723 | 0.862875 | 1.00498 | 0.0672021 | 0.0776789 |
| controller | 0.0001 | 0.221894 | 0.812286 | 0.910747 | 2.57743 | 46.8084 | 0.720047 |
| controller | 0.001 | 0.224539 | 0.445556 | 0.921603 | 1.41377 | 18.5063 | 0.376412 |
| controller | 0.01 | 0.228966 | 0.34437 | 0.939774 | 1.0927 | 7.28532 | 0.185935 |
| controller | 0.1 | 0.235592 | 0.320185 | 0.966972 | 1.01596 | 1.95762 | 0.087046 |
| controller | 1 | 0.239977 | 0.317274 | 0.984971 | 1.00673 | 0.413389 | 0.0599797 |
| controller | 10 | 0.241873 | 0.315997 | 0.992749 | 1.00268 | 0.0885079 | 0.0468819 |
| controller_value_rms_equalized | 0.0001 | 0.221877 | 0.770239 | 0.91068 | 2.44401 | 114.139 | 0.692065 |
| controller_value_rms_equalized | 0.001 | 0.224398 | 0.447113 | 0.921028 | 1.41871 | 46.4533 | 0.381194 |
| controller_value_rms_equalized | 0.01 | 0.22885 | 0.342858 | 0.9393 | 1.08791 | 17.9229 | 0.180709 |
| controller_value_rms_equalized | 0.1 | 0.235258 | 0.320457 | 0.965599 | 1.01683 | 4.95982 | 0.0870364 |
| controller_value_rms_equalized | 1 | 0.239814 | 0.317152 | 0.984302 | 1.00634 | 1.07842 | 0.0581747 |
| controller_value_rms_equalized | 10 | 0.241863 | 0.315987 | 0.99271 | 1.00264 | 0.218742 | 0.0464782 |
| projection_P | 0.0001 | 5.57969e-05 | 0.873812 | 0.000229015 | 2.77265 | 6.42769 | 0.741694 |
| projection_P | 0.001 | 0.00224065 | 0.756969 | 0.00919659 | 2.4019 | 4.97907 | 0.655627 |
| projection_P | 0.01 | 0.0198136 | 0.592569 | 0.0813234 | 1.88025 | 2.63689 | 0.514117 |
| projection_P | 0.1 | 0.0701681 | 0.453223 | 0.288 | 1.4381 | 1.04388 | 0.37087 |
| projection_P | 1 | 0.145107 | 0.353544 | 0.595581 | 1.12181 | 0.313301 | 0.217127 |
| projection_P | 10 | 0.209965 | 0.316835 | 0.861788 | 1.00533 | 0.0680672 | 0.0776845 |

### Product16 / scalar_add / update 5000

Original calibration MSE: 0.266202931; original evaluation MSE: 0.343837022.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 8.41134e-05 | 1.11907 | 0.000315975 | 3.25466 | 8.05704 | 0.897697 |
| full_ffn | 0.001 | 0.00298714 | 0.933051 | 0.0112213 | 2.71364 | 6.14313 | 0.778387 |
| full_ffn | 0.01 | 0.0252821 | 0.675121 | 0.094973 | 1.96349 | 3.21767 | 0.572165 |
| full_ffn | 0.1 | 0.0873759 | 0.490449 | 0.32823 | 1.4264 | 1.20079 | 0.381311 |
| full_ffn | 1 | 0.168062 | 0.388036 | 0.63133 | 1.12855 | 0.343865 | 0.225369 |
| full_ffn | 10 | 0.231929 | 0.347853 | 0.87125 | 1.01168 | 0.0742194 | 0.089527 |
| controller | 0.0001 | 0.204573 | 1.40086 | 0.768485 | 4.07419 | 107.998 | 1.058 |
| controller | 0.001 | 0.214713 | 0.665618 | 0.806576 | 1.93585 | 41.7174 | 0.58528 |
| controller | 0.01 | 0.229872 | 0.439851 | 0.86352 | 1.27924 | 13.6007 | 0.322137 |
| controller | 0.1 | 0.245424 | 0.363367 | 0.921942 | 1.0568 | 3.41952 | 0.157246 |
| controller | 1 | 0.254497 | 0.350723 | 0.956026 | 1.02003 | 0.796384 | 0.099554 |
| controller | 10 | 0.259801 | 0.346959 | 0.97595 | 1.00908 | 0.201239 | 0.0673324 |
| controller_value_rms_equalized | 0.0001 | 0.20156 | 1.42633 | 0.757166 | 4.14828 | 589.242 | 1.07637 |
| controller_value_rms_equalized | 0.001 | 0.210691 | 0.686855 | 0.791468 | 1.99762 | 231.489 | 0.610469 |
| controller_value_rms_equalized | 0.01 | 0.225101 | 0.456574 | 0.845601 | 1.32788 | 81.9296 | 0.350806 |
| controller_value_rms_equalized | 0.1 | 0.242847 | 0.368224 | 0.912264 | 1.07093 | 21.5414 | 0.171736 |
| controller_value_rms_equalized | 1 | 0.253818 | 0.350857 | 0.953476 | 1.02042 | 4.66954 | 0.103841 |
| controller_value_rms_equalized | 10 | 0.259249 | 0.346446 | 0.973878 | 1.00759 | 1.18314 | 0.0700519 |
| projection_P | 0.0001 | 8.46257e-05 | 1.10336 | 0.000317899 | 3.20897 | 8.12244 | 0.888443 |
| projection_P | 0.001 | 0.00299968 | 0.926689 | 0.0112684 | 2.69514 | 6.18434 | 0.773637 |
| projection_P | 0.01 | 0.025238 | 0.676093 | 0.0948072 | 1.96632 | 3.23599 | 0.572438 |
| projection_P | 0.1 | 0.0870781 | 0.491024 | 0.327112 | 1.42807 | 1.20938 | 0.381946 |
| projection_P | 1 | 0.167701 | 0.388061 | 0.629973 | 1.12862 | 0.346989 | 0.225771 |
| projection_P | 10 | 0.231791 | 0.347711 | 0.87073 | 1.01127 | 0.0750144 | 0.0890576 |

### Product16 / vector_mul / update 5000

Original calibration MSE: 0.220589365; original evaluation MSE: 0.274189988.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 7.18749e-05 | 0.865251 | 0.000325831 | 3.15566 | 6.34857 | 0.742706 |
| full_ffn | 0.001 | 0.00279477 | 0.715494 | 0.0126696 | 2.60948 | 4.79194 | 0.643489 |
| full_ffn | 0.01 | 0.0226334 | 0.527984 | 0.102604 | 1.92561 | 2.38999 | 0.49245 |
| full_ffn | 0.1 | 0.0716404 | 0.39914 | 0.324768 | 1.45571 | 0.896273 | 0.360061 |
| full_ffn | 1 | 0.137549 | 0.310914 | 0.623552 | 1.13394 | 0.268556 | 0.212497 |
| full_ffn | 10 | 0.193708 | 0.275648 | 0.878139 | 1.00532 | 0.0555038 | 0.0809804 |
| controller | 0.0001 | 0.000630679 | 1.55406 | 0.00285906 | 5.66781 | 60.5722 | 1.14743 |
| controller | 0.001 | 0.0104048 | 0.901763 | 0.0471684 | 3.28883 | 35.7332 | 0.807511 |
| controller | 0.01 | 0.0444943 | 0.51591 | 0.201707 | 1.88158 | 15.2212 | 0.519966 |
| controller | 0.1 | 0.105097 | 0.347645 | 0.476436 | 1.2679 | 4.85377 | 0.323033 |
| controller | 1 | 0.164052 | 0.288273 | 0.743699 | 1.05136 | 1.27151 | 0.176084 |
| controller | 10 | 0.202223 | 0.272845 | 0.91674 | 0.995094 | 0.25907 | 0.0703336 |
| controller_value_rms_equalized | 0.0001 | 0.000393877 | 1.46929 | 0.00178557 | 5.35866 | 231.477 | 1.10534 |
| controller_value_rms_equalized | 0.001 | 0.0080314 | 0.948091 | 0.0364088 | 3.45779 | 146.437 | 0.832441 |
| controller_value_rms_equalized | 0.01 | 0.0398205 | 0.546509 | 0.180519 | 1.99318 | 63.694 | 0.537816 |
| controller_value_rms_equalized | 0.1 | 0.0988416 | 0.36915 | 0.448079 | 1.34633 | 21.0323 | 0.341595 |
| controller_value_rms_equalized | 1 | 0.159877 | 0.295582 | 0.724772 | 1.07802 | 5.55326 | 0.187633 |
| controller_value_rms_equalized | 10 | 0.200315 | 0.273415 | 0.90809 | 0.997173 | 1.1503 | 0.0755115 |
| projection_P | 0.0001 | 7.72551e-05 | 0.869563 | 0.000350221 | 3.17139 | 6.53861 | 0.740716 |
| projection_P | 0.001 | 0.00291195 | 0.719157 | 0.0132008 | 2.62284 | 4.89891 | 0.642629 |
| projection_P | 0.01 | 0.0230138 | 0.532276 | 0.104329 | 1.94127 | 2.4283 | 0.494697 |
| projection_P | 0.1 | 0.071949 | 0.400917 | 0.326167 | 1.46219 | 0.907625 | 0.361267 |
| projection_P | 1 | 0.137428 | 0.311628 | 0.623002 | 1.13654 | 0.272739 | 0.213406 |
| projection_P | 10 | 0.193592 | 0.275812 | 0.877611 | 1.00592 | 0.0564749 | 0.0812089 |

### Product16 / vector_add / update 5000

Original calibration MSE: 0.163468193; original evaluation MSE: 0.170235419.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 7.48111e-05 | 0.742811 | 0.000457649 | 4.36343 | 5.36731 | 0.755289 |
| full_ffn | 0.001 | 0.00256351 | 0.568348 | 0.015682 | 3.3386 | 3.97683 | 0.632127 |
| full_ffn | 0.01 | 0.0199467 | 0.348956 | 0.122022 | 2.04984 | 1.89196 | 0.437232 |
| full_ffn | 0.1 | 0.057505 | 0.230304 | 0.351781 | 1.35286 | 0.669659 | 0.28803 |
| full_ffn | 1 | 0.103097 | 0.17919 | 0.630683 | 1.0526 | 0.201585 | 0.162392 |
| full_ffn | 10 | 0.14272 | 0.168636 | 0.873078 | 0.990602 | 0.0434125 | 0.0642714 |
| controller | 0.0001 | 0.000874213 | 1.29153 | 0.00534791 | 7.58675 | 25.6156 | 1.06564 |
| controller | 0.001 | 0.00951741 | 0.709386 | 0.0582218 | 4.16709 | 14.5102 | 0.747262 |
| controller | 0.01 | 0.0376582 | 0.343178 | 0.23037 | 2.01591 | 5.8017 | 0.443413 |
| controller | 0.1 | 0.0804222 | 0.21405 | 0.491974 | 1.25738 | 1.83088 | 0.252329 |
| controller | 1 | 0.122076 | 0.1804 | 0.746788 | 1.05971 | 0.47883 | 0.142615 |
| controller | 10 | 0.14901 | 0.171815 | 0.911556 | 1.00928 | 0.0988707 | 0.0725489 |
| controller_value_rms_equalized | 0.0001 | 0.000683917 | 1.42594 | 0.00418379 | 8.37626 | 207.43 | 1.1328 |
| controller_value_rms_equalized | 0.001 | 0.00979445 | 0.700322 | 0.0599166 | 4.11384 | 118.504 | 0.747958 |
| controller_value_rms_equalized | 0.01 | 0.0395147 | 0.330477 | 0.241727 | 1.94129 | 46.8609 | 0.431323 |
| controller_value_rms_equalized | 0.1 | 0.0837256 | 0.211411 | 0.512183 | 1.24188 | 14.4548 | 0.249633 |
| controller_value_rms_equalized | 1 | 0.124391 | 0.177565 | 0.760947 | 1.04306 | 3.64008 | 0.139817 |
| controller_value_rms_equalized | 10 | 0.148992 | 0.170271 | 0.911441 | 1.00021 | 0.77821 | 0.0693157 |
| projection_P | 0.0001 | 9.55944e-05 | 0.780119 | 0.000584789 | 4.58259 | 6.00492 | 0.786164 |
| projection_P | 0.001 | 0.00294414 | 0.579299 | 0.0180105 | 3.40293 | 4.32608 | 0.64779 |
| projection_P | 0.01 | 0.0209042 | 0.343297 | 0.127879 | 2.0166 | 2.01019 | 0.437462 |
| projection_P | 0.1 | 0.0578684 | 0.22536 | 0.354004 | 1.32382 | 0.709403 | 0.283508 |
| projection_P | 1 | 0.102729 | 0.17749 | 0.628436 | 1.04261 | 0.215829 | 0.158618 |
| projection_P | 10 | 0.142691 | 0.1683 | 0.872895 | 0.988634 | 0.0467626 | 0.0588339 |

### Product16 / swiglu_budget / update 5000

Original calibration MSE: 0.0907378307; original evaluation MSE: 0.0939688764.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 1.20535e-06 | 0.19612 | 1.32839e-05 | 2.08707 | 1.24947 | 0.326991 |
| full_ffn | 0.001 | 8.5728e-05 | 0.185961 | 0.000944788 | 1.97896 | 1.14862 | 0.312533 |
| full_ffn | 0.01 | 0.002087 | 0.151129 | 0.0230003 | 1.60828 | 0.779779 | 0.25701 |
| full_ffn | 0.1 | 0.0121165 | 0.116198 | 0.133533 | 1.23655 | 0.369882 | 0.184597 |
| full_ffn | 1 | 0.0352173 | 0.0960542 | 0.388121 | 1.02219 | 0.14195 | 0.11547 |
| full_ffn | 10 | 0.0679076 | 0.0907404 | 0.748393 | 0.965644 | 0.0359742 | 0.0501115 |

### Product16 / swiglu / update 5000

Original calibration MSE: 0.108778374; original evaluation MSE: 0.100125124.

| Component | λ | Calibration MSE | Evaluation MSE | Calibration share | Evaluation share | Update norm | Evaluation correction RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_ffn | 0.0001 | 1.28674e-06 | 0.20979 | 1.1829e-05 | 2.09527 | 1.60396 | 0.341598 |
| full_ffn | 0.001 | 9.37392e-05 | 0.199127 | 0.000861745 | 1.98878 | 1.49117 | 0.326254 |
| full_ffn | 0.01 | 0.0025158 | 0.160096 | 0.0231278 | 1.59896 | 1.05361 | 0.262144 |
| full_ffn | 0.1 | 0.0165492 | 0.122613 | 0.152137 | 1.2246 | 0.503261 | 0.180044 |
| full_ffn | 1 | 0.0492373 | 0.102983 | 0.452639 | 1.02854 | 0.181132 | 0.105975 |
| full_ffn | 10 | 0.0880024 | 0.0991904 | 0.809006 | 0.990665 | 0.0406217 | 0.042134 |

## Audit coverage

24 cases; 72 kernels; 432 ridge rows; 96 layer rows; 2,160 time rows. Maximum absolute discrepancy when reconstructing stored evaluation MSEs: 0. No case was removed. See `analysis-audit.json` for input hashes and `SHA256SUMS.json` for generated files.
