# Statistical units and checks

One fixed20kseed0checkpoint per model. The comparison unit for uncertainty is
an input bank, with eight banks containing32sources each. Report every model
and every bank; the same sources are reused across multipliers and resolutions.

For each bank, save residual squared-error sum and displacement squared-error
sum. Jointly resample these sums, then form1-ratio; do not average bank ratios.
The exact bootstrap enumerates6435multisets weighted by their multinomial
counts, totaling8^8ordered resamples. Report conditional95%and nominal
twelve-contrast Bonferroni intervals. No normality assumption or t-test is
used. These are input-conditional intervals, not training-variability estimates.

The scientific null is no improvement over predicting zero displacement,
F<=0. Numerical qualification is a separate deterministic all-component gate.
Positive point estimates alone do not establish a positive adjusted interval.
No sampling-rank claim, p-value or architecture winner is inferred here.

| Task/model | Bank | Full-scale fraction | Residual energy | Displacement energy |
|---|---:|---:|---:|---:|
| Ring8 / Alternating | 0 | 0.986 | 5.360e-04 | 3.888e-02 |
| Ring8 / Alternating | 1 | 0.517 | 5.391e-02 | 1.116e-01 |
| Ring8 / Alternating | 2 | 0.862 | 1.570e-01 | 1.134e+00 |
| Ring8 / Alternating | 3 | 0.219 | 1.291e+00 | 1.654e+00 |
| Ring8 / Alternating | 4 | -2.370 | 3.259e-01 | 9.669e-02 |
| Ring8 / Alternating | 5 | 0.241 | 3.571e+00 | 4.704e+00 |
| Ring8 / Alternating | 6 | -2.199 | 3.429e-01 | 1.072e-01 |
| Ring8 / Alternating | 7 | 0.993 | 3.231e-04 | 4.888e-02 |
| Ring8 / Free ISRU | 0 | 0.988 | 5.898e-04 | 4.755e-02 |
| Ring8 / Free ISRU | 1 | 0.564 | 5.819e-02 | 1.334e-01 |
| Ring8 / Free ISRU | 2 | -6.680 | 6.582e-01 | 8.570e-02 |
| Ring8 / Free ISRU | 3 | 0.375 | 1.215e+00 | 1.943e+00 |
| Ring8 / Free ISRU | 4 | 0.642 | 2.681e-02 | 7.483e-02 |
| Ring8 / Free ISRU | 5 | 0.261 | 1.945e+00 | 2.633e+00 |
| Ring8 / Free ISRU | 6 | -2.730 | 5.053e-01 | 1.355e-01 |
| Ring8 / Free ISRU | 7 | 0.994 | 3.550e-04 | 5.770e-02 |
| Ring8 / Palindrome | 0 | 0.986 | 6.128e-04 | 4.397e-02 |
| Ring8 / Palindrome | 1 | 0.818 | 2.013e-02 | 1.104e-01 |
| Ring8 / Palindrome | 2 | 0.193 | 6.028e-01 | 7.472e-01 |
| Ring8 / Palindrome | 3 | 0.904 | 7.146e-03 | 7.410e-02 |
| Ring8 / Palindrome | 4 | 0.219 | 6.903e-02 | 8.837e-02 |
| Ring8 / Palindrome | 5 | 0.241 | 1.810e+00 | 2.384e+00 |
| Ring8 / Palindrome | 6 | -2.200 | 3.529e-01 | 1.103e-01 |
| Ring8 / Palindrome | 7 | 0.972 | 1.444e-03 | 5.211e-02 |
| Ring8 / Released stages | 0 | 0.898 | 6.366e-03 | 6.215e-02 |
| Ring8 / Released stages | 1 | 0.029 | 1.239e-01 | 1.277e-01 |
| Ring8 / Released stages | 2 | 0.432 | 7.729e-01 | 1.361e+00 |
| Ring8 / Released stages | 3 | 0.409 | 1.007e+00 | 1.705e+00 |
| Ring8 / Released stages | 4 | 0.433 | 8.824e-01 | 1.556e+00 |
| Ring8 / Released stages | 5 | 0.294 | 1.985e+00 | 2.812e+00 |
| Ring8 / Released stages | 6 | 0.096 | 9.968e-01 | 1.103e+00 |
| Ring8 / Released stages | 7 | 0.950 | 3.277e-03 | 6.528e-02 |
| Ring8 / Shared P | 0 | 0.972 | 1.264e-03 | 4.569e-02 |
| Ring8 / Shared P | 1 | 0.760 | 2.229e-02 | 9.277e-02 |
| Ring8 / Shared P | 2 | 0.945 | 7.328e-02 | 1.327e+00 |
| Ring8 / Shared P | 3 | 0.214 | 8.809e-01 | 1.121e+00 |
| Ring8 / Shared P | 4 | -0.135 | 9.205e-02 | 8.110e-02 |
| Ring8 / Shared P | 5 | 0.261 | 1.861e+00 | 2.517e+00 |
| Ring8 / Shared P | 6 | -2.445 | 4.399e-01 | 1.277e-01 |
| Ring8 / Shared P | 7 | 0.968 | 1.855e-03 | 5.831e-02 |
| Ring8 / SwiGLU64/Muon | 0 | 0.811 | 2.462e-02 | 1.300e-01 |
| Ring8 / SwiGLU64/Muon | 1 | 0.130 | 1.425e+00 | 1.638e+00 |
| Ring8 / SwiGLU64/Muon | 2 | 0.951 | 6.790e-02 | 1.380e+00 |
| Ring8 / SwiGLU64/Muon | 3 | 0.993 | 5.530e-04 | 8.395e-02 |
| Ring8 / SwiGLU64/Muon | 4 | 0.986 | 1.180e-03 | 8.202e-02 |
| Ring8 / SwiGLU64/Muon | 5 | 0.264 | 1.858e+00 | 2.523e+00 |
| Ring8 / SwiGLU64/Muon | 6 | -1.220 | 3.213e+00 | 1.448e+00 |
| Ring8 / SwiGLU64/Muon | 7 | 0.126 | 1.527e+00 | 1.747e+00 |
| Product16 / Alternating | 0 | 0.305 | 4.936e+01 | 7.102e+01 |
| Product16 / Alternating | 1 | 0.147 | 5.735e+01 | 6.721e+01 |
| Product16 / Alternating | 2 | -10.498 | 1.062e+03 | 9.234e+01 |
| Product16 / Alternating | 3 | 0.265 | 2.658e+01 | 3.614e+01 |
| Product16 / Alternating | 4 | 0.153 | 4.145e+01 | 4.893e+01 |
| Product16 / Alternating | 5 | 0.328 | 3.456e+01 | 5.146e+01 |
| Product16 / Alternating | 6 | 0.507 | 8.124e+00 | 1.647e+01 |
| Product16 / Alternating | 7 | 0.192 | 5.721e+01 | 7.083e+01 |
| Product16 / Free ISRU | 0 | 0.196 | 7.713e+01 | 9.595e+01 |
| Product16 / Free ISRU | 1 | 0.132 | 6.429e+01 | 7.410e+01 |
| Product16 / Free ISRU | 2 | -16.062 | 1.585e+03 | 9.291e+01 |
| Product16 / Free ISRU | 3 | 0.239 | 3.966e+01 | 5.209e+01 |
| Product16 / Free ISRU | 4 | 0.154 | 5.051e+01 | 5.968e+01 |
| Product16 / Free ISRU | 5 | 0.282 | 3.299e+01 | 4.593e+01 |
| Product16 / Free ISRU | 6 | 0.297 | 3.677e+01 | 5.232e+01 |
| Product16 / Free ISRU | 7 | 0.278 | 4.593e+01 | 6.359e+01 |
| Product16 / Palindrome | 0 | 0.222 | 8.192e+01 | 1.053e+02 |
| Product16 / Palindrome | 1 | 0.175 | 4.907e+01 | 5.951e+01 |
| Product16 / Palindrome | 2 | -45.578 | 4.233e+03 | 9.089e+01 |
| Product16 / Palindrome | 3 | 0.183 | 4.364e+01 | 5.340e+01 |
| Product16 / Palindrome | 4 | 0.179 | 4.966e+01 | 6.050e+01 |
| Product16 / Palindrome | 5 | 0.439 | 2.495e+01 | 4.448e+01 |
| Product16 / Palindrome | 6 | 0.405 | 1.493e+01 | 2.510e+01 |
| Product16 / Palindrome | 7 | 0.208 | 4.792e+01 | 6.054e+01 |
| Product16 / Released stages | 0 | 0.196 | 7.711e+01 | 9.591e+01 |
| Product16 / Released stages | 1 | 0.206 | 3.334e+01 | 4.197e+01 |
| Product16 / Released stages | 2 | -93.202 | 5.313e+03 | 5.640e+01 |
| Product16 / Released stages | 3 | 0.183 | 4.418e+01 | 5.406e+01 |
| Product16 / Released stages | 4 | 0.194 | 4.676e+01 | 5.799e+01 |
| Product16 / Released stages | 5 | 0.357 | 3.140e+01 | 4.886e+01 |
| Product16 / Released stages | 6 | 0.478 | 1.280e+01 | 2.451e+01 |
| Product16 / Released stages | 7 | 0.204 | 5.697e+01 | 7.159e+01 |
| Product16 / Shared P | 0 | 0.213 | 7.998e+01 | 1.016e+02 |
| Product16 / Shared P | 1 | 0.164 | 4.994e+01 | 5.976e+01 |
| Product16 / Shared P | 2 | -21.666 | 1.521e+03 | 6.711e+01 |
| Product16 / Shared P | 3 | 0.277 | 2.488e+01 | 3.442e+01 |
| Product16 / Shared P | 4 | 0.142 | 4.655e+01 | 5.425e+01 |
| Product16 / Shared P | 5 | 0.458 | 2.159e+01 | 3.981e+01 |
| Product16 / Shared P | 6 | 0.467 | 1.120e+01 | 2.101e+01 |
| Product16 / Shared P | 7 | 0.247 | 4.353e+01 | 5.778e+01 |
| Product16 / SwiGLU192/AdamW | 0 | 0.197 | 6.590e+01 | 8.203e+01 |
| Product16 / SwiGLU192/AdamW | 1 | 0.160 | 6.103e+01 | 7.262e+01 |
| Product16 / SwiGLU192/AdamW | 2 | -9.278 | 9.759e+02 | 9.494e+01 |
| Product16 / SwiGLU192/AdamW | 3 | 0.199 | 2.345e+01 | 2.928e+01 |
| Product16 / SwiGLU192/AdamW | 4 | 0.205 | 2.534e+01 | 3.189e+01 |
| Product16 / SwiGLU192/AdamW | 5 | 0.362 | 2.514e+01 | 3.940e+01 |
| Product16 / SwiGLU192/AdamW | 6 | 0.515 | 7.885e+00 | 1.624e+01 |
| Product16 / SwiGLU192/AdamW | 7 | 0.220 | 5.315e+01 | 6.811e+01 |
