# Units and independent statistics

Six models, six separately intervened layers/model,80inputs/model and eight
common banks. Each bank retains ten inputs and six layer interventions. These
are repeated conditional effects, not independent training replicates. The
reference response changes one layer at a time, never all layers simultaneously.

Jointly bootstrap the denominator and all three residual-energy sums across
banks before forming ratios. Exact enumeration uses6435weighted multisets,
equivalent to8^8ordered resamples. The18registered quantities are F_T,F_T-F_N
andF_T-F_rayfor each of six models. Both ordinary95%and nominal18-comparison
Bonferroni intervals are retained in summary.json. No Gaussian assumption or
p-value is used. All zero-response/unresolved-ray cases would remain visible.

| Task/model | Layer | Normal fraction | Tangential fraction | Ray fraction | Native oracle MSE | Full-reference oracle MSE |
|---|---|---:|---:|---:|---:|---:|
| Ring8 / Alternating | blocks.0.ffn.ode1 | 0.008 | 0.994 | -0.014 | 0.007 | 0.007 |
| Ring8 / Alternating | blocks.0.ffn.ode2 | 0.291 | 0.913 | 0.221 | 0.007 | 0.007 |
| Ring8 / Alternating | blocks.1.ffn.ode1 | -0.007 | 0.994 | 0.023 | 0.007 | 0.007 |
| Ring8 / Alternating | blocks.1.ffn.ode2 | -0.015 | 0.885 | -0.055 | 0.007 | 0.007 |
| Ring8 / Alternating | blocks.2.ffn.ode1 | 0.027 | 0.990 | 0.001 | 0.007 | 0.007 |
| Ring8 / Alternating | blocks.2.ffn.ode2 | -0.306 | 0.726 | -0.153 | 0.007 | 0.007 |
| Ring8 / Palindrome | blocks.0.ffn.ode1 | -0.007 | 0.999 | -0.003 | 0.007 | 0.007 |
| Ring8 / Palindrome | blocks.0.ffn.ode2 | 0.258 | 0.936 | 0.178 | 0.007 | 0.008 |
| Ring8 / Palindrome | blocks.1.ffn.ode1 | 0.024 | 0.997 | -0.004 | 0.007 | 0.007 |
| Ring8 / Palindrome | blocks.1.ffn.ode2 | 0.014 | 0.984 | 0.014 | 0.007 | 0.007 |
| Ring8 / Palindrome | blocks.2.ffn.ode1 | 0.058 | 0.994 | 0.034 | 0.007 | 0.007 |
| Ring8 / Palindrome | blocks.2.ffn.ode2 | -0.131 | 0.806 | -0.150 | 0.007 | 0.007 |
| Ring8 / Shared P | blocks.0.ffn.ode1 | 0.010 | 0.996 | -0.009 | 0.007 | 0.007 |
| Ring8 / Shared P | blocks.0.ffn.ode2 | 0.252 | 0.938 | 0.160 | 0.007 | 0.007 |
| Ring8 / Shared P | blocks.1.ffn.ode1 | -0.047 | 0.991 | -0.042 | 0.007 | 0.007 |
| Ring8 / Shared P | blocks.1.ffn.ode2 | 0.160 | 0.897 | 0.115 | 0.007 | 0.007 |
| Ring8 / Shared P | blocks.2.ffn.ode1 | 0.077 | 0.974 | 0.023 | 0.007 | 0.007 |
| Ring8 / Shared P | blocks.2.ffn.ode2 | -0.570 | 0.454 | -0.579 | 0.007 | 0.007 |
| Product16 / Alternating | blocks.0.ffn.ode1 | 0.014 | 0.990 | -0.003 | 0.017 | 0.017 |
| Product16 / Alternating | blocks.0.ffn.ode2 | 0.018 | 0.993 | 0.019 | 0.017 | 0.017 |
| Product16 / Alternating | blocks.1.ffn.ode1 | 0.010 | 0.996 | 0.007 | 0.017 | 0.017 |
| Product16 / Alternating | blocks.1.ffn.ode2 | 0.112 | 0.960 | 0.076 | 0.017 | 0.017 |
| Product16 / Alternating | blocks.2.ffn.ode1 | 0.005 | 0.997 | -0.002 | 0.017 | 0.018 |
| Product16 / Alternating | blocks.2.ffn.ode2 | 0.037 | 0.990 | 0.035 | 0.017 | 0.017 |
| Product16 / Palindrome | blocks.0.ffn.ode1 | -0.035 | 0.952 | -0.010 | 0.019 | 0.019 |
| Product16 / Palindrome | blocks.0.ffn.ode2 | -0.016 | 0.941 | -0.014 | 0.019 | 0.019 |
| Product16 / Palindrome | blocks.1.ffn.ode1 | -0.001 | 0.997 | -0.004 | 0.019 | 0.019 |
| Product16 / Palindrome | blocks.1.ffn.ode2 | 0.073 | 0.983 | 0.034 | 0.019 | 0.021 |
| Product16 / Palindrome | blocks.2.ffn.ode1 | 0.008 | 0.967 | -0.020 | 0.019 | 0.020 |
| Product16 / Palindrome | blocks.2.ffn.ode2 | 0.061 | 0.985 | 0.047 | 0.019 | 0.020 |
| Product16 / Shared P | blocks.0.ffn.ode1 | 0.089 | 0.932 | 0.085 | 0.017 | 0.017 |
| Product16 / Shared P | blocks.0.ffn.ode2 | 0.017 | 0.990 | 0.008 | 0.017 | 0.018 |
| Product16 / Shared P | blocks.1.ffn.ode1 | 0.004 | 0.998 | 0.001 | 0.017 | 0.017 |
| Product16 / Shared P | blocks.1.ffn.ode2 | 0.103 | 0.952 | 0.047 | 0.017 | 0.017 |
| Product16 / Shared P | blocks.2.ffn.ode1 | 0.036 | 0.991 | 0.034 | 0.017 | 0.018 |
| Product16 / Shared P | blocks.2.ffn.ode2 | 0.023 | 0.993 | 0.026 | 0.017 | 0.017 |
