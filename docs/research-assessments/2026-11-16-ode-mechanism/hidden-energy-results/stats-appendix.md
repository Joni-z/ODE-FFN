# Units and numerical uncertainty

Six frozen models,36local layers and80inputs per model, totaling2880local
states. Inputs share banks and times across models and recur across layers;
these are repeated deterministic measurements, not2880independent training runs.
No p-value, performance confidence interval or model winner is inferred.

Reference uncertainty uses per-component tightened-tolerance agreement,
work agreement and exact potential closure under the preregistered gates.
The empirical floor combines reference shift and a floating-point scale;
it is not a rigorous error bound. Both reference endpoints and every input
are retained. The normal share pools squared-error sums across states/layers.
The smooth-potential Taylor remainder is checked for every state and step.

| Task/model | Layer | Reference closure max | Qualified states | Four-step normal share | Four-step feature RMS |
|---|---|---:|---:|---:|---:|
| Ring8 / Alternating | blocks.0.ffn.ode1 | 3.869e-12 | 80/80 | 0.046 | 0.041 |
| Ring8 / Alternating | blocks.0.ffn.ode2 | 3.666e-12 | 80/80 | 0.232 | 0.071 |
| Ring8 / Alternating | blocks.1.ffn.ode1 | 8.958e-12 | 80/80 | 0.044 | 0.023 |
| Ring8 / Alternating | blocks.1.ffn.ode2 | 5.048e-12 | 80/80 | 0.188 | 0.042 |
| Ring8 / Alternating | blocks.2.ffn.ode1 | 8.619e-12 | 80/80 | 0.107 | 0.023 |
| Ring8 / Alternating | blocks.2.ffn.ode2 | 2.821e-12 | 80/80 | 0.150 | 0.057 |
| Ring8 / Palindrome | blocks.0.ffn.ode1 | 5.809e-12 | 80/80 | 0.039 | 0.094 |
| Ring8 / Palindrome | blocks.0.ffn.ode2 | 8.423e-12 | 80/80 | 0.214 | 0.093 |
| Ring8 / Palindrome | blocks.1.ffn.ode1 | 8.811e-12 | 80/80 | 0.048 | 0.070 |
| Ring8 / Palindrome | blocks.1.ffn.ode2 | 9.329e-12 | 80/80 | 0.054 | 0.044 |
| Ring8 / Palindrome | blocks.2.ffn.ode1 | 1.677e-11 | 80/80 | 0.078 | 0.080 |
| Ring8 / Palindrome | blocks.2.ffn.ode2 | 5.265e-12 | 80/80 | 0.182 | 0.063 |
| Ring8 / Shared P | blocks.0.ffn.ode1 | 4.039e-12 | 80/80 | 0.049 | 0.023 |
| Ring8 / Shared P | blocks.0.ffn.ode2 | 3.471e-12 | 80/80 | 0.208 | 0.052 |
| Ring8 / Shared P | blocks.1.ffn.ode1 | 3.601e-12 | 80/80 | 0.069 | 0.011 |
| Ring8 / Shared P | blocks.1.ffn.ode2 | 1.702e-12 | 80/80 | 0.279 | 0.034 |
| Ring8 / Shared P | blocks.2.ffn.ode1 | 3.872e-12 | 80/80 | 0.079 | 0.013 |
| Ring8 / Shared P | blocks.2.ffn.ode2 | 5.205e-12 | 80/80 | 0.338 | 0.059 |
| Product16 / Alternating | blocks.0.ffn.ode1 | 4.487e-12 | 80/80 | 0.045 | 0.028 |
| Product16 / Alternating | blocks.0.ffn.ode2 | 5.555e-12 | 80/80 | 0.027 | 0.067 |
| Product16 / Alternating | blocks.1.ffn.ode1 | 5.535e-12 | 80/80 | 0.024 | 0.055 |
| Product16 / Alternating | blocks.1.ffn.ode2 | 4.158e-12 | 80/80 | 0.051 | 0.099 |
| Product16 / Alternating | blocks.2.ffn.ode1 | 9.660e-12 | 80/80 | 0.022 | 0.105 |
| Product16 / Alternating | blocks.2.ffn.ode2 | 6.249e-12 | 80/80 | 0.034 | 0.155 |
| Product16 / Palindrome | blocks.0.ffn.ode1 | 7.045e-12 | 80/80 | 0.118 | 0.086 |
| Product16 / Palindrome | blocks.0.ffn.ode2 | 6.079e-12 | 80/80 | 0.099 | 0.090 |
| Product16 / Palindrome | blocks.1.ffn.ode1 | 1.459e-11 | 80/80 | 0.021 | 0.107 |
| Product16 / Palindrome | blocks.1.ffn.ode2 | 2.837e-12 | 80/80 | 0.033 | 0.173 |
| Product16 / Palindrome | blocks.2.ffn.ode1 | 9.390e-12 | 80/80 | 0.139 | 0.183 |
| Product16 / Palindrome | blocks.2.ffn.ode2 | 2.716e-11 | 80/80 | 0.025 | 0.222 |
| Product16 / Shared P | blocks.0.ffn.ode1 | 2.389e-12 | 80/80 | 0.113 | 0.011 |
| Product16 / Shared P | blocks.0.ffn.ode2 | 9.628e-13 | 80/80 | 0.070 | 0.039 |
| Product16 / Shared P | blocks.1.ffn.ode1 | 2.444e-12 | 80/80 | 0.017 | 0.014 |
| Product16 / Shared P | blocks.1.ffn.ode2 | 1.537e-12 | 80/80 | 0.054 | 0.043 |
| Product16 / Shared P | blocks.2.ffn.ode1 | 4.587e-12 | 80/80 | 0.050 | 0.031 |
| Product16 / Shared P | blocks.2.ffn.ode2 | 3.991e-12 | 80/80 | 0.013 | 0.039 |
