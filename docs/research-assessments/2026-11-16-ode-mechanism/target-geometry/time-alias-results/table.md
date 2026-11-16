| Task | Model | Original J error | Alias J error | Off-grid change / fit RMS | FP64 endpoint max | FP32 endpoint max |
|---|---|---:|---:|---:|---:|---:|
| Ring8 | Shared P | 0.146 | 0.155 | 0.130 | 8.882e-16 | 7.153e-07 |
| Product16 | Shared P | 0.272 | 0.271 | 0.070 | 7.383e-15 | 1.550e-06 |
| Ring8 | Free ISRU | 0.306 | 0.344 | 0.100 | 8.882e-16 | 7.153e-07 |
| Ring8 | SwiGLU64/Muon | 0.213 | 0.210 | 0.117 | 8.882e-16 | 9.537e-07 |
| Product16 | Free ISRU | 0.280 | 0.278 | 0.071 | 3.997e-15 | 2.384e-06 |
| Product16 | SwiGLU192/AdamW | 0.304 | 0.304 | 0.059 | 1.652e-13 | 3.016e-05 |
| Ring8 | Released stages | 0.160 | 0.168 | 0.154 | 8.882e-16 | 4.768e-07 |
| Product16 | Released stages | 0.265 | 0.266 | 0.063 | 1.554e-15 | 7.153e-07 |
| Ring8 | Palindrome | 0.155 | 0.162 | 0.120 | 8.882e-16 | 9.537e-07 |
| Ring8 | Alternating | 0.168 | 0.177 | 0.140 | 8.882e-16 | 7.153e-07 |
| Product16 | Palindrome | 0.281 | 0.280 | 0.066 | 2.554e-15 | 1.907e-06 |
| Product16 | Alternating | 0.269 | 0.269 | 0.069 | 5.884e-15 | 9.537e-07 |
