# Operator-sharing comparison

Fresh sharing-cohort confirmation, lower is better.

| Task | Model | MSE | Pooled SW2 | Parameters | Dense MACs/example | CPU seconds/512 samples |
|---|---|---:|---:|---:|---:|---:|
| Ring8 | Shared P | 0.010 | 0.091 | 56,706 | 227,648 | 1.512 |
| Ring8 | Stage P | 0.010 | 0.111 | 130,434 | 227,648 | 1.559 |
| Ring8 | Free ISRU | 0.011 | 0.116 | 56,706 | 55,616 | 0.333 |
| Ring8 | SwiGLU | 0.011 | 0.120 | 56,766 | 55,232 | 0.393 |
| Product16 | Shared P | 0.019 | 0.064 | 58,512 | 229,440 | 1.527 |
| Product16 | Stage P | 0.018 | 0.073 | 132,240 | 229,440 | 1.534 |
| Product16 | Free ISRU | 0.015 | 0.055 | 58,512 | 57,408 | 0.318 |
| Product16 | SwiGLU | 0.015 | 0.044 | 108,624 | 106,560 | 0.545 |

CPU times are process CPU time on Mac mini, not GPU throughput. Dense MACs exclude pointwise work. The two focal models have the same matrix work and different parameter counts.

SwiGLU retains Ring8 width64/Muon0.02 and Product16 width192/AdamW0.0003. All focal models use AdamW0.0003.
