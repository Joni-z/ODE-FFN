| Task | Trained solver | Inference solver | Field MSE | Pooled SW2 |
|---|---|---|---:|---:|
| Ring8 | midpoint | midpoint | 0.010 | 0.091 |
| Ring8 | midpoint | rk4 | 0.010 | 0.092 |
| Ring8 | rk4 | rk4 | 0.010 | 0.091 |
| Ring8 | rk4 | midpoint | 0.009 | 0.091 |
| Ring8 | free_isru | native | 0.011 | 0.116 |
| Ring8 | swiglu64 | native | 0.011 | 0.120 |
| Product16 | midpoint | midpoint | 0.019 | 0.064 |
| Product16 | midpoint | rk4 | 0.020 | 0.070 |
| Product16 | rk4 | rk4 | 0.019 | 0.065 |
| Product16 | rk4 | midpoint | 0.019 | 0.060 |
| Product16 | free_isru | native | 0.015 | 0.055 |
| Product16 | swiglu192 | native | 0.015 | 0.044 |
