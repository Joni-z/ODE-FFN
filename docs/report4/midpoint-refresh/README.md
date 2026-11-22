# Does the response have to follow the state?

On Ring8 the shared four-step midpoint beats matched-compute SwiGLU (pooled SW2 0.071 vs 0.085). On Product16 it trails SwiGLU (0.052 vs 0.047), including the stronger earlier checkpoint (0.034). Freezing the diagonal throughout training worsened both 20k endpoints. That intervention changed the learned weights as well as the step rule. Eight tied Euler updates also trailed midpoint with the same matrix-call count. These controls motivate locating the contribution of the midpoint response on fixed weights.

1. Restore the original four-step checkpoint. At each substep keep the midpoint state and P unchanged, then evaluate its diagonal either at the substep entrance or the midpoint. Compare the predicted field and endpoints on identical selection inputs. A third intervention freezes the diagonal for the entire solve. The difference between the latter two locates refresh across successive substeps.
2. Train a model in which the gate can observe other coordinates: evaluate the diagonal at B y, initializing B to identity so the starting field is identical to midpoint. Use the same four substeps on both tasks. Charge the extra matrix operations against the existing training budget. If the same trained architecture fails either control, continue design selection using the selection stream.
3. Keep seed 0, the training data, sample banks, optimizer, state width and outer sampler fixed. Select on both tasks. Compare pooled Heun128 SW2 on confirmation against the native SwiGLU and Free ISRU endpoints, including their stronger earlier checkpoints. Reserve a fresh paired bank stream for the final candidate and ordinary controls after adaptive search on previously seen results. Record sampler cost separately.

[Existing compute controls](../equal-compute/README.md) · [Frozen-response controls](../reduced-state/README.md)

The frozen checkpoint replay used 512 selection field queries and 512 shared trajectories. The oracle MSE on Ring8 was 0.009 with both stages refreshed, 0.019 with the substep entrance response used at its midpoint, and 0.132 with the solve entrance response held fixed. Product16 gave 0.019, 0.034 and 0.458. The native replay reproduced the original field exactly. This is evidence that these weights use state-dependent refresh on both tasks. Distributional comparisons use all eight selection banks separately.

ARIS ablation review also identified an equal-cost placement control. On the original weights, compare four substeps before SiLU and two after it with the reverse allocation. The two mixed settings use the same 12 generator applications. Use the same ordering for both tasks only if both selection scores favor it. A separate Mac M1 replay from steps 40,000 to 43,367 reproduced the archived Ring8 Adams–Bashforth endpoint weights and training RNG exactly.

The final paired source/target stream is reserved before inspecting candidate results: 2026112300 through 2026112307, with the existing 256 directions. The endpoint comparison includes original 20k ordinary controls and their matched-compute continuations. Previously viewed selection and confirmation scores guide development.

Both laws are invariant under global sign reversal. As a diagnostic, split the fixed velocity into its odd and even components. The population velocity is odd, so the even-component energy measures a removable field error. Compare this for ordinary FFNs and midpoint before using symmetry to guide a new design.

On the same 20k checkpoints, the part of the learned midpoint field that assigns the same velocity to x and -x accounts for 31.0% of paired oracle MSE on Ring8 and 40.3% on Product16. SwiGLU also has this error (31.2% and 37.3%). To remove it in one forward pass, learn a signed carrier u = W x. Supply squared carrier features and time to the existing network, multiply the resulting coefficients by u, then apply a bias-free output projection. The coefficients are identical at x and -x, so the velocity changes sign exactly. No mode centers, rotation axes or oracle labels enter training. The first trainable version retains four-step midpoint. A SwiGLU with the same construction separates the effect of the symmetry from the choice of FFN.

## Selection results

| Change to the four-step model | Ring8 pooled SW2 | Product16 pooled SW2 |
|---|---:|---:|
| Original checkpoint | 0.061 | 0.054 |
| Read the entrance gate at the midpoint of each substep | 0.094 | 0.101 |
| Freeze the gate across all four substeps | 0.367 | 0.344 |
| Four substeps before SiLU, two after | 0.069 | 0.046 |
| Two before SiLU, four after | 0.062 | 0.057 |
| Two before and after | 0.067 | 0.046 |
| Train a dense gate map at matched compute | 0.085 | 0.071 |

The first six rows use the same fitted weights. Moving two of the substeps from one side of SiLU to the other improves Ring8 in one direction and Product16 in the other. We found no common placement rule. The seventh row fits new weights. Its extra matrix multiplication allows about 11k training updates at the cost of 20k native updates. It worsens both endpoints.

[Fixed-weight results](frozen-selection.json) · [Dense gate training endpoints](cross-selection.json) · [Reflection error on six checkpoints](parity.json)

The 10k Product16 reflection model exposes a tail problem on the training monitor's Heun32 validation bank. The generated median radius was 6.234 (target 6.282) and the 99th percentile 6.811 (target 6.750), yet the maximum reached 261.673 (target 6.942). Validation SW2 was 1.515. The native midpoint's maximum on the same bank was 7.667. The readout multiplies a normalized feature by an unbounded signed carrier. The next design changes only that multiplier to u / sqrt(1 + u²), leaving the coordinates used to compute the gate untouched. For fixed parameters this bounds the entire output velocity. A nonzero representational error can remain within the bulk of the distribution. [Exact replay of all four 10k checkpoints](endpoint-tail.json).

The selection stream's stronger ordinary scores are 0.084 on Ring8 and 0.034 on Product16, taking the better of original 20k and full-budget SwiGLU and Free ISRU. [Endpoint scores and weight hashes](ordinary-selection.json).
