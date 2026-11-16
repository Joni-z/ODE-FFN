# Next intervention: which hidden difference reaches the output?

Registered after the complete six-model hidden-energy observation. All2880
local states qualify. Native4step normal feature-error shares are0.034--0.063
on Product16and0.109--0.258on Ring8. Most local error is tangential, but the
remaining network may amplify a small normal component. Test that directly.

## Fixed data and scope

Keep the same six20kshared/PAL/ALTcheckpoints and all80registered interpolation
inputs per checkpoint. Reuse the hash-bound native entrances, native4outputs
and qualified tight continuous references from hidden-energy/formal. No new
states, data selection, response, weights, time conditioning or training.

Change exactly one local layer output per forward call. Its upstream entrance
must be exactly the stored entrance. All downstream computation remains native.
Do not replace all layers at once, which would change the stored entrances.

For h4and reference h*, define delta=h4-h*, q*=ISRU(h*), normal delta_N as
the projection onto q*, and delta_T=delta-delta_N. Retain zero-gradient states.

| Mode | Replaced local output | Question |
|---|---|---|
| Identity | h4 | Does the hook preserve the exact native function? |
| Full reference | h* | Full output effect of refining this one layer |
| Normal correction | h4-delta_N | Effect of removing the potential-normal error |
| Tangential correction | h4-delta_T | Effect of removing tangential error |
| Exact-potential ray | a*h4, a>=0, E(a*h4)=E(h*) | Does matching potential while preserving feature direction reproduce the effect? |

The ray control prevents equating a first-order normal correction with exact
energy matching. Along a nonzero ray, E(a*h4)increases from0to infinity, so
the nonnegative scale is unique. Use64bisection steps after a doubling bracket;
require potential agreement within1e-11absolute/relative. Preserve a zero-output
state with nonzero target energy as unresolved; never remove it.

## Predictions and measurements

Let v4be the native full output and v*layer be the full output after replacing
this one layer by its continuous reference. For mode cdefine

```text
F_c = 1 - sum||v_c-v*layer||² / sum||v4-v*layer||².
```

Pool energies over all six separate single-layer interventions and all80inputs;
also retain each layer/bank/time. This measures reproduction of a particular
refinement response, not generation quality or the oracle velocity itself.
Fvalues need not add because downstream computation is nonlinear.

Hypothesis: tangential correction primarily carries the output refinement
response. Require F_T>0,F_T>F_NandF_T>F_rayfor each fixed checkpoint to support
the six-case claim. Any reversal rejects that general claim; mixed outcomes
stay model specific. No after-the-fact percentage margin is introduced.

Report the exact oracle velocity MSE on these same interpolation inputs for
every mode as a separate diagnostic. A lower80-input MSEdoes not imply lower
SW2orFIDand is not a new training comparison.

Eight input banks are the conditional uncertainty units; each retains its
ten inputs and six separate layer interventions. Jointly resample the four
residual/displacement sums before forming ratios. Enumerate all8^8ordered
resamples using6435weighted multisets. Report nominal18-comparison-adjusted
intervals for the six models' F_T,F_T-F_NandF_T-F_ray, plus ordinary95%intervals.
These describe fixed-weight input variation, not independent training runs.

## Execution gates and decision

Bind native FP32anchors and runtime buffers, replay original FP64prediction
exactly, and check every identity-hook output is bit-identical to the baseline.
Every intervention must see the exact stored upstream entrance and native layer
output. Remove hooks after every call; native output, parameters and buffers
must restore exactly. Check source/asset/reference hashes and reference
qualification before execution. Independently reconstruct local candidates,
ray potential, full-output metrics and bank intervals from retained arrays.

If directional corrections reproduce the output response while energy matching
does not, an energy-only output explanation is rejected on this fixed cohort.
If exact energy matching works better, the small normal feature share did not
identify the output's sensitivity. Neither outcome establishes a sampling
mechanism or an architecture gain. A generation intervention would need its
own affordable native implementation and qualified full-bank comparison.

Warm MacCPUis sufficient; source freeze and a fresh exact documented witness
are required before the full intervention. No145card is used while occupied.
