# Statistics and comparison units

The unit for paired evaluation is one common 4,096-sample bank at fixed
trained weights and fixed projection directions. There are eight banks per
model. These intervals describe evaluation variation, not variation across
training initializations. Training seed is 0 throughout.

Primary differences below are **released minus shared**, so positive is
better for sharing. Means, SDs and approximate paired-t intervals are
multiplied by 1,000 for readability; pooled differences remain separate.

| Task | Mean bank difference x1,000 | SD x1,000 | Approx. 95% interval x1,000 | Shared wins | Pooled difference |
|---|---:|---:|---|---:|---:|
| Ring8 | 12.939 | 9.793 | [4.752, 21.126] | 8/8 | 0.020 |
| Product16 | 6.782 | 1.249 | [5.738, 7.826] | 8/8 | 0.009 |

Exact values are in [comparison.json](comparison.json). The paired-t
interval assumes a sufficiently regular mean of the bank differences;
Shapiro-Wilk p-values are 0.761 and 0.849, respectively. With eight values
this check has little power, so it is not used to certify normality.

The two-sided exact sign test evaluates the median paired bank difference.
There are no ties. Its raw p-value is 0.008 on each task; Holm correction
over the two primary task comparisons gives 0.016 for each. This conditional
test does not test the pooled statistic or an architecture's behavior over
new training initializations. Four ordinary-reference contrasts are
descriptive; their unadjusted exploratory sign-test values are not used as
confirmatory evidence.

The new comparison uses streams 2026112000/2026112100. Earlier coefficient,
stage-coefficient and C8 results use other bound arrays. Do not combine their
absolute SW2 values into one ranking. C8 scores also use correlated group
orbits rather than iid pooled banks.

Matrix work is counted from actual dense operations at batch512. CPU timings
use nine forward trials and three complete Heun128 sampling trials, in
interleaved arm order. Process CPU time and wall time are both retained.
The plots use process CPU time; they do not estimate GPU or total training
cost. All eight trained states are unchanged by the benchmark.

The training-fit comparison preserves the native pair-loss definition.
The gradient probes evaluate eight fixed field-pair batches and implied
clip coefficients. They do not reconstruct training clip frequency.
