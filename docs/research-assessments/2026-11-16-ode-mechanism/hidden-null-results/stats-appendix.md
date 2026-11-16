# Statistical units and complete comparisons

Six fixed checkpoints, 80 inputs each, six separate layers and all 64 frame
directions. Evaluation RNG 0 prepares 36 data-independent orthonormal frames
before new outputs. Directions are correlated members of a complete frame;
they are not independent statistical samples. The eight original input banks
are the conditional uncertainty units, retaining ten inputs per bank and all
layers/directions. No training-population inference is made.

F uses the same full-reference response denominator. Null residual energies
are averaged over every direction before pooling, without a selected direction.
Delta N/T are actual minus null F. Exact bank bootstrap enumerates 6435 weighted
multisets representing all 8^8 ordered draws. Joint numerator/denominator
resampling preserves pairing. Ordinary 95% and nominal 12-contrast-adjusted
intervals are retained in summary.json. Adjusted two-sided tail probability
is 0.05/24. Zero denominators remain unresolved rather than excluded.

The strong tangential hypothesis requires positive Delta T in every model;
all 6 point contrasts and 6 adjusted
intervals instead have negative sign. Normal specificity is a separate
hypothesis, with 3 resolved positive adjusted intervals.
The frame means and paired F differences are the effect sizes. The finite
response gains divide squared field displacement by squared removed feature
length and do not assume local linearity. Raw precision is retained.
