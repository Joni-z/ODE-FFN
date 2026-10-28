# Two reference checks that change the next design

## A mathematically exact empirical answer can retrieve the wrong held-out image

At t=115/128, the finite-bank conditional mean has held-out clean-image MSE
0.115404, compared with native K1 0.0101893 and Swi86 0.00983193. Its error is
11.33times K1 and11.74times Swi86. The empirical reference exceeds both neural
models by at least10%at four of the five prespecified times. On paired training
images its MSE is exactly0at the last three times. This is a property of this
reference under the fixed probes, not evidence of memorization by a neural model.

On the held-out pairs, median effective support falls from187.83at13/128 to
1.00078at64/128 and1at90/128. The fixed first held-out image illustrates the
consequence: the bank mean settles on a training image while both neural
predictions retain the held-out image's layout. All times are displayed in
figure03; the exact image0weights and errors are in fixed-image0.csv.

## The same-state model comparison does not show the registered reversal

K1/Swi86 equal-time clean error to the bank mean is0.98876on validation pairs,
0.99689on K1-generated states and0.97702on Swi-generated states. There is no
10%generated-state advantage or paired-validation reversal. The original
weighted1024-pair validation metric and these equal-time64-image panels have
different weights and probes; their rankings are not interchangeable.

Cold fixed P, matched-init SiLU64 and SiLU131 remain in every comparison.
The unknown population conditional mean remains unknown. No distance to this
empirical reference is promoted to a population-error or FID explanation.

## Design consequence

Do not use the exact finite-prior posterior ODE as a neural teacher or justify
a hidden P constraint by proximity to this reference. This is not a failure
of its Gaussian-conditioning algebra. The next useful test needs a necessary
property of the population predictor that does not require replacing the
unknown image law by a discrete training bank. No neural architecture or new
training is selected in this diagnostic.
