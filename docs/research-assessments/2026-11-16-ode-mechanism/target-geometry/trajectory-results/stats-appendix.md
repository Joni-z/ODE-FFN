# Units and uncertainty

Twelve fixed checkpoints, eight common input banks per task,32sources per bank.
Jointly resample each bank's numerator and denominator sums before taking the
explained-fraction ratio. Do not average bank ratios or treat times/multipliers
as independent samples. Exact bootstrap enumerates6435multisets weighted to
8^8ordered resamples. Report95% and nominal99.583% intervals for the twelve-model
family. These concern fixed-weight input variation, not independent trainings.
No Gaussian assumption, across-model p-value or training significance is claimed.
Numerically unqualified rows remain visible and are not interpreted as model failures.
