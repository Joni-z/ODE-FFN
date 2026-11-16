# Analysis units and exact conditional intervals

Training seed 0, 20k updates, unchanged recipe. Quality: eight common 4096-source
banks per split, Heun 128, 256 projection directions; pooled score uses 32,768 samples.
All16 model/split score rows are independently reconstructed in NumPy. Fidelity:
four cases, six local solves and 80 fixed inputs per case; all 1,920 points retained
and qualified by two DOP853 tolerances, energy closure and independent per-input RK45.
Pooled rho is total squared hidden error divided by total continuous displacement.

Resampling unit is the original bank, retaining layers/times/pairs together.
Enumerate 6,435 multisets with exact multinomial weights for all 8^8 ordered bank
resamples. Quality intervals concern mean paired-bank SW2 differences, not pooled
SW2. Fidelity intervals recompute the difference of pooled energy ratios.
Report marginal 95%, registered nominal four-comparison intervals and a conservative
six-comparison sensitivity including both quality splits. These are conditional
evaluation-bank intervals for the retained weights. Do not interpret banks as
independent training runs. No arbitrary performance margin or checkpoint selection.

Cost uses the same process, alternating method order, 10 warmups and 100 repeats,
batch 512 and one CPU thread. Dense MACs and parameters match; weighted additions
differ. Native launch durations include polling granularity and final evaluation,
so they are recorded rather than used as matched throughput estimates.
