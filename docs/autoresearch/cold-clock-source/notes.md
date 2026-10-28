# Why this is the next bounded question

The completed P-coordinate real-state check verifies the exact conditional
optimizer transport at all six preserved boundaries, with 1.55e-15 maximum
post-prediction error. Its ordinary native-rule matrix increments are about
1.67 times larger. This supplies no FID cause and no new ODE computation.

The previous time-only setting was a fixed table calibrated from fitted
native clocks. Its stop rule explicitly does not reject every retrained
time-only model. It stays closed: no table, knots, interpolation or calibration
is reused or refined here. The later cold constant result shows why imposing
a clock after training and training with that clock are different experiments.

New matched Swi86 improves initialization without closing K1's FID lead.
Together these controls justify testing one remaining functional restriction:
whether a clock learned from known objective time can replace feature-clock
feedback. This is not a claim that the JiT body lacks time conditioning.

Actual source data flow was checked in the frozen model_jit.py: forward forms
c=t_embed(t)+y_embed(y), and every block passes that c into the FFN. In this
unconditional protocol y is always 0. No image-dependent hidden state enters
c. Use this existing pathway without a new normalization or raw-time encoder.

No new outcome exists for these two arms. The fixed initial-state pairing,
cost predictions, stronger control set and strict quality rule precede them.
Any positive result still requires novelty/transfer assessment; the experiment
is not a frontier-performance claim.
