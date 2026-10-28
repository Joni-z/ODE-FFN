# Fixed analysis layout before new FID extraction

This layout does not change training, the sampler, metric code or the quality
rule. It is frozen while the existing paired trajectories run, before any
new FID is computed. Incomplete evidence produces no result figures or
finished conclusions; the scripts run only after the actual controller exit
and every required audit pass.

The complete comparison has 41 endpoint rows: 35 preserved rows and six new
5k/10k/20k rows. The primary table has 17 endpoints, of which 16 are controls
for the time-clock candidate. Its gate remains exactly the registered 5%
FID improvement and both fitting guards against the best control. The paired
feature-clock result can tighten that reference. All 32 continuous contrasts
for the two new arms are reported without inferential significance claims.

The numerical unit is a fixed model endpoint. Checkpoints share their learned
trajectory; the common image starts and five panels are not independent
trained-model repetitions. No confidence bands or p-values are manufactured
from them. Clock-profile shading is within-panel standard deviation, not
uncertainty in the model comparison.

Seven figures are fixed: all primary FIDs; three-budget fitting/generation;
stored learning/clipping curves; physical P trajectories; paired clock
profiles; all P eigenvalues at initialization and 20k; and fixed-start cases.
The first eight original starts are shown for the eight models named in
FOCUS in curate.py. They are not selected after viewing new images.

P summaries retain four named families (native K1, fixed-clock, paired feature
and paired time), seven states and all six P matrices: 168 summary rows and
10,752 individual eigenvalue rows. Initial P values must match. All 420 new
clock records, 84 output-head summaries, 84 five-panel mean spans and 70
denoising-panel scores remain available. In the time-only arm, a fixed-time
clock is identical across images, allowing 210 exact local conditional-weight
summaries for W(t)=I+s(t)P. Those matrices exclude SiLU, attention and AdaLN;
their spectra are not full-denoiser Jacobians or stability certificates.

After actual controller exit0 and all audit passes, use the registered source:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct29-cold-clock-source-v1/curation-source/curate.py --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-cold-clock-source-v1
```

Plotting reuses the existing interpreter with Matplotlib 3.10.9 and NumPy
2.2.6. No new package or model evaluation is introduced:

```bash
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct29-cold-clock-source-v1/curation-source/plot.py --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-cold-clock-source-v1
```

Inspect all seven PNGs, verify one-page PDF outputs and explain each major
figure's observation and implication before publication. Preserve raw images,
features and checkpoints outside Git; publish complete derived tables and
the analysis bundle. Archive only after real tool receipts, visual checks and
numeric/source verification. The submitted Report1 remains unchanged.
