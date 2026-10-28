# Fixed analysis layout

This layout is registered while the one formal training process is live and
before any new generated-image quality score. It does not change the formal
source snapshot, optimization, sampler, reference banks or acceptance rule.

The analysis reads all32historical endpoint rows and all three new rows at
5k/10k/20k. Its primary table contains all15models/interventions at20k.
The initializer's decision uses only the registered default-Swi86 contrast:
FID at least5%lower, weighted and clean MSE no more than5%higher. All other
contrasts are reported as continuous differences. No new ODE method follows
from a successful ordinary baseline initialization.

Figures retain all15primary FIDs, the eight named models in FOCUS in
curate_init.py, all63new gate/value/output matrix summaries and all21branch
norm rows. The input/dynamics matching witness does not imply equal parameter
tangents or optimizer trajectories. For a zero matrix the singular-value CV
is undefined; leave it blank rather than assigning an apparent zero dispersion.

The case grid uses the first eight original Gaussian starts, at20k, for native
K1, fixed-clock0.6, matched SiLU64, default Swi86, matched Swi86 and Swi129.
These48examples are illustrative fixed cases, not separately selected winners
or independent trained models. The full earlier comparison/case grids and
physical/inferred P histories remain in the sealed cold-control study.

All comparisons are fixed-budget measured differences. Checkpoints share a
training trajectory and starts share its learned model. No inferential
p-values, confidence bands or population claims are created from these rows.

After the actual controller terminal tool exit0and all six phase exits0, run
the source frozen into curation-source; never read status.json alone as proof
of successful terminal execution:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct29-image-swiglu-init-v1/curation-source/curate_init.py --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-image-swiglu-init-v1
```

Plotting uses the existing Mac mini plotting interpreter (Matplotlib3.10.9,
NumPy2.2.6, Pillow12.3.0); no new package is installed. It consumes audited
tables and fixed raw images only, without constructing or training a model:

```bash
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct29-image-swiglu-init-v1/curation-source/plot_init.py --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-image-swiglu-init-v1
```

Inspect all five PNGs, verify their corresponding PDF files, and explain what
each figure changes in the decision. Only then publish the analysis bundle
and seal the private runtime with hashes of all preserved files. The submitted
Report1 and all frozen parent studies remain unchanged.
