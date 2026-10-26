# Figure catalog

1. [Endpoint comparisons](figures/augmented-state-endpoints.png)
   ([PDF](figures/augmented-state-endpoints.pdf)). All eight fresh/historical arms
   appear for each task and metric. Heights are ratios to the best eligible
   generic control for that metric, not one shared baseline. Hatching marks
   historical controls; they are not additional repetitions.
2. [Learning curves](figures/augmented-state-learning.png)
   ([PDF](figures/augmented-state-learning.pdf)). All eleven logged training steps
   from each of the six fresh arms are shown. The vertical axis is logarithmic.
   These are oracle-error trajectories, not test-set generalization or uncertainty
   bands. Full train/pair/sample columns remain in `learning-curves.csv`.
3. [Cost and extra paths](figures/augmented-state-cost-and-extra-paths.png)
   ([PDF](figures/augmented-state-cost-and-extra-paths.pdf)). The top row compares
   full-model dense MACs and primary errors, with historical controls marked by
   asterisks. The middle row shows expanded first-solve output and second-solve
   input RMS ratios; the bottom row shows learned extra receiving columns.
   Internal curves average three blocks, not trials. They indicate activity,
   not causal importance, and dense MACs are not latency. Abbreviated point labels
   and staggered Product16 GLU/SiLU annotations prevent overlap; original images
   are preserved in `qa/original-figures/`.

All three figures were inspected for readable labels, correct metric scaling,
control inclusion and unambiguous units. The plotting sources and exact CSVs
are retained. No error bar or repeated-run inference is implied.
