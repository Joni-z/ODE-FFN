# Figures

1. **paired-endpoints.png / .pdf.** All eight original/paired groups on all
   three targets, showing primary oracle MSE, complete Jacobian error and
   validation SW2. Each panel divides by its better paired SwiGLU value; the
   horizontal line is ratio1. Original ODE rows are reused, not rerun. The
   plot makes the retained nonlinear ODE deficit and metric disagreement
   visible while showing both sets of baselines.
2. **paired-learning.png / .pdf.** Complete oracle-error trajectories through
   5000 updates, using logarithmic y axes. Dashed lines are the original
   SwiGLU models; solid blue lines are paired SwiGLU. All four old ODE arms
   remain visible. These curves describe learning and are not checkpoint
   selection rules or isolated wall-clock benchmarks.

Both figures use only audited records. Their PNGs were visually inspected for
clipping and readability. Source is preserved in `source/analyze_backbone_pairing.py`;
input hashes and the source hash are recorded in `provenance.json`.
