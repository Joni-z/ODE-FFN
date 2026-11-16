# Figure catalog

1. **coordinate-comparison** (PNG/PDF). Purpose: compare field fit and final
   generation at the fixed 20k checkpoint. Rows: Ring8/Product16. Columns:
   oracle field MSE multiplied by 1,000 and pooled 32,768-source SW2. Bars
   have no uncertainty marks: these are fixed-function point estimates.
   Keep common-bank protocol, optimizer identities and computation context
   in the caption. Interpretation: Ring8 shared leads; Product16 ordinary
   SwiGLU leads. Fit and generation have different orderings.
2. **paired-banks-and-coordinate-effect** (PNG/PDF). Left: all eight paired
   bank SW2 differences, their means and pointwise empirical-bootstrap 95%
   intervals. Negative favors mean-preserving release. Right: medians over
   six layers and eight fixed batches of the effective mean-step/common-P-step
   ratio at each identical frozen function using fresh native AdamW state.
   These medians are descriptive, not independent inferential observations.
   Interpretation: Product16 sharing survives this coordinate control; Ring8
   is unresolved. Coordinate-dependent update amplification exists at identical
   functions. This does not replay historical optimizer states.

Both PNGs were rendered and visually inspected: labels, axes, legend,
intervals and units are readable without clipping. PDF variants retain
embedded text fonts. Full precision is in comparison.json.
