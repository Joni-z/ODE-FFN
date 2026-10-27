# Cycle13 figure catalog

All figures use real audited tables; PNG/PDF pairs are available. They have no significance markers or inferential error bars: values describe the full fixed bank of20,000states per checkpoint. Models/layers/checkpoints share sources and are not independent repetitions. All three PNGs were visually reviewed for labels, legends, clipping and agreement with the CSVs.

## Local action

- Files: [PNG](results/figures/local-action.png), [PDF](results/figures/local-action.pdf).
- Purpose: distinguish raw quadratic-term energy from action that cannot be absorbed by one fixed matrix and scalar clock.
- Data: [local.csv](results/local.csv), all36final layer/checkpoint pairs, pooled states.
- Variables: three heatmaps show raw quadratic RMS, centred best bounded-clock residual, and centred original-clock residual; each is normalized by native increment RMS. Each panel has its own labeled color scale. Columns are all block/solve positions.
- Caption requirements: A=P+.3P² is fixed analytically, the box is [.2,1], and the relaxed coefficient is not a trained finite controller. No family/layer is omitted.
- Notice: raw terms can be large while centred residuals are small for input-frame layers; thick original/global layers retain larger residuals.
- Decision: neither raw energy nor success of a separately trained K1 justifies declaring frozen K2 redundant. Future order-specific claims need the matrix-reparameterization control.
- Check: all six solves and both laws appear; no ratios of per-source norms or shared misleading color scales.

## Checkpoint response

- Files: [PNG](results/figures/checkpoint-response.png), [PDF](results/figures/checkpoint-response.pdf).
- Purpose: show all saved trajectories and the difference between local approximation and actual complete prediction change.
- Data: local.csv, [complete.csv](results/complete.csv), [intervention-parts.csv](results/intervention-parts.csv).
- Variables: maximum pooled local centred boxed-clock residual across six solves, actual full-field displacement, and reporting-only tangent displacement versus training update. Solid/dashed lines distinguish thin/thick laws; all six families/laws use the same checkpoints.
- Caption requirements: the1%horizontal guides in the first two panels are registered descriptive references only. Tangent normalization is a reporting split, not a new acceptance threshold. Cold denominators use the actual decoded field.
- Notice: thick original/global functions become progressively less preserved by the fixed-centre substitution; full output displacement is much smaller than tangent displacement.
- Decision: assess subspaces separately and do not infer whole-network preservation by adding local percentages.
- Check: no best-checkpoint selection, maxima are labeled, omitted undefined values are not silently replaced by a favorable score.

## Risk accounting

- Files: [PNG](results/figures/risk-accounting.png), [PDF](results/figures/risk-accounting.pdf).
- Purpose: show how a finite change aligns with existing tangent prediction error.
- Data: intervention-parts.csv, all six final tangent rows.
- Variables: displacement energy, twice the error/displacement inner product, and their signed sum, as percentages of original tangent oracle risk. The decomposition is exact on the fixed bank, within floating-point checks.
- Caption requirements: it is a postregistration subspace split of the single registered intervention, not new training, interval ablation or an SW2 decomposition.
- Notice: cross terms can oppose or reinforce displacement; thick original/global tangent risks rise25.17%/36.69% despite modest full aggregate changes.
- Decision: separate useful-error alignment from the magnitude of the K2 term; retain the existing model/generation stop.
- Check: signs and units are retained, all final models appear, negative risk changes are not promoted to method wins.
