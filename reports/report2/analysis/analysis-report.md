# Report2 evidence synthesis, 3 November 2026

The formal report now centers the professor's two suggestions. It separates
Muon update geometry from learned P, and separates diagonal feature gains from
an ODE-specific architectural gain. The latest ISRU-D result is retained with
the ordinary free-ISRU control and both optimizer settings.

| Finding | Evidence | Report decision |
|---|---|---|
| Muon broadens updates while P grows | Original Product16 scalar trajectories and replayed endpoints | Show both spectra; do not equate update regularity with stable weights. |
| Direct shape constraints worsen the tested fits/samples | Frozen shape/scale controls and constrained-training study | Retain the negative setting, without claiming all constraints fail. |
| Different D functions compute different nonlinear features | Exact K1 derivative, feature/order/controller experiments | Present various designs and their purposes rather than a single arbitrary gate. |
| ISRU-D is competitive on Ring8 | All18latest endpoints, training curves and exact references | Retain the task-specific positive; its2.25%field lead over free ISRU fails the registered continuation gate. |
| Ordinary controls explain much of the benefit | Free ISRU/Swi64/128/192, Product16 reversal | No accepted ODEFFN method or image-transfer claim. |
| Earlier readout/frame designs improved specific tasks | Preserved earlier source tables | Keep concise context with ordinary controls and thick-task failure. |

This is reporting from completed results. No new statistical test, model
selection, training or sampling was performed. The new table lists every
registered latest configuration. Individual layers, checkpoints and optimizer
settings are not treated as independent replications. See the
[claim map](../EVIDENCE.md) and [verification](../verification.json).
