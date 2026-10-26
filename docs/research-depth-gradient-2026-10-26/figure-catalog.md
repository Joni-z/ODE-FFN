# Figure catalog

## all-family-depth-modes.png / .pdf

Purpose: determine whether tying removes unusually large gradient energy in ODE relative to the generic controls. Source: independently verified output/mean-modes.csv, bank A, all complete-block coordinates, updates1000/2500/5000. Stacks are exact constant/linear/quadratic depth-mode fractions. Labels inside blue bars give retained fractions. No error bars; zero-gradient initialization is omitted.

Observation: potential-gradient readouts retain similar final fractions across the three families; the restriction does not track the much larger cold ODE sharing penalty uniquely. Interpretation must remain in native Euclidean coordinates and cannot equate mode fractions with achievable task error.

Checks: all controls shown, both readouts shown, sums equal1, no undefined initialization plotted, title describes excluded directions without asserting a successful architecture.

## P-selection-and-transfer.png / .pdf

Purpose: show the actual prospective decision and distinguish it from secondary evidence. Sources: output/P-selection.csv, mean-modes.csv, batch-modes.csv. Upper left: P retention for both bank means plus eight bank-A batches per late state; dotted cutoff .50. Upper right: share of blocked energy attributable to P and linear share of lost P energy, with their .50/.70 cutoffs. Lower left: normalized bank-A descent direction alignment with bank B. Lower right: pairwise P-gradient cosines from the bank-A mean.

Observation: the severity cutoff fails, while secondary energy/transfer conditions pass and bank-mean pairwise directions are positive. This stops P-depth training rather than declaring a failed direction beneficial. Batch dots are not training replicates or confidence intervals. Positive cosines do not prove absence of all per-example or optimizer interference.

Checks: final labels/legends visually inspected; secondary legend separated from bar labels; no post-result threshold changes. Both PNG/PDF files are generated from the same plotting call.
