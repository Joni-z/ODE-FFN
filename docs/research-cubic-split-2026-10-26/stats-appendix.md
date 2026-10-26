# Measurement and audit scope

This is a fixed descriptive synthetic screen. Each dataset/family trajectory is a fitted model; panel rows, coordinates and snapshots are repeated measurements rather than independent training replications. No p-value or training-variability confidence interval is claimed. Historical controls retain their reused status.

The primary risk is per-coordinate oracle velocity MSE on the fixed uniform-time bank:10000 rows for each2D task and20000 for Product16. Secondary measurements are five512-row time panels, full input-Jacobian squared Frobenius error and common-source2048-sample32-step SW2. The underlying data split is validation, not a new test set. The data-stream generator initialization100000 preserves the historical stream; model training seed is0.

The audit reconstructs all15000 distinct dataset-stream batches directly and checks snapshot digests against every family. It restores actual runtime buffers and replays84 checkpoints,105 endpoint panels and21 sampling outputs. It independently verifies72 P spectra using NumPy, and18 trained local forward/Jacobian/adjoint comparisons using a separate matrix-polynomial expression. Maximum discrepancies are2.13e-14 for forward values,1.89e-15 for Jacobians and7.11e-15 for parameter adjoints.

The ready file binds227 source/input artifacts and130 prior documents. Existing histories and raw files remain unchanged. Analysis exports all21 outcomes,105 panels,84 validation measurements,231 training windows,21 costs,72 P trajectory records and4608 spectrum-coordinate rows. No result row is omitted for failing an eligibility guard.

Operational advancement criteria are recorded before outcomes in protocol.md. Every field and derivative guard fails; Product16 also fails the free-ISRU mechanism comparison. These gates organize research decisions and are not statistical significance tests or impossibility theorems.
