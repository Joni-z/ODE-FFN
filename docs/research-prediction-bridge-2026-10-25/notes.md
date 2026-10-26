# Execution notes

- Existing Stage1–24 studies are terminal. No useful ODE architecture has passed the recorded strong-baseline criteria.
- Earlier screens used direct velocity prediction and uniform time, so their failure does not directly measure the image-prediction configuration discussed with the user.
- This experiment changes prediction space within one common declared recipe and compares against both prediction-space versions of SwiGLU. A generic x-prediction benefit is not an ODE benefit.
- Execution host: Mac mini M1, torch2.10.0, approximately4.97GiB available memory at preparation,98GiB available disk. Two CPU workers are appropriate; no GPU is reserved.
- Training seed0 only. Offsets used to define data/evaluation generators do not introduce additional training seeds.
- Existing dirty repository changes and historical Report1 are preserved; new runtime code is isolated in a new runner.
- Eight unittest methods passed, including loss-and-gradient equivalence inside and outside the clipped region, raw-state/backbone pairing, original recurrence, actual projection-call compute counts, conversion Jacobian, and deterministic clipped-AdamW updates.
- The successful preflight ran three actual three-update cells and reloaded their final checkpoints; all three evaluation measures replay exactly. Cross-arm actual training inputs match. No preflight outcome is used to choose a model.
- Formal execution is session16497. Monitoring at10:12:10UTC found two training Python workers of approximately295–297MiB RSS and4.95GiB available memory;6/24 cells were already complete. No GPU is reserved.
- All24 formal cells completed at10:15:58UTC; the same session16497 closed with exit0. The analyzer ran once, session85151, exited0, and recomputed every final field/derivative/sample metric. Actual initial pairing and96checkpoint input-stream/RNG checks pass. No nominee survives the registered rule.
- Ring8 primary MSE: ODE K2-v .153102 versus SwiGLU192-v .083171. Product16 best ODE K1-v .332418 versus SwiGLU192-v .199982. Image prediction does not rescue the ODE comparison. All metrics and arms remain in results/comparison.csv; no favorable metric is substituted for the registered primary measure.
- This fixed setting stops. No long-budget extension, fresh test consumption, isolated speed benchmark, schedule/width/initializer rescue or additional architecture has been launched. No ODE superiority or full JiT result is claimed.
