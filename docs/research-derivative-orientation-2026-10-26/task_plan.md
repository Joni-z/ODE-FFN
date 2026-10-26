# Transpose versus radial derivative computation

Active goal: obtain a useful ODELayer design through mathematical understanding and controlled experiments; use both hosts when available. Previous turn was progress:21 cold-trained complete-gradient readout cells were independently audited. The original ODE fitting gap substantially shrank, but the full generic frontier and Gaussian sampling prevented acceptance. That exact potential setting stays stopped.

- [x] Revalidate the completed parent study and host/process state. No experiment is live; Lenovo remains offline.
- [x] Register a distinct derivative-orientation comparison before new outcomes, including reused controls and fixed-weight crossover measurements.
- [x] Implement a true forward-mode radial JVP with correct parameter/input derivatives. Native LayerNorm JVP higher-derivative failure was isolated before training and corrected with exact native primal values plus explicit derivatives; seven cases pass.
- [x] Freeze36 source files; all three new actual trainers reload exactly, two default parent step3 histories remain exact, two CLI smoke children match, banks match, and45 old potential finite-difference checks pass (max2.03e-9 relative).
- [x] Launch nine new seed0 cells with two one-thread workers and ten-second monitoring; session96092 terminated successfully. Reuse the21 hash-bound parent controls without rerunning them.
- [x] Independently replay fields/Jacobians/samples, audit complete histories, report all crossover cases and apply the unchanged practical standard. Analyzer77517 terminated successfully; audit PASS, native B candidate false.

Status: orientation study complete. The radial directional setting stops. The complete-gradient function remains a useful mechanistic lead, with no accepted practical architecture. Report and saved trajectories are preserved in results/.

No spectral constraint, objective, clock, optimizer, width or potential tuning is introduced. Original frozen runtimes and submitted Report1 remain intact. No git commit/push. The overall research goal remains active until the complete architecture requirements are established.
