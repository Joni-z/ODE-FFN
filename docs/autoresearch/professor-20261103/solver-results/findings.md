# Frozen sampling-resolution check

All six preregistered final checkpoints were evaluated without further training.
Heun32 reproduced every stored generated array exactly before Heun64/128 were
interpreted. Test targets and256 directions stay fixed. Time profiles use the
existing validation field bank: fixed_eval_data's sampling_split argument
selects the sampling targets only; its x/t/oracle always come from validation.

| Task/model | Heun32 | Heun64 | Heun128 |
|---|---:|---:|---:|
|Product16 raw K2|0.107425|0.107224|0.107487|
|Product16 scalar K2|0.088098|0.088069|0.088517|
|Product16 SwiGLU128|0.081848|0.081665|0.082087|
|Ring8 raw K2|0.172697|0.173531|0.174452|
|Ring8 scalar K2|0.181819|0.181628|0.182302|
|Ring8 SwiGLU128|0.164520|0.164687|0.164662|

Entries are test sliced W2, with64/128/256 field evaluations respectively.
The ordering is unchanged across these resolutions. On Product16, raw K2's
sampling gap to SwiGLU remains about0.0254; increasing resolution does not
explain it. The sampled diagnostic therefore weakens a coarse-solver explanation
for this frozen case. It does not certify global integration convergence or
prove the cause of the remaining field-versus-sampling mismatch.

The target-versus-independent-target reference distances are0.058493 on
Product16 and0.153257 on Ring8. These are finite-sample reference values, not
lower bounds on every measured generated-versus-target distance. Differences
near those values need care; no test ranking selects a new checkpoint or budget.

Product16 raw K2 has larger validation error than the shown SwiGLU in every one
of the ten time bins. Against scalar K2 it is worse early and somewhat better
late, despite similar mean field errors. This identifies an error-profile
difference, not a demonstrated transport mechanism. The conditional transport
bound in ../transport-bound.md requires more than these finite local readouts.

Full hashes, timings and profiles: [145](145.json), [Lenovo](lenovo.json).
The first timings include warm-up and are descriptive, not throughput benchmarks.
[Plots](profiles.png). The preregistered primary remains Heun32.
