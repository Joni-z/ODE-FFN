# Pruned augmented-state prototype: independent code review

2026-10-25. **PASS for the reviewed implementation and registered Phase A configuration, subject to the separate launch preflight.** No formal training or outcome evaluation was performed for this review. Passing implementation checks does not establish a useful architecture or complete the project goal.

## Verified implementation

Compared the changes directly with the actual Stage23 595-file runtime, not an outdated Git index. Existing model branches are unchanged; additions are the new helper/kind, dedicated collector, CLI choice and functional-projection MAC accounting. The independent preflight still needs to verify historical execution and resume compatibility.

The first map executes `pad(h)+s1(h)*U1V1h`; the second executes `y[:64]+s2(y)*U2V2y`, with one middle SiLU. The first projection/controller take64 inputs and return128 coordinates; the second take128 and return64. No permanently zero input columns or discarded output rows are parameterized. Both factor products use functional linear operations, counted exactly once through `projection_macs`; controller Linear hooks account for their actual executions. There is no dense substitute P in the training forward.

Construction consumes the original VectorODEFFN reference draws once. CPU float64 SVD factors are cast to the reference dtype/device; newly constructed controller Linears stay inside an RNG fork. Controller subsets come from the recorded random reference, and final controller maps are zeroed internally. The new family's `ode_solve_init='legacy'` config does not mean random initial clocks. Reference guards reject incompatible clocks, map types and dimensions.

Narrow/expanded common tensors and surrounding draws agree. Extra U1 rows copy the nonzero common producer; extra V2 columns and second-controller input columns start at zero. Initial functions agree algebraically within dtype-specific GEMM tolerance. At the native zero velocity head, backbone responses vanish and complete initial kernels agree. With a nonzero receiving head, extra V2 responses are live and kernels generally differ. The expanded U/V pair is not globally SVD-balanced after copying extra U1 rows; no post-hoc rescaling is performed.

The full zero-padded operators and reduced `V D U` matrices are correct for both rectangular roles, including batched durations. The reduced eigenvalues do not see first-map extra U rows or second-map extra V columns at fixed D. Collector labels correctly limit the reduced matrix to probe-mean clocks and distinguish it from the complete input Jacobian. Spectral calculations ultimately use CPU NumPy; no MPS float64 spectral operation is required.

New layers stay outside legacy `ode_layers`. Consequently, legacy `anchored_jacobian` is equal to native for this family and is not a gate-derivative ablation. The protocol states this explicitly; native whole-network field/Jacobian evaluation remains valid.

## Independent checks

`/Users/mr.z/ode-workspace-prep/.venv-test/bin/python -m unittest tests.test_augmented_state -v` passed all **7 tests in0.164s**. They cover independently derived input/factor/controller derivatives, the complete SiLU chain, native RNG/backbone pairing, executed dense operations, SVD truncation and balancing, masked versus live parameter responses, dead-branch counterexamples, padded-generator identities, actual statistics and invalid settings.

Executed `F.linear` counting agrees with the runner, rather than merely checking a repeated formula. Product16 totals are:

| Arm | Parameters | Dense MACs |
|---|---:|---:|
| Dense parent / unexpanded factor32 |83,664|81,984|
| Unexpanded narrow21 |66,702|65,088|
| Expanded21 |83,022|81,216|

The tests also cover2D totals. These are operation counts, not latency claims.

A separate full-width MPS float32 check used8 random16D inputs and an explicit nonzero head. Forward values and every parameter gradient were finite; all6 collector records serialized without nonfinite values. Extra V2 gradient norms across the three blocks were `.00205421937`, `.00180524553`, `.00146212592`. There were **zero optimizer steps**. All6 implementation/test file hashes below matched before and after this check; it is not a training run or speed benchmark.

## Configuration and decision scope

Both task plans contain the same six arms, seed0,5000 updates and two CPU workers; the queue covers12+6 cells. Independently verified all18 backbone-reference entries against their3 distinct checkpoint hashes. Low-rank arms use explicit dimensions and the declared initializer; data/target/head/optimizer changes are absent. The runner inherits the existing bank-pair trainer settings.

The same-budget factor32 control is present. The narrow control is correctly treated as cheaper rather than a same-cost augmentation comparison. Protocol decisions separately credit an augmentation lead, a factorized lead or a cheaper quality-preserving narrow lead. Historical SwiGLU96/GLU128 stay in the generic frontier without becoming new cells or repetitions; their outcome provenance must be pinned and verified by preflight/analysis, separately from the backbone-reference hashes. Larger SwiGLU models are context only.

A lead still requires separately registered controls, matched cheaper SwiGLU84 where applicable, isolated timing, fresh evaluation and transfer. Failure closes this allocation without a rank, width, initializer, clock or order rescue. The implementation differs from legacy `src/ffn_blocks.py:549` LowRankStateODEFFN, which has learned input/output projections, a diagonal-plus-low-rank generator and an extra output gate. Neither family name establishes novelty.

## Stable reviewed inputs

| File | SHA256 |
|---|---|
| `synthetic/augmented.py` | `0d7cfa085e096518ddff9e035b15402fd022fe1ba06643b00b131ea07c3008ee` |
| `synthetic/models.py` | `85c3dabd318ad30eecd8d89c5513d0c1d31b690fc20fd89cde6fd5f2962e44aa` |
| `synthetic/diagnostics.py` | `58c9d56d57c0b4435143b74c40841514b998aad19fa7496c3ce0aa0a5efc1561` |
| `synthetic/train.py` | `3657e26c610340d69a723c08a18d64317b5bf4f16aedb281e84c770c4e47aa06` |
| `scripts/run_architecture_study.py` | `cf347594c54e34721007297b3013469fb8e518a80401bd46438cc0c9cefbcd09` |
| `tests/test_augmented_state.py` | `671d442eaced8052c8787f61a7d3e5565713e6efc8bfaaa262e5e5c1b4da6b6d` |
| `protocol-augmented-state.md` | `852ad5cc013e45dffda978760eb153af0c0ed6cf2cbf059d2fd63f2ca7ff15a1` |
| `augmented-state-design-review.md` | `6e3e8a9029096564d1f6b9b2f801b6e863c01486c45be2a5fde2c949c2f814eb` |
| `augmented-state-prior-review.md` | `ba8a6e4bfa7a6aa70e60d58ad02653029aa8af80f6c2c07ab79ee8388eef0eb4` |
| `augmented_state_v1_20261025_m1.json` | `311bf87b3e27f5b30b0c934096f45380c28baa264ebbfbf4f505c6eedb9776ae` |
| `augmented_state_product16_20261025_m1.json` | `7684cd0f3e00406882caa31f24f6a0f43f69048e25eb37aa71654e5b49ffdffc` |
| `augmented_state_queue_20261025_m1.json` | `cdd23e0da67da2d5c2058aced4727789e26ae60b224437b6b376885526cc2775` |

Protocol/review paths are under `docs/research-architecture-search`; plans are under `configs/synthetic/architecture_search`. This review edits only this new review file. Source, protocol, plans, tests and frozen evidence were not modified by the reviewer.
