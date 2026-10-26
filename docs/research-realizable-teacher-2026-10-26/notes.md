# Evidence and execution notes

The prior generator-tangent turn was progress, not goal completion. Its sealed report and next-question file motivate a realizable-target control; no trained-ODE order-gradient cancellation was found. The larger objective remains finding and validating a useful ODELayer architecture under strong controls.

The six new cells change only the deterministic regression teacher. They are not architecture improvements or a reuse of old FM endpoints. The exact-representer control uses cloned teacher weights in a separate nontraining instance and must not leak those weights into any cold student.

Current resources: Mac mini8logical CPUs,4.88GiB available RAM. No architecture-search Python task was observed alive. SSH Lenovo timed out; no remote task assigned. Two one-thread CPU workers are appropriate for these small models. Reuse old model/data/training primitives where possible and freeze new glue independently.

The source input generator uses the historical fixed derived data state100000 for training seed0. The actual model seed remains0, and the formal six cells share one input stream. Existing buffers and initial state snapshots are authoritative; platform-generated replacements are not silently substituted.

The first full preflight passed before formal launch. A code-review follow-up added hashes of the **actually consumed** x/t/label tensors, beyond the preparer's original target/source/time chain. The original passing preflight and its exact source are retained in preflight-source-v1. A focused stream/snapshot delta check verifies the added bookkeeping; no model formula, input, label, optimizer or threshold changes. Formal readiness requires both receipts.

Formal source9files froze before launch. Supervisor session40507 ran once, actual PID1521; worker PIDs1535,1536,1538,1540,1543,1551 each completed5000updates, and the supervisor exited0. The six complete rows were retained. Source/input hashes and recorded Fourier buffers remain unchanged.

Independent audit source froze separately in audit-source/ before replay. Session33667/PID1596 exited0; full5000input batches and both teacher label streams match bitwise, all24CP/full fields and30native endpoint panels replay, and separate complete float64 forward/Jacobian reconstructions have zero discrepancy. The exact teacher representers match all20000field outputs. No audit failure, job restart, architecture alteration or result-dependent criterion change occurred.

The specific learning-difficulty criterion is false. An ODE student can fit this own-class function at0.629%relative MSE, although both ordinary students fit it better. Its10.652%error on the Swi teacher contrasts with0.299%/0.573%for the generic students, but does not establish that the richer target lies outside the ODE class. Teacher energies and derivative energies are reported separately. No new architecture is selected.

An intermediate status read imported psutil with system Python and failed before observing files. It was replaced with a standard-library read, followed by polling the same live supervisor handle; no process was restarted. Final models, optimizer states, source/cache bindings, monitor records and all checkpoint arrays remain authoritative.

The hypothesis-generation, experimental-design, daily-coding and planning-with-files procedures remain in use. Previously verified workflow provenance is Kassis, Agarwal, He, Patel and Brueckner (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, https://doi.org/10.48550/arXiv.2609.00065 . This acknowledgment is not empirical model evidence.
