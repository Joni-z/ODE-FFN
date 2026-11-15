# Response-screen execution notes

Question: compare five matched bounded odd responses, normalized SiLU/GELU and
linear, at K1/K8 on Ring8/Product16. Keep seed0,20k,batch512,AdamW3e-4.
Preserve the pinned native trainer/sampler and shared entrance-frozen D.

Previous evidence: docs/research-assessments/2026-11-15-order-response/findings.md.
The earlier tanh-only image proposal is superseded by comparative selection.

145 warm py312/Torch2.8.0+cu128; no package changes. /data2 free116GB.
Data reuses professor-followup-20261115/data with byte checks.
Selection stream2026111500 and confirmation2026111600 are evaluation RNGs.
Prebind both sets before training and keep confirmation closed until shortlist.

GPU4/5 already occupied;6/7 became occupied between inventory and setup.
Do not share another task's cards or stop it. Recheck before GPU work.
User explicitly revoked all prior permission to stop yp2841 tasks. No tasks
were stopped. Torch scheduler allocations are the fallback while145 is busy.
Historical Ring8 linearK1 was on Lenovo; inspect145 for a mirrored checkpoint.
If absent, explicitly register one deterministic seed0 replay within this wave.

Current stage: Mac mini CPU preparation at the user's request. Existing local
synthetic cache is present. Complete all32 focal trajectories on the same CPU
host so the new response comparison does not mix training backends; preserve
historical weights as context. This changes run count from23missing/replayed
to32same-host runs and will be budgeted from actual200step timings.
Queue3761 now trains36same-host trajectories with2CPUworkers. The fresh
documented check passed all16 continuations and the real evaluator witness.
The benchmark selected2workers at1.889x aggregate throughput. Watcher3767
automatically measures cost, independently recomputes selection metrics,
freezes a shortlist and evaluates disjoint confirmation arrays afterward.
Training sourceb5fe7aa; analysis0e78ce7. No GPU job stopped or Torch allocation
submitted. Current progress and outputs are under response-screen-20261115.

Latest partial audit:19/36 pipelines completed. Ring8 SwiGLU64/Muon SW2
.11823; Softsign-K1 .09097 and Softsign-K8 .06956. All Ring8 response families
are now compared; Product16 is in progress. The earlier pending-baseline
statement referred only to this CPU rerun, not to historical synthetic/image
baselines. Keep the registered cross-task selection and disjoint confirmation.
All19 completed20k arrays independently recompute within2.8e-17.
