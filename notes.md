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

Execution history: Mac mini CPU preparation at the user's request. Existing local
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

Final audit:36/36 training/selection pipelines completed, all exits zero.
Selection72rows and confirmation12rows independently recompute within5.56e-17.
ISRU/arctan shortlist was frozen before confirmation. Both queue and watcher
have exited; no synthetic jobs remain in this wave. The pending-baseline
statement from earlier updates referred only to the new CPU rerun.

Confirmation pooled SW2 Ring8/Product16:
SwiGLU .141952/.039954; Free ISRU .145836/.058042;
ISRU-K1 .148122/.065610; ISRU-K8 .141806/.063602;
arctan-K1 .125849/.062764; arctan-K8 .135097/.060880.
Ring8 arctan-K1 wins all8paired banks versus SwiGLU; Product16 loses all8.
Softsign-K8 selection .069559/.060901 misses the Product16 MSE gate
(.020125 versus worse(ISRU,tanh)=.018331), so has no confirmation result.

Recommendation: arctan-K1 is the first new response for a bounded JiT
readiness benchmark, with K8 retained only as a comparison. Its warmed
CPU forward cost is6.82x/4.70x SwiGLU on Ring8/Product16; image GPU cost must
be measured rather than inferred. No response demonstrates cross-task
superiority; parameter-matched Free ISRU also beats the ODE family on Product16.
No new CelebA submission. Preserve native image baseline and finalized reports.

Full evidence and interpretation:
docs/research-assessments/2026-11-15-response-screen/analysis-report.md.
