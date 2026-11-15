# 145 readiness witness

Use the existing synthetic environment. All commands below run on145. This
checks five native entries, exact8->16 continuation, three original-cycle13
8step prefixes, initialization, derivatives and nonzero gradient dispatch.
It does not launch20ktraining. Run only on physicalGPU4, which must be idle.

```bash
cd /data2/zz5070/ODE_FFN/professor-followup-20261115/repo
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/data2/zz5070/miniconda3/envs/py312/bin/python studies/muon_pd_20261102/followup_20261115/check.py --device cuda --data-root /data2/zz5070/ODE_FFN/professor-followup-20261115/data --output /data2/zz5070/ODE_FFN/professor-followup-20261115/preflight/documented-v3
```

Required: kernel.json and PASS.json passed=true, five completed entries.
Outputs are deliberately exclusive: inspect failures; use a newly registered
attempt directory after correction. Never remove a failure or re-run blindly.

Attempt1 failed before trainer dispatch: a constant output row annihilates
the centered LayerNorm output. The witness now uses deterministic nonconstant
readout weights. Training initialization is unchanged. The empty attempt1
directory and original command failure are preserved.

## Frozen-evaluator witness

Run after the native-entry witness. This uses the retained Product16 ISRU-D
checkpoint. It checks actual full-state restoration and exact reproduction of
the archived4096sample Heun32 bank, followed by a64sample Heun128 dispatch.
C8 self-tests check the oracle, a non-equivariant linear fixture, idempotence
and rotated trajectories against the native Heun implementation.

```bash
cd /data2/zz5070/ODE_FFN/professor-followup-20261115/repo
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/data2/zz5070/miniconda3/envs/py312/bin/python studies/muon_pd_20261102/followup_20261115/evaluate.py --folder /data2/zz5070/ODE_FFN/professor-autoresearch-cycle13-20261103/output/professor_autoresearch_cycle13_20261103/rotated_product_bimodal_16d/rotated_product_bimodal_16d__pd_isru_k1__adamw_0.0003/seed000 --data-root /data2/zz5070/ODE_FFN/professor-followup-20261115/data --output /data2/zz5070/ODE_FFN/professor-followup-20261115/preflight/evaluator-v1 --device cuda --steps 20000 --witness
```

Required: self-test.json, native-anchor.json, COMPLETE.json passed=true;
COMPLETE.json must say witness=true. No formal generation cohort in this command.

Attempt2 passed the kernel but the old145 v1 cache lacked Ring8 files.
A byte-verified local mirror supplies Ring8 in the new runtime/data/v1;
product16_v1 is a symlink to the original cache. No historical cache changed.
Raw spectrum extraction succeeded; plotting is performed locally because the
warm training environment does not contain matplotlib. No packages installed.
