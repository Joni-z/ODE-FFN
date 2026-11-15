# 145 readiness witness

Use the existing synthetic environment. All commands below run on145. This
checks five native entries, exact8->16 continuation, three original-cycle13
8step prefixes, initialization, derivatives and nonzero gradient dispatch.
It does not launch20ktraining. Run only on physicalGPU4, which must be idle.

```bash
cd /data2/zz5070/ODE_FFN/professor-followup-20261115/repo
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/data2/zz5070/miniconda3/envs/py312/bin/python studies/muon_pd_20261102/followup_20261115/check.py --device cuda --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --output /data2/zz5070/ODE_FFN/professor-followup-20261115/preflight/documented-v2
```

Required: kernel.json and PASS.json passed=true, five completed entries.
Outputs are deliberately exclusive: inspect failures; use a newly registered
attempt directory after correction. Never remove a failure or re-run blindly.

Attempt1 failed before trainer dispatch: a constant output row annihilates
the centered LayerNorm output. The witness now uses deterministic nonconstant
readout weights. Training initialization is unchanged. The empty attempt1
directory and original command failure are preserved.
