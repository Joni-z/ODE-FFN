# Professor follow-up execution notes

User authorizes145 GPUs4-7, with Torch available as overflow. GPU0-3 belong
to another user. Reuse /data2/zz5070/miniconda3/envs/py312 (Torch2.8.0+cu128).
Both v1 and product16_v1 caches exist under muon-pd-20261102/data.
Product16 cycle13 raw weights/spectra are retained on145; Ring8 originals
were on Lenovo. Existing local cycle14/15 ZIP files are evaluation outputs.

Prior completed analysis: [three comments](docs/research-assessments/2026-11-15-professor-comments.md).
The execution protocol and status will be bound to a separate15November study.
No changes to final reports, the image trainer, or the original baseline.

The old145v1 directory exists but lacks Ring8. Local original
/Users/mr.z/ode-workspace-prep/data/synthetic/v1/gmm_ring8_2d and its manifest
match every registered SHA256. Copy into the new runtime/data, leaving old
caches intact. Product16 is symlinked. Readiness3uses this new data root.
