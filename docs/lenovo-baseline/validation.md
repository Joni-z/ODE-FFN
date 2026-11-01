# Fresh documented validation

Read the compute environment ledger `.aris/compute/lenovo.md`. Run each command
below once from Mac mini, sequentially, after the owner finishes installation
and any capacity probe. Do not install, repair or retry during this pass. Report
exit status, sentinel, actual memory/time and any discrepancy. This is a short
readiness check, not formal benchmark training or a FID-quality result.

```sh
ssh lenovo D:/Research/ODE-FFN/envs/upstream-jit/Scripts/python.exe -m pip check
```

```sh
ssh lenovo D:/Research/ODE-FFN/envs/upstream-jit/Scripts/python.exe -B D:/Research/ODE-FFN/upstream/main_jit.py --help
```

```sh
ssh lenovo D:/Research/ODE-FFN/envs/upstream-jit/Scripts/python.exe -B D:/Research/ODE-FFN/upstream/tests/check_upstream.py --upstream D:/Research/ODE-FFN/references/jit-cbc743a2
```

```sh
ssh lenovo D:/Research/ODE-FFN/envs/upstream-jit/Scripts/python.exe -B D:/Research/ODE-FFN/upstream/docs/lenovo-baseline/check_cuda.py --data-path D:/Research/ODE-FFN/datasets/cifar10 --batch-size 8 --check-metrics
```

Expected: pip reports no broken dependencies; CLI help includes the FFN switch;
reference check emits `UPSTREAM_COMPATIBILITY_PASS`; CUDA check emits
`CUDA_BASELINE_WITNESS_PASS` with finite active gradients, both EMA updates,
short native sampling and four actual official Inception2048 feature vectors.
The author code can emit AMP deprecation or torch.load FutureWarnings. Do not
interpret those as quality results. No long training, FID extraction for a
benchmark, checkpoint writing, report editing or package changes are requested.
