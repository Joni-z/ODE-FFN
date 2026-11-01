# Lenovo upstream baseline

The user authorized execution on the restored Lenovo on 1 November 2026.
Run the author's native SwiGLU first. This is a local unconditional CIFAR-10
port of JiT; the author's released recipe is for ImageNet. FID below 50 is the
minimum gate before interpreting an ISRU comparison. Read [task_plan.md](task_plan.md)
for the current deployment state; an installed environment is not a quality result.

## Paths

- SSH: `ssh lenovo`, Windows user `jonat`.
- Source: `D:/Research/ODE-FFN/upstream`, synchronized to GitHub main.
- Python: `D:/Research/ODE-FFN/envs/upstream-jit/Scripts/python.exe`.
- Data: `D:/Research/ODE-FFN/datasets/cifar10` (official torchvision CIFAR10).
- Old `repo`, `envs/synthetic`, runtimes and outputs remain intact.

## Environment

[env-spec.json](env-spec.json) declares a separate environment. Its Torch 2.5.1,
Torchvision 0.20.1 and CUDA 12.4 versions follow the author's CUDA profile.
Python is the PSF-signed Windows 3.12.10 installer; auxiliary package versions
are explicitly recorded. The older unsigned managed Python is blocked by
Windows application control. No security policy is changed.

Set `TORCH_COMPILE_DISABLE=1` before Python starts: use eager CUDA execution
on native Windows, with the author's BF16 autocast. Do not install an unofficial
Triton build. Use four CPU threads per process.

## Bounded readiness command

From PowerShell in `D:/Research/ODE-FFN/upstream`, after provisioning the spec
and verified dataset:

```powershell
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:TORCH_COMPILE_DISABLE='1'
& D:/Research/ODE-FFN/envs/upstream-jit/Scripts/python.exe -B docs/lenovo-baseline/check_cuda.py --data-path D:/Research/ODE-FFN/datasets/cifar10 --batch-size 8
```

This executes eight real-data updates through the copied epoch function,
both EMAs, and a short native sampling call. It prints
`CUDA_BASELINE_WITNESS_PASS` only after finite active gradients and outputs.
It measures the actual B model, not the tiny CPU compatibility fixture.
It saves no model checkpoint and establishes no FID claim.

The registered [configuration](train.json) uses batch8 and200epochs. Capacity
checks measured5.292GiB peak allocation and0.323s/update for8 short updates,
including startup; the longer training log determines steady throughput. Batch16
needed7.380GiB and was slower per image. Reports are unchanged by this deployment.
