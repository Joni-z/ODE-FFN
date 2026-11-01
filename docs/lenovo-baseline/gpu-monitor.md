# Interactive GPU monitor on Lenovo

Verified on 1 November 2026: a fresh interactive SSH login resolves `nvitop`,
opens the live GPU/process display, and returns to the shell when `q` is pressed.

```text
ssh lenovo
nvitop
```

Use `nvitop -1` for a single snapshot, or `ssh lenovo nvitop -1` from the Mac.
GPU utilization, total device memory, temperature, power, and the training PID
were visible. The training process was PID 16984, GPU utilization was 98–99%,
and device memory was 6,346 MiB / 8,188 MiB during verification. Windows WDDM
reported per-process GPU memory as `WDDM:N/A`; this field is unavailable here.

## Installation

The [official nvitop instructions](https://github.com/XuehaiPan/nvitop#installation)
support Windows through `windows-curses`. An isolated Python 3.12.10 environment
uses the existing signed Python installation:

- Environment: `D:/Research/ODE-FFN/envs/gpu-monitor`.
- Command: `C:/Users/Jonat/.local/bin/nvitop.cmd`.
- The command directory was already on the user PATH; no PATH or SSH setting changed.
- Training environment: `D:/Research/ODE-FFN/envs/upstream-jit`, unchanged.

The wrapper is:

```bat
@echo off
setlocal
set PYTHONIOENCODING=utf-8
"D:\Research\ODE-FFN\envs\gpu-monitor\Scripts\python.exe" -m nvitop %*
```

The UTF-8 setting is scoped to this command and fixes redirected SSH output.
No global console or PowerShell execution policy was changed.

Installed versions: nvitop 1.7.2, nvidia-ml-py 13.615.71, psutil 7.2.2,
windows-curses 2.4.2, and colorama 0.4.6. Direct download on Lenovo stalled;
all five official PyPI wheels were relayed through the Mac, their SHA-256
hashes verified on arrival, and installed offline. `pip check` passed.
Package cache: `D:/Research/ODE-FFN/cache/nvitop-1.7.2`.

Verification covered both a fresh interactive SSH session and UTF-8 one-shot
output. Training continued throughout; its environment, source, command,
checkpoint, and scheduled task were not modified or restarted.
