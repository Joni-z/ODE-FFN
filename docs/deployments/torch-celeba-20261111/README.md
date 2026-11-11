# torch development and CelebA setup

11 November 2026. The user explicitly moves new development to torch and
authorizes dataset/environment preparation under scratch. No training is
launched by this setup.

## Paths

| Purpose | Path on torch |
|---|---|
| Active development checkout | `/scratch/zz5070/ODE_FFN/ODE-FFN` |
| Initial CUDA12.4 environment, preserved | `/scratch/zz5070/ODE_FFN/envs/jit-celeba` |
| CUDA12.8 environment for all GPU types | `/scratch/zz5070/ODE_FFN/envs/jit-celeba-cu128` |
| Original aligned CelebA | `/scratch/zz5070/ODE_FFN/datasets/celeba` |
| Images | `/scratch/zz5070/ODE_FFN/datasets/celeba/img_align_celeba` |
| Setup logs and local receipts | `/scratch/zz5070/ODE_FFN/operations/torch-celeba-20261111` |
| Future outputs | `/scratch/zz5070/ODE_FFN/runs/celeba` |
| Historical checkout, preserved | `/scratch/zz5070/ODE_FFN/jit-codebase` |

The historical checkout is clean at `2d2259a` but contains the older research
framework and links to historical outputs. The new checkout uses canonical
GitHub main. Do not update the other `/scratch/zz5070/jit-codebase` worktree,
which the workspace record identifies as separately modified.

Initial `myquota` reports a 5 TB scratch allocation, about 1.45 TB used and
1,386,140 of 5,000,000 files. This is a setup-time snapshot, not a live quota.

## Environment

The new conda environment is installed from
[requirements-cuda.txt](../../../requirements-cuda.txt), following the pinned
author's CUDA stack. Verified versions are Python 3.10.22, PyTorch 2.5.1+cu124,
torchvision 0.20.1+cu124, NumPy 1.22.4, SciPy 1.9.1 and timm 0.9.12. The
author's torch-fidelity fork is retained; nvitop 1.6.2 is installed.

The subsequent [K8 training deployment](../../celeba-k8-20261111/validation.md)
adds a CUDA12.8 environment for Blackwell as well as older cards. The current
activation script selects that environment. Activate from the checkout with:

```bash
cd /scratch/zz5070/ODE_FFN/ODE-FFN
source scripts/activate_torch.sh
python main_jit.py --help
```

The activation script places pip, Torch, Hugging Face, TorchInductor and
Triton caches under the scratch project root. It does not change shell startup
files or the existing synthetic environment. `pip check`, package imports,
NumPy/Torch conversion, PIL/torchvision conversion and the training CLI pass.
The pinned upstream compatibility fixture passes the state/RNG, gradients,
optimizer/EMA, sampler and checkpoint checks on CPU.

The initial CUDA runtime check `22652701` completed0:0 on H200, passing matrix
and convolution forward/backward. The newer CUDA12.8/K8 checks are recorded
separately with the training deployment. The login node has no allocated GPU.

`nvitop` is available after activation. Login nodes do not expose allocated
GPUs; inspect it inside a Slurm GPU allocation. The CUDA 12.4 profile follows
the author's H200 stack. The separate CUDA12.8 profile provides Blackwell
support; its kernel validation is part of the new deployment.

## Dataset

Prepare original **CelebA**, not CelebA-HQ. Use torchvision's official CelebA
download implementation and its Google Drive IDs/MD5s, pinned by torchvision
0.20.1. [Source implementation](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/datasets/celeba.py).

The [preparation wrapper](../../../scripts/prepare_celeba.py) invokes that
implementation, then checks all six official file MD5s, archive CRC, the
202,599 expected image filenames, official split counts and 32 fixed JPEG
decodes. It preserves original images and metadata. Training resolution,
cropping and final evaluation reference are separate protocol decisions.

Verified split counts are 162,770 train, 19,867 validation and 19,962 test.
The original Google Drive archive exceeded its download quota. All six files
were instead downloaded from the public [Yuehao/celeba mirror](https://huggingface.co/datasets/Yuehao/celeba/tree/35ebb31405e9da028409199a494f2e77eb10f253),
pinned to revision `35ebb31405e9da028409199a494f2e77eb10f253`. Every file matches
the official torchvision MD5. Source URLs and SHA-256s are recorded in
`datasets/celeba/source.json` on torch.

All **202,599** original RGB images are extracted at **178 x 218** pixels.
The six official MD5s, archive CRC, complete filename sets, metadata/split
counts and 32 fixed JPEG decodes pass. Extraction reused standard Python ZIP
reading with bounded I/O concurrency because sequential scratch writes were
slow; the torchvision dataset implementation and image contents are unchanged.
The raw archive is retained. Images, environments and execution logs stay
outside Git.

The [verification snapshot](verification.json) records file hashes, versions,
source provenance and check results. Complete local receipts are
`datasets/celeba/preparation.json`, `datasets/celeba/source.json` and the files
under the operations directory above. To repeat the dataset audit:

```bash
python scripts/prepare_celeba.py --root /scratch/zz5070/ODE_FFN/datasets --verify-only
```

The CUDA result will be written to
`operations/torch-celeba-20261111/cuda-smoke.json` when the queued probe runs.
Its pending status in the original snapshot predates the completed H200
probe; the snapshot preserves the original preparation-time evidence.

## Current model scope

The selected candidate remains ISRU-D K8. The active image trainer now
supports native SwiGLU, ordinary Free ISRU and ISRU-D K8, plus the official
CelebA adapter. See the [training protocol](../../celeba-k8-20261111/protocol.md)
for the registered comparison and bounded priority allocations.
