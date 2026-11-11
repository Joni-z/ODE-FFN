# torch development and CelebA setup

11 November 2026. The user explicitly moves new development to torch and
authorizes dataset/environment preparation under scratch. No training is
launched by this setup.

## Paths

| Purpose | Path on torch |
|---|---|
| Active development checkout | `/scratch/zz5070/ODE_FFN/ODE-FFN` |
| New CUDA environment | `/scratch/zz5070/ODE_FFN/envs/jit-celeba` |
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

Use Python 3.10 and [requirements-cuda.txt](../../../requirements-cuda.txt),
which follows the pinned author's CUDA 12.4 / PyTorch 2.5.1 stack. CPU/MPS
requirements remain a separate profile. The author's torch-fidelity fork is
pinned to the existing project commit; nvitop is installed for monitoring.

Environment installation and import checks are pending. Setup commands and
the measured package versions will be recorded after completion.

## Dataset

Prepare original **CelebA**, not CelebA-HQ. Use torchvision's official CelebA
download implementation and its Google Drive IDs/MD5s, pinned by torchvision
0.20.1. [Source implementation](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/datasets/celeba.py).

The [preparation wrapper](../../../scripts/prepare_celeba.py) invokes that
implementation, then checks all six official file MD5s, archive CRC, the
202,599 expected image filenames, official split counts and 32 fixed JPEG
decodes. It preserves original images and metadata. Training resolution,
cropping and final evaluation reference are separate protocol decisions.

Expected split counts are 162,770 train, 19,867 validation and 19,962 test.
The raw archive and extracted images stay outside Git. Download and full
verification are pending.

## Current model scope

The selected candidate remains ISRU-D K8. The active image trainer currently
supports native SwiGLU and ordinary Free ISRU; K8 and a CelebA training adapter
have not been implemented by this infrastructure task. A successful data
loader/import check is not an image-quality or baseline result.
