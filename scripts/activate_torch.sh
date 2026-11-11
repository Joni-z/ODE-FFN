# Source this file in a torch shell: source scripts/activate_torch.sh
ODE_FFN_ROOT=/scratch/zz5070/ODE_FFN
source /scratch/zz5070/miniconda3/etc/profile.d/conda.sh
conda activate "$ODE_FFN_ROOT/envs/jit-celeba"
export PIP_CACHE_DIR="$ODE_FFN_ROOT/cache/pip"
export TORCH_HOME="$ODE_FFN_ROOT/cache/torch"
export HF_HOME="$ODE_FFN_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$ODE_FFN_ROOT/cache"
export TORCHINDUCTOR_CACHE_DIR="$ODE_FFN_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$ODE_FFN_ROOT/cache/triton"
