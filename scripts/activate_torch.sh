# Source this file in a torch shell: source scripts/activate_torch.sh
ODE_FFN_ROOT=/scratch/zz5070/ODE_FFN
source "$ODE_FFN_ROOT/envs/jit-celeba-cu128/bin/activate"
export PIP_CACHE_DIR="$ODE_FFN_ROOT/cache/pip"
export TORCH_HOME="$ODE_FFN_ROOT/cache/torch"
export HF_HOME="$ODE_FFN_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$ODE_FFN_ROOT/cache"
export TORCHINDUCTOR_CACHE_DIR="$ODE_FFN_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$ODE_FFN_ROOT/cache/triton"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2
