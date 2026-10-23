#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ ! -x "$UV_BIN" ]]; then echo 'uv executable is required' >&2; exit 1; fi
# Remove every casing, including mixed-case inherited proxy variables.
for key in $(compgen -e); do
  if [[ "${key,,}" == *_proxy ]]; then unset "$key"; fi
done
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$ROOT_DIR/.cache/huggingface/datasets"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="$ROOT_DIR/.cache"
export TORCH_HOME="$ROOT_DIR/.cache/torch"
export TORCH_EXTENSIONS_DIR="$ROOT_DIR/.cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/.cache/torch_inductor"
export TRITON_CACHE_DIR="$ROOT_DIR/.cache/triton"
export CUDA_CACHE_PATH="$ROOT_DIR/.cache/cuda"
export UV_CACHE_DIR="$ROOT_DIR/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT_DIR/.cache/uv/python"
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_INDEX_URL="$UV_INDEX_URL"
export UV_HTTP_TIMEOUT=600 UV_LINK_MODE=hardlink
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS="${MAX_JOBS:-4}"
SOURCE="$ROOT_DIR/third_party/dreamdojo"
VENV="$ROOT_DIR/environment/world_models/dreamdojo"
NVIDIA_LIBS="$VENV/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$ROOT_DIR/scripts/setup/common.sh"
ensure_source "$SOURCE" https://github.com/NVIDIA/DreamDojo.git 02f119b759d5c7f84a399fdeea3c6e82e7ed6cff
if [[ ! -x "$VENV/bin/python" ]]; then "$UV_BIN" venv --python 3.10 "$VENV"; fi
"$UV_BIN" pip install --python "$VENV/bin/python" \
  --index-url "$UV_INDEX_URL" \
  --overrides "$ROOT_DIR/environment/requirements/dreamdojo.txt" \
  -r "$ROOT_DIR/environment/requirements/dreamdojo.txt"
# Upstream cosmos-oss declares many training, moderation and cloud dependencies.
# Install its source without expanding those unrelated dependency groups.
"$UV_BIN" pip install --python "$VENV/bin/python" --no-deps --no-sources \
  "$SOURCE/packages/cosmos-oss" "$SOURCE/packages/cosmos-cuda" "$SOURCE"
# Official config imports GR1 transforms even during inference. Only the
# pure-Python transforms are required; no PyTorch3D rendering/CUDA ops are used.
PYTORCH3D_NO_EXTENSION=1 "$UV_BIN" pip install --python "$VENV/bin/python" \
  --no-build-isolation --no-deps \
  'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@89653419d0973396f3eff1a381ba09a07fffc2ed'
# CUDA extension installation is explicit: do not silently call this runtime ready.
if [[ "${1:-}" == --cuda-extensions ]]; then
  # PyPI cuDNN ships headers in site-packages rather than /usr/local/cuda.
  CUDNN_INCLUDE="$VENV/lib/python3.10/site-packages/nvidia/cudnn/include"
  export CPATH="$CUDNN_INCLUDE${CPATH:+:$CPATH}"
  export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=80 TORCH_CUDA_ARCH_LIST=8.0
  "$UV_BIN" pip install --python "$VENV/bin/python" --no-build-isolation \
    --overrides "$ROOT_DIR/environment/requirements/dreamdojo.txt" \
    'transformer-engine[pytorch]==2.2.0'
  # Inference-only build via upstream's own compile guard. Use an isolated
  # cache so no other model can pick up a backward-disabled wheel by accident.
  UV_CACHE_DIR="$ROOT_DIR/.cache/uv/dreamdojo-flash-inference" \
  FLASH_ATTN_LOCAL_VERSION=dreamdojo.inference \
  CFLAGS="${CFLAGS:-} -DFLASHATTENTION_DISABLE_BACKWARD -DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_ALIBI -DFLASHATTENTION_DISABLE_SOFTCAP" \
  CXXFLAGS="${CXXFLAGS:-} -DFLASHATTENTION_DISABLE_BACKWARD -DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_ALIBI -DFLASHATTENTION_DISABLE_SOFTCAP" \
  NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -DFLASHATTENTION_DISABLE_BACKWARD -DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_ALIBI -DFLASHATTENTION_DISABLE_SOFTCAP" \
    "$UV_BIN" pip install --python "$VENV/bin/python" --no-build-isolation --no-deps 'flash-attn==2.7.3'
fi
cd "$ROOT_DIR"
PYTHONPATH="$ROOT_DIR:$SOURCE" "$VENV/bin/python" -c \
  'import torch; print("Torch:", torch.__version__, "CUDA:", torch.version.cuda); from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference; print("Official runtime import passed; inference NOT performed")'
