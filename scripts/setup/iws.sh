#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ ! -x "$UV_BIN" ]]; then echo 'Set UV_BIN to an executable uv' >&2; exit 1; fi
for variable in $(compgen -e); do
  if [[ "${variable,,}" == *_proxy ]]; then unset "$variable"; fi
done
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$ROOT/.cache/huggingface" HF_HUB_DISABLE_XET=1
export UV_CACHE_DIR="$ROOT/.uv-cache" UV_PYTHON_INSTALL_DIR="$ROOT/.cache/uv/python"
export XDG_CACHE_HOME="$ROOT/.cache" TORCH_HOME="$ROOT/.cache/torch"
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=600
SOURCE="$ROOT/third_party/iws"
REVISION=3ba69b41d070d7a56e79c287da5b24a7d90742aa
ROOT_DIR="$ROOT"
source "$ROOT/scripts/setup/common.sh"
ensure_source "$SOURCE" https://github.com/WangYixuan12/interactive_world_sim.git "$REVISION"
FK_SOURCE="$ROOT/third_party/iws_dependencies/yixuan_utilities"
URDF_SOURCE="$ROOT/third_party/iws_dependencies/urdfpy"
ensure_source "$FK_SOURCE" https://github.com/WangYixuan12/yixuan_utilities.git 805ea8bbb58eb2bf727d05970b7e287c99b82b42
ensure_source "$URDF_SOURCE" https://github.com/WangYixuan12/urdfpy.git 90737eab7af4399c97b23cda81152994c07a35cd
VENV_DIR="$ROOT/environment/world_models/iws"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then "$UV_BIN" venv --python 3.11 "$VENV_DIR"; fi
"$UV_BIN" pip install --python "$VENV_DIR/bin/python" -r "$ROOT/environment/requirements/iws.txt"
"$UV_BIN" pip install --python "$VENV_DIR/bin/python" --no-deps -e "$SOURCE"
"$UV_BIN" pip install --python "$VENV_DIR/bin/python" --no-deps --no-build-isolation -e "$URDF_SOURCE" -e "$FK_SOURCE"
"$VENV_DIR/bin/python" "$ROOT/scripts/download/iws.py" --root "$ROOT"
CUDA_VISIBLE_DEVICES="" "$VENV_DIR/bin/python" -c 'import torch; from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import LatentWorldModel; assert not torch.cuda.is_initialized(); print("IWS CPU import passed", torch.__version__, torch.version.cuda)'
CUDA_VISIBLE_DEVICES="" "$VENV_DIR/bin/python" -c 'import torch; from yixuan_utilities.kinematics_helper import KinHelper; kin = KinHelper("trossen_vx300s"); assert not torch.cuda.is_initialized(); print("IWS official CPU FK ready", kin.eef_name)'
